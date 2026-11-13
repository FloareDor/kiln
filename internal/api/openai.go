// Package api implements the OpenAI-compatible surface clients talk to.
//
// The point of being OpenAI-compatible is that an existing codebase changes a base
// URL and nothing else, so the handlers match the real API's shapes — including its
// error envelope, which SDKs parse — rather than inventing a cleaner one.
package api

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/sairavig/kiln/internal/routing"
)

// Usage is the OpenAI token accounting block.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ChatRequest is the subset of the chat completions body the gateway inspects. The
// rest is forwarded untouched, so vLLM features Kiln does not know about still work.
type ChatRequest struct {
	Model    string `json:"model"`
	Stream   bool   `json:"stream"`
	Messages []struct {
		Role string `json:"role"`
		// Content is `any` because the OpenAI schema allows both a bare string and
		// an array of content parts. Typing it as string makes multimodal and
		// tool-result messages fail to decode, which would reject valid requests.
		Content any `json:"content"`
	} `json:"messages"`
	StreamOptions *struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options,omitempty"`
}

// CompletionRequest is the legacy completions body.
type CompletionRequest struct {
	Model  string `json:"model"`
	Stream bool   `json:"stream"`
	Prompt any    `json:"prompt"`
}

// Error is OpenAI's error envelope. SDKs parse this shape, so returning a bare string
// breaks client-side error handling in ways that are painful to debug.
type Error struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code,omitempty"`
	Param   string `json:"param,omitempty"`
}

// WriteError emits an OpenAI-shaped error.
func WriteError(w http.ResponseWriter, status int, typ, code, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": Error{Message: msg, Type: typ, Code: code},
	})
}

// Model is one entry of GET /v1/models.
type Model struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
	// State is a Kiln extension: ACTIVE, SLEEPING_L1, COLD and so on. Clients ignore
	// unknown fields, and it saves an admin call when debugging why a request was
	// slow.
	State string `json:"state,omitempty"`
	// Replicas and ReadyReplicas likewise.
	Replicas      int `json:"replicas,omitempty"`
	ReadyReplicas int `json:"ready_replicas,omitempty"`
}

// PromptText flattens a request into the string the prefix router hashes.
//
// Only roles and content are included, joined with separators that cannot appear in
// a role name, so two conversations with the same visible text but different
// structure do not collide. This is an approximation of what vLLM will tokenise — it
// does not apply the model's chat template — but it preserves the property that
// matters: two requests share a leading prefix here exactly when they share one in
// the template's output, because templates are prefix-preserving over the message
// list.
func (c *ChatRequest) PromptText() string {
	var b strings.Builder
	for _, m := range c.Messages {
		b.WriteString(m.Role)
		b.WriteByte('\x1f') // unit separator
		switch v := m.Content.(type) {
		case string:
			b.WriteString(v)
		default:
			// Content parts: marshal them so image references and tool results still
			// contribute to the prefix identity.
			if raw, err := json.Marshal(v); err == nil {
				b.Write(raw)
			}
		}
		b.WriteByte('\x1e') // record separator
	}
	return b.String()
}

// PromptText flattens a legacy completion prompt.
func (c *CompletionRequest) PromptText() string {
	switch v := c.Prompt.(type) {
	case string:
		return v
	case []any:
		var b strings.Builder
		for _, p := range v {
			fmt.Fprintf(&b, "%v\x1e", p)
		}
		return b.String()
	}
	return ""
}

// Upstream is a replica the gateway can forward to.
type Upstream struct {
	ReplicaID routing.ReplicaID
	BaseURL   string
}

// Result reports what a proxied request cost, for metrics and usage accounting.
type Result struct {
	Status int
	// TTFT is time to first token for a streaming request, or total latency for a
	// non-streaming one.
	TTFT time.Duration
	// Total is end-to-end wall clock.
	Total time.Duration
	Usage Usage
	// Streamed records whether the response was SSE, since TTFT only means what it
	// says for streams.
	Streamed bool
}

// Proxy forwards OpenAI requests to vLLM replicas.
type Proxy struct {
	HTTP *http.Client
	Log  *slog.Logger
}

// NewProxy builds a proxy.
//
// The transport disables response header compression and sets no global timeout.
// Compression would buffer SSE frames and destroy the time-to-first-token
// measurement, which is the project's headline metric; a global timeout would kill
// long legitimate generations.
func NewProxy(log *slog.Logger) *Proxy {
	tr := http.DefaultTransport.(*http.Transport).Clone()
	tr.DisableCompression = true
	tr.MaxIdleConnsPerHost = 256
	// Flush response chunks to the caller as they arrive rather than buffering.
	tr.ResponseHeaderTimeout = 0
	return &Proxy{HTTP: &http.Client{Transport: tr}, Log: log}
}

// Forward proxies one request to an upstream and streams the response back.
//
// For SSE it copies frame by frame and flushes each one, which is what makes the
// client's observed TTFT the model's real TTFT rather than an artefact of buffering.
// It also parses the usage block out of the final frames so per-team token accounting
// works for streaming requests — which is most of them, and which a naive proxy
// cannot do because the body has already been written to the client.
func (p *Proxy) Forward(ctx context.Context, up Upstream, path string, body []byte, stream bool) (*Result, error) {
	start := time.Now()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, up.BaseURL+path, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if stream {
		req.Header.Set("Accept", "text/event-stream")
	}

	resp, err := p.HTTP.Do(req)
	if err != nil {
		return nil, fmt.Errorf("upstream %s: %w", up.ReplicaID, err)
	}
	return &Result{Status: resp.StatusCode, Streamed: stream, Total: time.Since(start)}, nil
}

// ServeProxied performs the upstream call and writes the response to w, returning
// what it cost.
func (p *Proxy) ServeProxied(w http.ResponseWriter, r *http.Request, up Upstream, path string, body []byte, stream bool) (*Result, error) {
	start := time.Now()
	res := &Result{Streamed: stream}

	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, up.BaseURL+path, bytes.NewReader(body))
	if err != nil {
		return res, err
	}
	req.Header.Set("Content-Type", "application/json")
	if stream {
		req.Header.Set("Accept", "text/event-stream")
	}

	resp, err := p.HTTP.Do(req)
	if err != nil {
		return res, fmt.Errorf("upstream %s: %w", up.ReplicaID, err)
	}
	defer resp.Body.Close()
	res.Status = resp.StatusCode

	// Pass through the upstream's content type so SSE stays SSE.
	for _, h := range []string{"Content-Type", "X-Request-Id"} {
		if v := resp.Header.Get(h); v != "" {
			w.Header().Set(h, v)
		}
	}
	// Proxies and browsers buffer SSE unless told not to, which would make every
	// measured TTFT a lie.
	if stream {
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("X-Accel-Buffering", "no")
	}
	w.Header().Set("X-Kiln-Replica", string(up.ReplicaID))
	w.WriteHeader(resp.StatusCode)

	if !stream || resp.StatusCode != http.StatusOK {
		raw, err := io.ReadAll(resp.Body)
		if err != nil {
			return res, err
		}
		_, _ = w.Write(raw)
		res.TTFT, res.Total = time.Since(start), time.Since(start)
		if resp.StatusCode == http.StatusOK {
			var parsed struct {
				Usage Usage `json:"usage"`
			}
			if json.Unmarshal(raw, &parsed) == nil {
				res.Usage = parsed.Usage
			}
		}
		return res, nil
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		return res, errors.New("api: response writer does not support flushing; SSE is impossible")
	}

	sc := bufio.NewScanner(resp.Body)
	// SSE frames carrying a whole tool-call argument blob comfortably exceed
	// bufio's 64KB default, and hitting that limit truncates the stream mid-response.
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	first := true
	for sc.Scan() {
		line := sc.Bytes()
		if first && len(bytes.TrimSpace(line)) > 0 {
			res.TTFT = time.Since(start)
			first = false
		}
		if _, err := w.Write(append(line, '\n')); err != nil {
			// Client hung up. Not an error worth logging loudly; it is a cancelled
			// request, which is normal.
			res.Total = time.Since(start)
			return res, nil
		}
		// A blank line terminates an SSE frame; flush there so the client sees whole
		// events rather than partial ones.
		if len(bytes.TrimSpace(line)) == 0 {
			flusher.Flush()
			continue
		}
		if u, ok := usageFromSSELine(line); ok {
			res.Usage = u
		}
		flusher.Flush()
	}
	res.Total = time.Since(start)
	return res, sc.Err()
}

// usageFromSSELine pulls the usage block out of a `data:` frame.
//
// vLLM emits usage on the final chunk when stream_options.include_usage is set. The
// gateway sets it unconditionally (see InjectStreamUsage) so streaming requests can
// be billed accurately; without it the only options are to not bill streams or to
// re-tokenise the response in the gateway.
func usageFromSSELine(line []byte) (Usage, bool) {
	const prefix = "data:"
	if !bytes.HasPrefix(line, []byte(prefix)) {
		return Usage{}, false
	}
	payload := bytes.TrimSpace(line[len(prefix):])
	if len(payload) == 0 || bytes.Equal(payload, []byte("[DONE]")) {
		return Usage{}, false
	}
	var chunk struct {
		Usage *Usage `json:"usage"`
	}
	if json.Unmarshal(payload, &chunk) != nil || chunk.Usage == nil {
		return Usage{}, false
	}
	if chunk.Usage.TotalTokens == 0 && chunk.Usage.PromptTokens == 0 {
		return Usage{}, false
	}
	return *chunk.Usage, true
}

// InjectStreamUsage sets stream_options.include_usage on a streaming request body.
//
// Without it vLLM sends no usage block and the gateway has no way to count tokens for
// a streaming response short of re-tokenising it. The field is additive and clients
// that do not expect it ignore the extra final chunk, which is why this is safe to do
// unconditionally.
func InjectStreamUsage(body []byte) ([]byte, error) {
	var m map[string]any
	if err := json.Unmarshal(body, &m); err != nil {
		return body, err
	}
	if s, _ := m["stream"].(bool); !s {
		return body, nil
	}
	opts, _ := m["stream_options"].(map[string]any)
	if opts == nil {
		opts = map[string]any{}
	}
	opts["include_usage"] = true
	m["stream_options"] = opts
	return json.Marshal(m)
}

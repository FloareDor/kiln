package api

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/FloareDor/kiln/internal/routing"
)

func discardLog() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

// fakeVLLM stands in for a vLLM OpenAI server. The whole test suite runs against it,
// which is what keeps the gateway tests offline, fast and GPU-free — a suite that
// needs a GPU is a suite that stops being run.
type fakeVLLM struct {
	*httptest.Server
	id string
	// chunkDelay is inserted before the first SSE frame to simulate prefill, so TTFT
	// is a measurable quantity rather than noise.
	chunkDelay time.Duration
	requests   atomic.Int64
	// lastBody records what the gateway actually forwarded, used to prove
	// stream_options was injected.
	mu       sync.Mutex
	lastBody map[string]any
}

func newFakeVLLM(id string, chunkDelay time.Duration) *fakeVLLM {
	f := &fakeVLLM{id: id, chunkDelay: chunkDelay}
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", f.handle)
	mux.HandleFunc("/v1/completions", f.handle)
	f.Server = httptest.NewServer(mux)
	return f
}

func (f *fakeVLLM) handle(w http.ResponseWriter, r *http.Request) {
	f.requests.Add(1)
	raw, _ := io.ReadAll(r.Body)
	var body map[string]any
	_ = json.Unmarshal(raw, &body)
	f.mu.Lock()
	f.lastBody = body
	f.mu.Unlock()

	stream, _ := body["stream"].(bool)
	usage := map[string]any{"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}

	if !stream {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id": "cmpl-" + f.id, "object": "chat.completion",
			"choices": []any{map[string]any{"index": 0,
				"message": map[string]any{"role": "assistant", "content": "served by " + f.id}}},
			"usage": usage,
		})
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.WriteHeader(http.StatusOK)
	fl := w.(http.Flusher)
	time.Sleep(f.chunkDelay)

	for i, tok := range []string{"served", " by ", f.id} {
		chunk := map[string]any{
			"id": "cmpl-" + f.id, "object": "chat.completion.chunk",
			"choices": []any{map[string]any{"index": 0,
				"delta": map[string]any{"content": tok}}},
		}
		b, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "data: %s\n\n", b)
		fl.Flush()
		if i == 0 {
			time.Sleep(2 * time.Millisecond)
		}
	}
	// vLLM emits usage on a final chunk only when stream_options.include_usage is
	// set. Mirroring that is what makes the injection test meaningful.
	if opts, ok := body["stream_options"].(map[string]any); ok {
		if inc, _ := opts["include_usage"].(bool); inc {
			b, _ := json.Marshal(map[string]any{
				"id": "cmpl-" + f.id, "object": "chat.completion.chunk",
				"choices": []any{}, "usage": usage,
			})
			fmt.Fprintf(w, "data: %s\n\n", b)
			fl.Flush()
		}
	}
	fmt.Fprint(w, "data: [DONE]\n\n")
	fl.Flush()
}

type recorder struct {
	mu      sync.Mutex
	results []*Result
	errs    []error
	teams   []string
	picks   []routing.ReplicaID
	waits   []time.Duration
}

func (r *recorder) RequestDone(model, team string, rep routing.ReplicaID, res *Result, err error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.results = append(r.results, res)
	r.errs = append(r.errs, err)
	r.teams = append(r.teams, team)
	r.picks = append(r.picks, rep)
}

func (r *recorder) QueueWait(_ string, d time.Duration) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.waits = append(r.waits, d)
}

func (r *recorder) last() (*Result, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	n := len(r.results)
	return r.results[n-1], r.errs[n-1]
}

type harness struct {
	gw    *httptest.Server
	fakes []*fakeVLLM
	rec   *recorder
	reg   *StaticRegistry
}

func newHarness(t *testing.T, n int, router routing.Router, delay time.Duration) *harness {
	t.Helper()
	h := &harness{rec: &recorder{}}
	var reps []Replica
	for i := 0; i < n; i++ {
		id := fmt.Sprintf("r%d", i)
		f := newFakeVLLM(id, delay)
		h.fakes = append(h.fakes, f)
		reps = append(reps, Replica{
			ID: routing.ReplicaID(id), Model: "demo", BaseURL: f.URL, State: "ACTIVE", Ready: true,
		})
	}
	h.reg = NewStaticRegistry(reps)
	srv := NewServer(h.reg, router, DefaultConfig(), discardLog(), h.rec)
	h.gw = httptest.NewServer(srv.Routes())
	t.Cleanup(func() {
		h.gw.Close()
		for _, f := range h.fakes {
			f.Close()
		}
	})
	return h
}

func (h *harness) post(t *testing.T, path, body string) *http.Response {
	t.Helper()
	resp, err := http.Post(h.gw.URL+path, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST %s: %v", path, err)
	}
	return resp
}

// ------------------------------------------------------------------- basics

func TestNonStreamingChatCompletion(t *testing.T) {
	h := newHarness(t, 1, routing.NewRoundRobin(), 0)
	resp := h.post(t, "/v1/chat/completions",
		`{"model":"demo","messages":[{"role":"user","content":"hi"}]}`)
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status %s: %s", resp.Status, b)
	}
	if got := resp.Header.Get("X-Kiln-Replica"); got != "r0" {
		t.Fatalf("X-Kiln-Replica = %q, want r0", got)
	}
	res, err := h.rec.last()
	if err != nil {
		t.Fatal(err)
	}
	// Usage must be captured even for non-streaming, or per-team billing is blind.
	if res.Usage.TotalTokens != 14 {
		t.Fatalf("usage = %+v, want 14 total tokens", res.Usage)
	}
}

func TestStreamingPassesFramesThroughAndRecordsTTFT(t *testing.T) {
	// The 60ms delay stands in for prefill. If the proxy buffered the response, TTFT
	// would come out at roughly the total duration instead of the prefill time, and
	// the project's headline metric would be meaningless.
	h := newHarness(t, 1, routing.NewRoundRobin(), 60*time.Millisecond)
	resp := h.post(t, "/v1/chat/completions",
		`{"model":"demo","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	defer resp.Body.Close()

	if ct := resp.Header.Get("Content-Type"); !strings.Contains(ct, "text/event-stream") {
		t.Fatalf("Content-Type = %q, want text/event-stream", ct)
	}

	var frames []string
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		if line := strings.TrimSpace(sc.Text()); strings.HasPrefix(line, "data:") {
			frames = append(frames, line)
		}
	}
	if len(frames) < 4 {
		t.Fatalf("got %d data frames, want at least 4: %v", len(frames), frames)
	}
	if frames[len(frames)-1] != "data: [DONE]" {
		t.Fatalf("stream did not terminate with [DONE]: %q", frames[len(frames)-1])
	}

	res, err := h.rec.last()
	if err != nil {
		t.Fatal(err)
	}
	if res.TTFT < 50*time.Millisecond {
		t.Fatalf("TTFT = %s, want >= 50ms; the response was buffered", res.TTFT)
	}
	if res.TTFT > res.Total {
		t.Fatalf("TTFT %s exceeds total %s", res.TTFT, res.Total)
	}
}

func TestStreamingUsageIsCapturedForBilling(t *testing.T) {
	// Most real traffic streams. Without injecting stream_options.include_usage the
	// gateway sees no token counts at all and per-team quotas silently stop working
	// on exactly the requests that matter.
	h := newHarness(t, 1, routing.NewRoundRobin(), 0)
	resp := h.post(t, "/v1/chat/completions",
		`{"model":"demo","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()

	h.fakes[0].mu.Lock()
	opts, _ := h.fakes[0].lastBody["stream_options"].(map[string]any)
	h.fakes[0].mu.Unlock()
	if opts == nil || opts["include_usage"] != true {
		t.Fatalf("gateway did not inject stream_options.include_usage: %v", opts)
	}

	res, _ := h.rec.last()
	if res.Usage.CompletionTokens != 3 || res.Usage.PromptTokens != 11 {
		t.Fatalf("streaming usage = %+v, want 11/3", res.Usage)
	}
}

func TestInjectStreamUsageLeavesNonStreamingBodiesAlone(t *testing.T) {
	in := []byte(`{"model":"m","stream":false}`)
	out, err := InjectStreamUsage(in)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(in, out) {
		t.Fatalf("non-streaming body was modified: %s", out)
	}
}

func TestInjectStreamUsagePreservesExistingOptions(t *testing.T) {
	out, err := InjectStreamUsage([]byte(`{"model":"m","stream":true,"stream_options":{"something":1}}`))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	_ = json.Unmarshal(out, &m)
	opts := m["stream_options"].(map[string]any)
	if opts["include_usage"] != true || opts["something"] == nil {
		t.Fatalf("existing stream_options were clobbered: %v", opts)
	}
}

// --------------------------------------------------------------------- errors

func TestUnknownModelIsRejectedWithOpenAIErrorShape(t *testing.T) {
	h := newHarness(t, 1, routing.NewRoundRobin(), 0)
	resp := h.post(t, "/v1/chat/completions", `{"model":"nope","messages":[]}`)
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("status = %s, want 503", resp.Status)
	}
	// SDKs parse this envelope; a bare string breaks client error handling.
	var env struct {
		Error Error `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&env); err != nil {
		t.Fatal(err)
	}
	if env.Error.Message == "" || env.Error.Type == "" {
		t.Fatalf("error envelope missing fields: %+v", env.Error)
	}
}

func TestMissingModelFieldIsABadRequest(t *testing.T) {
	h := newHarness(t, 1, routing.NewRoundRobin(), 0)
	resp := h.post(t, "/v1/chat/completions", `{"messages":[]}`)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %s, want 400", resp.Status)
	}
}

func TestMalformedJSONIsABadRequestNotAPanic(t *testing.T) {
	h := newHarness(t, 1, routing.NewRoundRobin(), 0)
	resp := h.post(t, "/v1/chat/completions", `{"model":`)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %s, want 400", resp.Status)
	}
}

func TestSleepingReplicasAreNotRoutedTo(t *testing.T) {
	h := newHarness(t, 2, routing.NewRoundRobin(), 0)
	h.reg.SetReady("r0", false, "SLEEPING_L1")

	for i := 0; i < 10; i++ {
		resp := h.post(t, "/v1/chat/completions", `{"model":"demo","messages":[]}`)
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
	}
	if n := h.fakes[0].requests.Load(); n != 0 {
		t.Fatalf("sleeping replica received %d requests, want 0", n)
	}
	if n := h.fakes[1].requests.Load(); n != 10 {
		t.Fatalf("awake replica received %d requests, want 10", n)
	}
}

func TestAllReplicasAsleepReturns503(t *testing.T) {
	h := newHarness(t, 1, routing.NewRoundRobin(), 0)
	h.reg.SetReady("r0", false, "SLEEPING_L2")
	resp := h.post(t, "/v1/chat/completions", `{"model":"demo","messages":[]}`)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("status = %s, want 503", resp.Status)
	}
}

// -------------------------------------------------------------------- routing

func TestRequestsAreSpreadAcrossReplicas(t *testing.T) {
	h := newHarness(t, 3, routing.NewRoundRobin(), 0)
	for i := 0; i < 30; i++ {
		resp := h.post(t, "/v1/chat/completions", `{"model":"demo","messages":[]}`)
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
	}
	for i, f := range h.fakes {
		if n := f.requests.Load(); n != 10 {
			t.Fatalf("replica %d served %d of 30, want 10", i, n)
		}
	}
}

func TestListModelsReportsStateAndReadiness(t *testing.T) {
	h := newHarness(t, 2, routing.NewRoundRobin(), 0)
	h.reg.SetReady("r1", false, "SLEEPING_L1")

	resp, err := http.Get(h.gw.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var out struct {
		Object string  `json:"object"`
		Data   []Model `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if out.Object != "list" || len(out.Data) != 1 {
		t.Fatalf("unexpected body: %+v", out)
	}
	m := out.Data[0]
	if m.ID != "demo" || m.Replicas != 2 || m.ReadyReplicas != 1 {
		t.Fatalf("model entry = %+v, want demo with 1/2 ready", m)
	}
	if m.State != "ACTIVE" {
		t.Fatalf("state = %q; a model with one ACTIVE replica is servable", m.State)
	}
}

// ------------------------------------------------------------------ backpressure

func TestQueueFullShedsLoadWithRetryAfter(t *testing.T) {
	// A request that joins a full queue will time out anyway. Telling the client
	// immediately lets it retry elsewhere instead of holding a connection open.
	block := make(chan struct{})
	slow := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-block
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"usage":{"total_tokens":1}}`)
	}))
	defer slow.Close()

	reg := NewStaticRegistry([]Replica{{ID: "r0", Model: "demo", BaseURL: slow.URL, State: "ACTIVE", Ready: true}})
	cfg := DefaultConfig()
	cfg.MaxQueueDepth = 2
	srv := NewServer(reg, routing.NewRoundRobin(), cfg, discardLog(), &recorder{})
	gw := httptest.NewServer(srv.Routes())
	defer gw.Close()

	var wg sync.WaitGroup
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			resp, err := http.Post(gw.URL+"/v1/chat/completions", "application/json",
				strings.NewReader(`{"model":"demo","messages":[]}`))
			if err == nil {
				io.Copy(io.Discard, resp.Body)
				resp.Body.Close()
			}
		}()
	}
	// Wait for both to be counted as queued before probing.
	deadline := time.Now().Add(3 * time.Second)
	for srv.QueueDepth("demo") < 2 && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}

	resp, err := http.Post(gw.URL+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"demo","messages":[]}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("status = %s, want 429", resp.Status)
	}
	if resp.Header.Get("Retry-After") == "" {
		t.Fatal("429 without Retry-After; clients need it to back off sensibly")
	}
	close(block)
	wg.Wait()
}

func TestInFlightIsTrackedAndReleased(t *testing.T) {
	h := newHarness(t, 1, routing.NewRoundRobin(), 0)
	srv := NewServer(h.reg, routing.NewRoundRobin(), DefaultConfig(), discardLog(), h.rec)
	gw := httptest.NewServer(srv.Routes())
	defer gw.Close()

	resp, err := http.Post(gw.URL+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"demo","messages":[]}`))
	if err != nil {
		t.Fatal(err)
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()

	// Leaking in-flight counts would make the scheduler believe a model can never be
	// drained, so it could never be evicted.
	if n := srv.InFlight("r0"); n != 0 {
		t.Fatalf("in-flight after completion = %d, want 0", n)
	}
	if d := srv.QueueDepth("demo"); d != 0 {
		t.Fatalf("queue depth after completion = %d, want 0", d)
	}
}

// --------------------------------------------------------------- prompt handling

func TestPromptTextHandlesContentPartArrays(t *testing.T) {
	// OpenAI allows content to be an array of parts. Typing it as a string makes
	// multimodal and tool-result messages fail to decode and rejects valid requests.
	body := `{"model":"demo","messages":[
		{"role":"system","content":"you are helpful"},
		{"role":"user","content":[{"type":"text","text":"describe this"}]}]}`
	var req ChatRequest
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatalf("content-part array failed to decode: %v", err)
	}
	got := req.PromptText()
	if !strings.Contains(got, "you are helpful") || !strings.Contains(got, "describe this") {
		t.Fatalf("PromptText dropped content: %q", got)
	}
}

func TestPromptTextDistinguishesRoleBoundaries(t *testing.T) {
	// Two conversations with the same visible characters but different structure must
	// not hash to the same prefix.
	var a, b ChatRequest
	_ = json.Unmarshal([]byte(`{"messages":[{"role":"user","content":"ab"}]}`), &a)
	_ = json.Unmarshal([]byte(`{"messages":[{"role":"user","content":"a"},{"role":"user","content":"b"}]}`), &b)
	if a.PromptText() == b.PromptText() {
		t.Fatal("different message structures flattened to the same prompt text")
	}
}

func TestCompletionsEndpointWorks(t *testing.T) {
	h := newHarness(t, 1, routing.NewRoundRobin(), 0)
	resp := h.post(t, "/v1/completions", `{"model":"demo","prompt":"once upon a time"}`)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %s, want 200", resp.Status)
	}
}

func TestEnsureIsCalledBeforeRouting(t *testing.T) {
	// Ensure is where the swap scheduler will wake a sleeping model. If routing ran
	// first, every request to a sleeping model would 503 instead of waiting for a
	// wake, and tiering would be unreachable from the request path.
	called := make(chan string, 1)
	reg := &ensureSpy{StaticRegistry: NewStaticRegistry([]Replica{
		{ID: "r0", Model: "demo", BaseURL: "http://127.0.0.1:1", State: "ACTIVE", Ready: true},
	}), called: called}

	srv := NewServer(reg, routing.NewRoundRobin(), DefaultConfig(), discardLog(), &recorder{})
	gw := httptest.NewServer(srv.Routes())
	defer gw.Close()

	resp, _ := http.Post(gw.URL+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"demo","messages":[]}`))
	if resp != nil {
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
	}
	select {
	case m := <-called:
		if m != "demo" {
			t.Fatalf("Ensure called with %q", m)
		}
	default:
		t.Fatal("Ensure was never called")
	}
}

type ensureSpy struct {
	*StaticRegistry
	called chan string
}

func (e *ensureSpy) Ensure(ctx context.Context, model string) error {
	select {
	case e.called <- model:
	default:
	}
	return e.StaticRegistry.Ensure(ctx, model)
}

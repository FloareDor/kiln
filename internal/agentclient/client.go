// Package agentclient is the control plane's HTTP client for node agents.
package agentclient

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/FloareDor/kiln/internal/nodeagent"
)

// Client talks to one node agent.
type Client struct {
	BaseURL string
	HTTP    *http.Client
}

// New builds a client. The transport has no global timeout because starting a model
// legitimately takes minutes; each call sets its own deadline instead.
func New(baseURL string) *Client {
	return &Client{
		BaseURL: strings.TrimRight(baseURL, "/"),
		HTTP:    &http.Client{},
	}
}

// Status fetches the node's GPUs, host memory and replicas in one round trip.
func (c *Client) Status(ctx context.Context) (nodeagent.NodeStatus, error) {
	var out nodeagent.NodeStatus
	err := c.do(ctx, http.MethodGet, "/status", nil, &out, 15*time.Second)
	return out, err
}

// Start brings up a replica. Idempotent on the agent side.
func (c *Client) Start(ctx context.Context, spec nodeagent.Spec) (nodeagent.Status, error) {
	var out nodeagent.Status
	err := c.do(ctx, http.MethodPost, "/replicas", spec, &out, 20*time.Minute)
	return out, err
}

// Sleep offloads a replica to the host (1) or disk (2) tier.
func (c *Client) Sleep(ctx context.Context, id string, level int, mode string) (nodeagent.Status, error) {
	var out nodeagent.Status
	body := map[string]any{"level": level, "mode": mode}
	err := c.do(ctx, http.MethodPost, "/replicas/"+url.PathEscape(id)+"/sleep", body, &out, 12*time.Minute)
	return out, err
}

// Wake restores a sleeping replica, including the weight reload a level-2 wake needs.
func (c *Client) Wake(ctx context.Context, id string) (nodeagent.Status, error) {
	var out nodeagent.Status
	err := c.do(ctx, http.MethodPost, "/replicas/"+url.PathEscape(id)+"/wake", nil, &out, 20*time.Minute)
	return out, err
}

// Stop terminates a replica.
func (c *Client) Stop(ctx context.Context, id string) error {
	return c.do(ctx, http.MethodDelete, "/replicas/"+url.PathEscape(id), nil, nil, 2*time.Minute)
}

func (c *Client) do(ctx context.Context, method, path string, in, out any, timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	var body io.Reader
	if in != nil {
		b, err := json.Marshal(in)
		if err != nil {
			return err
		}
		body = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.BaseURL+path, body)
	if err != nil {
		return err
	}
	if in != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := c.HTTP.Do(req)
	if err != nil {
		return fmt.Errorf("agentclient %s %s: %w", method, path, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 300 {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("agentclient %s %s: %s: %s", method, path, resp.Status, strings.TrimSpace(string(msg)))
	}
	if out == nil || resp.StatusCode == http.StatusNoContent {
		_, _ = io.Copy(io.Discard, resp.Body)
		return nil
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"sort"
	"sync"
	"time"

	"github.com/sairavig/kiln/internal/routing"
)

// Replica is everything the gateway knows about one servable replica.
type Replica struct {
	ID      routing.ReplicaID
	Model   string
	BaseURL string
	// State is the scheduler state string; only ACTIVE replicas are routable.
	State string
	Ready bool
}

// Registry supplies the current fleet. It is an interface so M1 can run off a static
// list from flags while the control plane and Postgres land behind it unchanged.
type Registry interface {
	// Replicas returns every replica of a model, routable or not.
	Replicas(model string) []Replica
	// Models lists deployed model ids.
	Models() []string
	// Ensure is called when a request arrives for a model that has no ready replica.
	// It is where the swap scheduler hooks in: it blocks until the model is servable
	// or the context expires. In M1 it is a no-op.
	Ensure(ctx context.Context, model string) error
}

// Observer receives per-request telemetry. The metrics package implements it; tests
// use a recording fake, which keeps the handlers free of metric plumbing.
type Observer interface {
	RequestDone(model string, team string, replica routing.ReplicaID, res *Result, err error)
	QueueWait(model string, d time.Duration)
}

type nopObserver struct{}

func (nopObserver) RequestDone(string, string, routing.ReplicaID, *Result, error) {}
func (nopObserver) QueueWait(string, time.Duration)                              {}

// Config tunes the gateway.
type Config struct {
	// MaxQueueDepth is how many requests may wait for one model before the gateway
	// sheds load. Unbounded queueing turns a short overload into a latency
	// catastrophe that outlives the overload itself, so the bound is the point.
	MaxQueueDepth int
	// QueueTimeout bounds how long a request waits for a model to become servable —
	// which, once swapping exists, includes waiting out a wake.
	QueueTimeout time.Duration
	// MaxConcurrentPerReplica caps in-flight requests per replica. vLLM batches
	// internally, so this is a backstop against unbounded memory growth rather than
	// a throughput control.
	MaxConcurrentPerReplica int
}

// DefaultConfig returns sensible defaults.
func DefaultConfig() Config {
	return Config{MaxQueueDepth: 256, QueueTimeout: 120 * time.Second, MaxConcurrentPerReplica: 64}
}

// Server is the gateway.
type Server struct {
	Reg    Registry
	Router routing.Router
	Proxy  *Proxy
	Cfg    Config
	Log    *slog.Logger
	Obs    Observer

	mu sync.Mutex
	// inflight and queued are per-replica and per-model counters the routers read.
	// They are the gateway's own view, not the agent's: a router must react to load
	// it created microseconds ago, long before any agent poll would reflect it.
	inflight map[routing.ReplicaID]int
	// inflightModel is the same count aggregated per model. The swap controller
	// drains a model, not a replica, and summing replicas on every drain poll would
	// mean walking the whole fleet every 25ms during an eviction.
	inflightModel map[string]int
	queued        map[string]int
	ttft          map[routing.ReplicaID]float64
}

// NewServer builds a gateway.
func NewServer(reg Registry, router routing.Router, cfg Config, log *slog.Logger, obs Observer) *Server {
	if obs == nil {
		obs = nopObserver{}
	}
	return &Server{
		Reg: reg, Router: router, Proxy: NewProxy(log), Cfg: cfg, Log: log, Obs: obs,
		inflight:      map[routing.ReplicaID]int{},
		inflightModel: map[string]int{},
		queued:        map[string]int{},
		ttft:          map[routing.ReplicaID]float64{},
	}
}

// Routes returns the HTTP surface. Auth and quota middleware wrap this in cmd/gateway.
func (s *Server) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"status":"ok"}`)
	})
	mux.HandleFunc("GET /v1/models", s.listModels)
	mux.HandleFunc("POST /v1/chat/completions", s.chatCompletions)
	mux.HandleFunc("POST /v1/completions", s.completions)
	return mux
}

func (s *Server) listModels(w http.ResponseWriter, r *http.Request) {
	ids := s.Reg.Models()
	sort.Strings(ids)
	out := make([]Model, 0, len(ids))
	for _, id := range ids {
		reps := s.Reg.Replicas(id)
		ready := 0
		state := "COLD"
		for _, rp := range reps {
			if rp.Ready {
				ready++
			}
			// Report the most available state across replicas: a model with one
			// ACTIVE replica is servable now, whatever the others are doing.
			if rp.State == "ACTIVE" || state == "COLD" {
				state = rp.State
			}
		}
		out = append(out, Model{
			ID: id, Object: "model", Created: time.Now().Unix(), OwnedBy: "kiln",
			State: state, Replicas: len(reps), ReadyReplicas: ready,
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": out})
}

func (s *Server) chatCompletions(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 32<<20))
	if err != nil {
		WriteError(w, http.StatusBadRequest, "invalid_request_error", "", "could not read request body: "+err.Error())
		return
	}
	var req ChatRequest
	if err := json.Unmarshal(body, &req); err != nil {
		WriteError(w, http.StatusBadRequest, "invalid_request_error", "", "malformed JSON: "+err.Error())
		return
	}
	s.serve(w, r, body, req.Model, req.Stream, req.PromptText(), "/v1/chat/completions")
}

func (s *Server) completions(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 32<<20))
	if err != nil {
		WriteError(w, http.StatusBadRequest, "invalid_request_error", "", "could not read request body: "+err.Error())
		return
	}
	var req CompletionRequest
	if err := json.Unmarshal(body, &req); err != nil {
		WriteError(w, http.StatusBadRequest, "invalid_request_error", "", "malformed JSON: "+err.Error())
		return
	}
	s.serve(w, r, body, req.Model, req.Stream, req.PromptText(), "/v1/completions")
}

// TeamFromContext is set by the auth middleware; empty when auth is disabled.
type teamKey struct{}

// WithTeam attaches a team id to the request context.
func WithTeam(ctx context.Context, team string) context.Context {
	return context.WithValue(ctx, teamKey{}, team)
}

// TeamFromContext reads the team id, or "" when unauthenticated.
func TeamFromContext(ctx context.Context) string {
	t, _ := ctx.Value(teamKey{}).(string)
	return t
}

func (s *Server) serve(w http.ResponseWriter, r *http.Request, body []byte, model string, stream bool, prompt, path string) {
	team := TeamFromContext(r.Context())
	if model == "" {
		WriteError(w, http.StatusBadRequest, "invalid_request_error", "missing_model", "the 'model' field is required")
		return
	}

	// Shed load before queueing rather than after. A request that joins a 256-deep
	// queue will time out anyway, and telling the client now lets it retry elsewhere
	// instead of holding a connection open for two minutes to be told no.
	s.mu.Lock()
	depth := s.queued[model]
	if depth >= s.Cfg.MaxQueueDepth {
		s.mu.Unlock()
		w.Header().Set("Retry-After", "1")
		WriteError(w, http.StatusTooManyRequests, "rate_limit_error", "queue_full",
			"queue for model "+model+" is full; retry shortly")
		s.Obs.RequestDone(model, team, "", &Result{Status: http.StatusTooManyRequests}, nil)
		return
	}
	s.queued[model]++
	s.mu.Unlock()

	queueStart := time.Now()
	defer func() {
		s.mu.Lock()
		s.queued[model]--
		s.mu.Unlock()
	}()

	// Ensure is where a sleeping or cold model gets woken. Everything from here to
	// the first token may legitimately take tens of seconds on a cold start, which is
	// precisely what the tiering work is meant to shrink.
	ctx, cancel := context.WithTimeout(r.Context(), s.Cfg.QueueTimeout)
	defer cancel()
	if err := s.Reg.Ensure(ctx, model); err != nil {
		status, code := http.StatusServiceUnavailable, "model_unavailable"
		if errors.Is(err, context.DeadlineExceeded) {
			status, code = http.StatusGatewayTimeout, "model_wake_timeout"
		}
		// Log the reason. A 503 with no explanation on the server side makes a
		// failing benchmark run unexplainable, which cost several investigations.
		s.Log.Warn("could not make model servable", "model", model, "status", status, "err", err)
		WriteError(w, status, "server_error", code, err.Error())
		s.Obs.RequestDone(model, team, "", &Result{Status: status}, err)
		return
	}
	s.Obs.QueueWait(model, time.Since(queueStart))

	up, replicaID, err := s.pick(model, prompt)
	if err != nil {
		s.Log.Warn("no routable replica after the model was made servable",
			"model", model, "err", err, "replicas", s.describeReplicas(model))
		WriteError(w, http.StatusServiceUnavailable, "server_error", "no_replica", err.Error())
		s.Obs.RequestDone(model, team, "", &Result{Status: http.StatusServiceUnavailable}, err)
		return
	}

	if stream {
		// Ask vLLM for a usage block on the final chunk. Without it, streaming
		// requests — which is most of them — cannot be billed without re-tokenising
		// the response in the gateway.
		if patched, err := InjectStreamUsage(body); err == nil {
			body = patched
		}
	}

	s.enter(replicaID, model)
	defer s.leave(replicaID, model)

	res, err := s.Proxy.ServeProxied(w, r.WithContext(ctx), up, path, body, stream)
	if err != nil {
		s.Log.Error("proxy failed", "model", model, "replica", replicaID, "err", err)
		// The status line may already have been written for a stream, in which case
		// the client sees a truncated response. Nothing better is possible once bytes
		// are on the wire; the error is recorded for the metrics instead.
		if res == nil || res.Status == 0 {
			WriteError(w, http.StatusBadGateway, "server_error", "upstream_error", err.Error())
		}
		s.Obs.RequestDone(model, team, replicaID, res, err)
		return
	}

	if res.TTFT > 0 {
		s.recordTTFT(replicaID, res.TTFT)
	}
	s.Router.Observe(routing.Request{Model: model, PrefixBlocks: blocks(prompt)}, replicaID)
	s.Obs.RequestDone(model, team, replicaID, res, nil)
}

// describeReplicas renders a model's replicas for a diagnostic log line: which exist,
// what state each is in, and whether the concurrency cap is what made it unroutable.
func (s *Server) describeReplicas(model string) []string {
	reps := s.Reg.Replicas(model)
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]string, 0, len(reps))
	for _, r := range reps {
		out = append(out, fmt.Sprintf("%s=%s ready=%t inflight=%d",
			r.ID, r.State, r.Ready, s.inflight[r.ID]))
	}
	if len(out) == 0 {
		return []string{"none registered"}
	}
	return out
}

func blocks(prompt string) []uint64 {
	return routing.BlockHashes(prompt, routing.DefaultBlockSize, 64)
}

// pick chooses a replica using the configured router.
func (s *Server) pick(model, prompt string) (Upstream, routing.ReplicaID, error) {
	reps := s.Reg.Replicas(model)
	if len(reps) == 0 {
		return Upstream{}, "", errors.New("no replicas deployed for model " + model)
	}

	s.mu.Lock()
	states := make([]routing.ReplicaState, 0, len(reps))
	byID := make(map[routing.ReplicaID]Replica, len(reps))
	for _, rp := range reps {
		byID[rp.ID] = rp
		states = append(states, routing.ReplicaState{
			ID: rp.ID, Model: rp.Model,
			// A replica at its concurrency cap is temporarily not routable, which
			// keeps one slow replica from accumulating the whole fleet's backlog.
			Ready:            rp.Ready && s.inflight[rp.ID] < s.Cfg.MaxConcurrentPerReplica,
			InFlight:         s.inflight[rp.ID],
			RecentTTFTMillis: s.ttft[rp.ID],
		})
	}
	s.mu.Unlock()

	id, err := s.Router.Pick(routing.Request{Model: model, PrefixBlocks: blocks(prompt)}, states)
	if err != nil {
		return Upstream{}, "", err
	}
	rp := byID[id]
	return Upstream{ReplicaID: id, BaseURL: rp.BaseURL}, id, nil
}

func (s *Server) enter(id routing.ReplicaID, model string) {
	s.mu.Lock()
	s.inflight[id]++
	s.inflightModel[model]++
	s.mu.Unlock()
}

func (s *Server) leave(id routing.ReplicaID, model string) {
	s.mu.Lock()
	if s.inflight[id] > 0 {
		s.inflight[id]--
	}
	if s.inflightModel[model] > 0 {
		s.inflightModel[model]--
	}
	s.mu.Unlock()
}

// recordTTFT keeps a decayed per-replica TTFT for the least-loaded tiebreak. A simple
// EWMA: one slow request should nudge the estimate, not redefine it.
func (s *Server) recordTTFT(id routing.ReplicaID, d time.Duration) {
	const alpha = 0.2
	s.mu.Lock()
	prev := s.ttft[id]
	ms := float64(d.Milliseconds())
	if prev == 0 {
		s.ttft[id] = ms
	} else {
		s.ttft[id] = prev*(1-alpha) + ms*alpha
	}
	s.mu.Unlock()
}

// InFlight exposes per-replica load for the control plane's drain check: a model is
// safe to sleep only when the gateway reports zero in-flight requests for it.
func (s *Server) InFlight(id routing.ReplicaID) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.inflight[id]
}

// InFlightForModel is the swap controller's drain signal: a model is safe to sleep
// only once this reaches zero.
func (s *Server) InFlightForModel(model string) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.inflightModel[model]
}

// QueueDepth exposes a model's current queue depth for metrics and the scheduler.
func (s *Server) QueueDepth(model string) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.queued[model]
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

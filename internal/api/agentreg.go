package api

import (
	"context"
	"fmt"
	"log/slog"
	"sort"
	"sync"
	"time"

	"github.com/FloareDor/kiln/internal/agentclient"
	"github.com/FloareDor/kiln/internal/routing"
)

// AgentRegistry builds the gateway's fleet view by polling node agents.
//
// This is the M1 registry: it discovers whatever replicas the agents happen to be
// running, with no database and no control plane. Later milestones put the control
// plane behind the same interface, which is why Ensure is a hook rather than a no-op
// baked into the handlers.
//
// Polling rather than push, deliberately. The gateway must keep routing when the
// control plane is down — a serving platform whose data path depends on its control
// path is a platform that has outages its users do not understand. A poll is also
// trivially idempotent and self-healing after a partition.
type AgentRegistry struct {
	agents []*agentclient.Client
	log    *slog.Logger

	mu      sync.RWMutex
	byModel map[string][]Replica
	// agentFor maps a replica to the agent that owns it, so the swap executor knows
	// where to send sleep and wake calls.
	agentFor map[routing.ReplicaID]*agentclient.Client
	lastErr  error
}

// NewAgentRegistry builds a registry over the given agent base URLs.
func NewAgentRegistry(urls []string, log *slog.Logger) *AgentRegistry {
	r := &AgentRegistry{
		log:      log,
		byModel:  map[string][]Replica{},
		agentFor: map[routing.ReplicaID]*agentclient.Client{},
	}
	for _, u := range urls {
		r.agents = append(r.agents, agentclient.New(u))
	}
	return r
}

// Refresh polls every agent once and rebuilds the fleet view.
func (r *AgentRegistry) Refresh(ctx context.Context) error {
	next := map[string][]Replica{}
	owner := map[routing.ReplicaID]*agentclient.Client{}
	var firstErr error

	for _, a := range r.agents {
		st, err := a.Status(ctx)
		if err != nil {
			// One unreachable agent must not blank the whole fleet. Its replicas drop
			// out of this snapshot, which is correct — they are unreachable — while
			// every other agent's replicas keep serving.
			if firstErr == nil {
				firstErr = err
			}
			r.log.Warn("agent poll failed", "agent", a.BaseURL, "err", err)
			continue
		}
		for _, rep := range st.Replicas {
			id := routing.ReplicaID(rep.Spec.Replica)
			next[rep.Spec.Model] = append(next[rep.Spec.Model], Replica{
				ID: id, Model: rep.Spec.Model, BaseURL: rep.BaseURL,
				State: rep.State,
				// Only ACTIVE serves. STARTING, WAKING, DRAINING, SLEEPING_* and
				// FAILED all mean "do not send traffic here", for different reasons
				// that the router does not need to distinguish.
				Ready: rep.State == "ACTIVE",
			})
			owner[id] = a
		}
	}
	for m := range next {
		sort.Slice(next[m], func(i, j int) bool { return next[m][i].ID < next[m][j].ID })
	}

	r.mu.Lock()
	r.byModel, r.agentFor, r.lastErr = next, owner, firstErr
	r.mu.Unlock()
	return firstErr
}

// Run polls until the context is cancelled.
func (r *AgentRegistry) Run(ctx context.Context, every time.Duration) {
	if every <= 0 {
		every = 2 * time.Second
	}
	_ = r.Refresh(ctx)
	t := time.NewTicker(every)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			_ = r.Refresh(ctx)
		}
	}
}

// Replicas returns a model's replicas.
func (r *AgentRegistry) Replicas(model string) []Replica {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]Replica, len(r.byModel[model]))
	copy(out, r.byModel[model])
	return out
}

// Models lists known model ids.
func (r *AgentRegistry) Models() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]string, 0, len(r.byModel))
	for m := range r.byModel {
		out = append(out, m)
	}
	sort.Strings(out)
	return out
}

// AgentFor returns the agent owning a replica, used by the swap executor.
func (r *AgentRegistry) AgentFor(id routing.ReplicaID) (*agentclient.Client, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	a, ok := r.agentFor[id]
	return a, ok
}

// Ensure waits briefly for a ready replica, refreshing as it goes.
//
// In M1 this only covers the gap between an agent finishing a start and the next poll
// reflecting it — without it, a request arriving in that window gets a spurious 503
// for a model that is in fact up. From M5 the swap scheduler replaces this with a
// wake.
func (r *AgentRegistry) Ensure(ctx context.Context, model string) error {
	if r.ready(model) {
		return nil
	}
	tick := time.NewTicker(250 * time.Millisecond)
	defer tick.Stop()
	for {
		select {
		case <-ctx.Done():
			return fmt.Errorf("model %q did not become ready: %w", model, ctx.Err())
		case <-tick.C:
			_ = r.Refresh(ctx)
			if r.ready(model) {
				return nil
			}
		}
	}
}

func (r *AgentRegistry) ready(model string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	for _, rep := range r.byModel[model] {
		if rep.Ready {
			return true
		}
	}
	return false
}

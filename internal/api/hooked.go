package api

import "context"

// HookedRegistry wraps a Registry and replaces its Ensure.
//
// It exists to break a construction cycle: the gateway's HTTP server needs a Registry
// up front, while the swap controller needs the server's live in-flight counters to
// decide when a model is safe to drain. Rather than making either own the other, the
// server is built with a registry whose Ensure is installed afterwards.
//
// StateFn lets the swap controller override the state a replica is reported in. The
// controller is authoritative for residency — it is the thing performing the
// transitions — while the underlying registry only reports what an agent poll last
// saw, which lags a wake by up to a poll interval. Routing to a replica the
// controller is in the middle of draining is exactly the in-flight request the drain
// exists to avoid.
type HookedRegistry struct {
	Registry
	Ensurer func(ctx context.Context, model string) error
	StateFn func(model string) (string, bool)
}

// Ensure delegates to the installed hook, falling back to the wrapped registry.
func (h *HookedRegistry) Ensure(ctx context.Context, model string) error {
	if h.Ensurer != nil {
		return h.Ensurer(ctx, model)
	}
	return h.Registry.Ensure(ctx, model)
}

// Replicas overlays the controller's authoritative residency state.
func (h *HookedRegistry) Replicas(model string) []Replica {
	reps := h.Registry.Replicas(model)
	if h.StateFn == nil {
		return reps
	}
	state, ok := h.StateFn(model)
	if !ok {
		return reps
	}
	for i := range reps {
		reps[i].State = state
		reps[i].Ready = state == "ACTIVE"
	}
	return reps
}

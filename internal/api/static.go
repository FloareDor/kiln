package api

import (
	"context"
	"fmt"
	"sort"
	"sync"
)

// StaticRegistry is a fixed fleet supplied at startup. It is what M1 runs on, and
// what the gateway tests run against, so the HTTP layer can be exercised without a
// control plane, a database or a GPU.
type StaticRegistry struct {
	mu   sync.RWMutex
	reps map[string][]Replica
}

// NewStaticRegistry builds a registry from a replica list.
func NewStaticRegistry(reps []Replica) *StaticRegistry {
	s := &StaticRegistry{reps: map[string][]Replica{}}
	for _, r := range reps {
		s.reps[r.Model] = append(s.reps[r.Model], r)
	}
	return s
}

// Replicas returns a model's replicas.
func (s *StaticRegistry) Replicas(model string) []Replica {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]Replica, len(s.reps[model]))
	copy(out, s.reps[model])
	return out
}

// Models lists deployed model ids.
func (s *StaticRegistry) Models() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]string, 0, len(s.reps))
	for m := range s.reps {
		out = append(out, m)
	}
	sort.Strings(out)
	return out
}

// Ensure checks the model exists. A static fleet cannot wake anything, so an unready
// model is a hard failure here rather than a wait.
func (s *StaticRegistry) Ensure(_ context.Context, model string) error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	reps, ok := s.reps[model]
	if !ok || len(reps) == 0 {
		return fmt.Errorf("model %q is not deployed", model)
	}
	for _, r := range reps {
		if r.Ready {
			return nil
		}
	}
	return fmt.Errorf("model %q has no ready replica", model)
}

// SetReady flips a replica's readiness, used by the node-agent poller and by tests.
func (s *StaticRegistry) SetReady(id string, ready bool, state string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for model, reps := range s.reps {
		for i := range reps {
			if string(reps[i].ID) == id {
				s.reps[model][i].Ready = ready
				s.reps[model][i].State = state
			}
		}
	}
}

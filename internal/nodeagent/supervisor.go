package nodeagent

import (
	"context"
	"fmt"
	"sort"
	"sync"
)

// Supervisor owns every vLLM process on one GPU and serialises the operations that
// contend for GPU memory.
//
// The lock discipline is the point of this type. Two concurrent starts, or a start
// racing a wake, both allocate from the same device, and the second one fails at the
// worst possible moment — after a minute of loading. A single mutation lock turns that
// into a queue. Reads (Status, List, Get) take only the map lock, so the gateway's
// routing decisions are never blocked behind a 40-second cold start.
type Supervisor struct {
	cfg Config

	mu       sync.RWMutex
	replicas map[string]*Replica

	// gpuMu serialises every operation that changes GPU residency. It is deliberately
	// separate from mu and is never held while mu is held.
	gpuMu sync.Mutex
}

// NewSupervisor builds a supervisor.
func NewSupervisor(cfg Config) *Supervisor {
	if cfg.PythonBin == "" {
		cfg.PythonBin = "python"
	}
	if cfg.PortBase == 0 {
		cfg.PortBase = 19000
	}
	if cfg.StartTimeout == 0 {
		cfg.StartTimeout = DefaultConfig().StartTimeout
	}
	if cfg.LogDir == "" {
		cfg.LogDir = DefaultConfig().LogDir
	}
	return &Supervisor{cfg: cfg, replicas: make(map[string]*Replica)}
}

// Get returns a replica by id.
func (s *Supervisor) Get(id string) (*Replica, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	r, ok := s.replicas[id]
	return r, ok
}

// List returns every replica's status, ordered by id for stable output.
func (s *Supervisor) List() []Status {
	s.mu.RLock()
	out := make([]Status, 0, len(s.replicas))
	for _, r := range s.replicas {
		out = append(out, r.Status())
	}
	s.mu.RUnlock()
	sort.Slice(out, func(i, j int) bool { return out[i].Spec.Replica < out[j].Spec.Replica })
	return out
}

// Start creates the replica if needed and brings it up.
//
// It is idempotent: starting an already-ACTIVE replica returns its status without
// touching the GPU. Deploy jobs are retried after a worker crash, and a retry that
// restarted a healthy model would evict whatever had been placed next to it.
func (s *Supervisor) Start(ctx context.Context, spec Spec) (Status, error) {
	if spec.Replica == "" || spec.Model == "" {
		return Status{}, fmt.Errorf("nodeagent: spec needs replica and model")
	}

	s.mu.Lock()
	r, ok := s.replicas[spec.Replica]
	if !ok {
		port, err := freePort(s.cfg.PortBase)
		if err != nil {
			s.mu.Unlock()
			return Status{}, err
		}
		r = newReplica(spec, s.cfg, port)
		s.replicas[spec.Replica] = r
	}
	s.mu.Unlock()

	if r.State().Serving() {
		return r.Status(), nil
	}

	s.gpuMu.Lock()
	defer s.gpuMu.Unlock()
	// Re-check under the GPU lock: another caller may have started it while this one
	// waited, which is the common case when a burst of requests arrives for a model
	// that is not yet up.
	if r.State().Serving() {
		return r.Status(), nil
	}
	if err := r.Start(ctx); err != nil {
		return r.Status(), err
	}
	return r.Status(), nil
}

// Sleep offloads a replica to the host (level 1) or disk (level 2) tier.
func (s *Supervisor) Sleep(ctx context.Context, id string, level int, mode string) (Status, error) {
	r, ok := s.Get(id)
	if !ok {
		return Status{}, fmt.Errorf("nodeagent: unknown replica %q", id)
	}
	s.gpuMu.Lock()
	defer s.gpuMu.Unlock()
	if err := r.Sleep(ctx, level, mode); err != nil {
		return r.Status(), err
	}
	return r.Status(), nil
}

// Wake restores a sleeping replica.
func (s *Supervisor) Wake(ctx context.Context, id string) (Status, error) {
	r, ok := s.Get(id)
	if !ok {
		return Status{}, fmt.Errorf("nodeagent: unknown replica %q", id)
	}
	// Already serving: a burst of requests for one sleeping model arrives here
	// together, and only the first should issue a wake. The scheduler coalesces at
	// the plan level; this is the backstop for the race between planning and
	// executing.
	if r.State().Serving() {
		return r.Status(), nil
	}
	s.gpuMu.Lock()
	defer s.gpuMu.Unlock()
	if r.State().Serving() {
		return r.Status(), nil
	}
	if err := r.Wake(ctx); err != nil {
		return r.Status(), err
	}
	return r.Status(), nil
}

// Stop terminates a replica and forgets it.
func (s *Supervisor) Stop(ctx context.Context, id string) error {
	r, ok := s.Get(id)
	if !ok {
		return fmt.Errorf("nodeagent: unknown replica %q", id)
	}
	s.gpuMu.Lock()
	err := r.Stop(ctx)
	s.gpuMu.Unlock()

	s.mu.Lock()
	delete(s.replicas, id)
	s.mu.Unlock()
	return err
}

// Shutdown stops every replica. Called on SIGTERM so a restarted agent does not find
// orphaned vLLM processes still holding the GPU.
func (s *Supervisor) Shutdown(ctx context.Context) {
	s.mu.RLock()
	ids := make([]string, 0, len(s.replicas))
	for id := range s.replicas {
		ids = append(ids, id)
	}
	s.mu.RUnlock()
	for _, id := range ids {
		_ = s.Stop(ctx, id)
	}
}

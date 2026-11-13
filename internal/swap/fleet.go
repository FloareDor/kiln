package swap

import (
	"context"
	"sync"
	"time"

	"github.com/sairavig/kiln/internal/store"
)

// StoreFleet implements Fleet over Postgres plus the gateway's live counters.
//
// Model metadata is cached rather than read per call. Snapshot runs on the request
// path for any model that is not already resident, and a database round trip there
// would add latency to exactly the requests that are already paying for a wake. The
// data it caches — footprints, measured wake costs, the pinned flag — changes only on
// a deploy, so a few seconds of staleness is harmless.
type StoreFleet struct {
	Store *store.Store
	// InFlightFn and QueueDepthFn come from the gateway, which is the only component
	// that knows these exactly and instantly.
	InFlightFn   func(model string) int
	QueueDepthFn func(model string) int

	TTL time.Duration

	mu       sync.RWMutex
	models   []*store.Model
	replicas map[string][]*store.Replica
	at       time.Time
}

// NewStoreFleet builds a fleet view.
func NewStoreFleet(s *store.Store, inFlight, queueDepth func(string) int) *StoreFleet {
	return &StoreFleet{
		Store: s, InFlightFn: inFlight, QueueDepthFn: queueDepth,
		TTL: 5 * time.Second, replicas: map[string][]*store.Replica{},
	}
}

// Refresh reloads the registry.
func (f *StoreFleet) Refresh(ctx context.Context) error {
	models, err := f.Store.Models(ctx)
	if err != nil {
		return err
	}
	all, err := f.Store.AllReplicas(ctx)
	if err != nil {
		return err
	}
	byModel := make(map[string][]*store.Replica, len(models))
	for _, r := range all {
		byModel[r.ModelID] = append(byModel[r.ModelID], r)
	}

	f.mu.Lock()
	f.models, f.replicas, f.at = models, byModel, time.Now()
	f.mu.Unlock()
	return nil
}

// Run refreshes on a timer.
func (f *StoreFleet) Run(ctx context.Context) {
	_ = f.Refresh(ctx)
	t := time.NewTicker(f.TTL)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			_ = f.Refresh(ctx)
		}
	}
}

// Models returns the cached registry, refreshing if it has gone stale.
func (f *StoreFleet) Models(ctx context.Context) ([]*store.Model, error) {
	f.mu.RLock()
	fresh := time.Since(f.at) < f.TTL && f.models != nil
	models := f.models
	f.mu.RUnlock()
	if fresh {
		return models, nil
	}
	if err := f.Refresh(ctx); err != nil {
		// Serve the stale copy rather than failing the swap. A model's footprint
		// from five seconds ago is a far better basis for a decision than no
		// decision at all, and a database blip should not stop the GPU working.
		f.mu.RLock()
		defer f.mu.RUnlock()
		if f.models != nil {
			return f.models, nil
		}
		return nil, err
	}
	f.mu.RLock()
	defer f.mu.RUnlock()
	return f.models, nil
}

// Replicas returns a model's replicas.
func (f *StoreFleet) Replicas(ctx context.Context, modelID string) ([]*store.Replica, error) {
	f.mu.RLock()
	reps, ok := f.replicas[modelID]
	fresh := time.Since(f.at) < f.TTL
	f.mu.RUnlock()
	if ok && fresh {
		return reps, nil
	}
	if err := f.Refresh(ctx); err != nil && !ok {
		return nil, err
	}
	f.mu.RLock()
	defer f.mu.RUnlock()
	return f.replicas[modelID], nil
}

// InFlight reports requests currently being served by a model.
func (f *StoreFleet) InFlight(modelID string) int {
	if f.InFlightFn == nil {
		return 0
	}
	return f.InFlightFn(modelID)
}

// QueueDepth reports requests waiting for a model.
func (f *StoreFleet) QueueDepth(modelID string) int {
	if f.QueueDepthFn == nil {
		return 0
	}
	return f.QueueDepthFn(modelID)
}

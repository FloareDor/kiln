// Package routing picks which replica of a model serves a request.
//
// Like the scheduler, everything here is pure: a Router receives a request summary
// and a slice of replica states and returns an id. No HTTP, no clock beyond what the
// caller passes in. That is what lets the four routers be compared on a recorded
// trace, deterministically, in a unit test.
package routing

import (
	"errors"
	"fmt"
	"sort"
	"sync"
	"sync/atomic"
)

// ErrNoReplica is returned when no replica can serve the request.
var ErrNoReplica = errors.New("routing: no ready replica")

// ReplicaID identifies one replica of one model.
type ReplicaID string

// ReplicaState is what the router knows about a replica. The gateway maintains it;
// the router never mutates it.
type ReplicaState struct {
	ID    ReplicaID
	Model string
	// Ready is false while a replica is starting, draining, sleeping or failed.
	Ready bool
	// InFlight is requests currently being served, and QueueDepth requests waiting.
	// Both feed the load penalty that keeps prefix affinity from creating a hot spot.
	InFlight   int
	QueueDepth int
	// RecentTTFTMillis is a decayed average time-to-first-token, used by
	// least-loaded as a tiebreak when queue depths are equal.
	RecentTTFTMillis float64
}

// Load is the scalar the load-sensitive routers minimise. Queue depth is weighted
// above in-flight because a queued request has not started at all, whereas an
// in-flight one is already being batched — vLLM's continuous batching absorbs
// concurrent decodes far more gracefully than it absorbs a backlog.
func (r ReplicaState) Load() float64 {
	return float64(r.QueueDepth)*2 + float64(r.InFlight)
}

// Request is what a router gets to see. Keeping it this narrow is deliberate: a
// router that could see the whole HTTP request would be tempting to make stateful in
// ways that break reproducibility.
type Request struct {
	Model string
	// PrefixBlocks are hashes of fixed-size token blocks from the start of the
	// prompt, longest-prefix-first. Produced by BlockHashes.
	PrefixBlocks []uint64
}

// Router picks a replica.
type Router interface {
	Name() string
	Pick(req Request, replicas []ReplicaState) (ReplicaID, error)
	// Observe tells the router which replica actually served a request, so
	// affinity-based routers can learn. No-op for stateless routers.
	Observe(req Request, picked ReplicaID)
}

func ready(replicas []ReplicaState) []ReplicaState {
	out := make([]ReplicaState, 0, len(replicas))
	for _, r := range replicas {
		if r.Ready {
			out = append(out, r)
		}
	}
	// Deterministic order so ties resolve identically across runs; without it the
	// router comparison is partly measuring Go's map and slice ordering.
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// --------------------------------------------------------------- round robin

// RoundRobin is the baseline: spread requests evenly, ignore everything else.
//
// It is the right default with no information, and it is exactly what defeats vLLM's
// prefix cache — consecutive turns of one conversation land on different replicas and
// each one recomputes the shared system prompt from scratch.
type RoundRobin struct{ n atomic.Uint64 }

// NewRoundRobin builds a round-robin router.
func NewRoundRobin() *RoundRobin { return &RoundRobin{} }

func (r *RoundRobin) Name() string { return "round-robin" }

func (r *RoundRobin) Pick(req Request, replicas []ReplicaState) (ReplicaID, error) {
	rs := ready(replicas)
	if len(rs) == 0 {
		return "", fmt.Errorf("%w for model %q", ErrNoReplica, req.Model)
	}
	// Add then subtract one so the first pick is index 0, which makes single-replica
	// tests read naturally.
	i := (r.n.Add(1) - 1) % uint64(len(rs))
	return rs[i].ID, nil
}

func (r *RoundRobin) Observe(Request, ReplicaID) {}

// --------------------------------------------------------------- least loaded

// LeastLoaded sends each request to the replica with the shallowest queue, breaking
// ties on recent TTFT.
//
// It is strictly better than round-robin when replicas are heterogeneous or when
// request costs vary, and it is the honest control for the prefix experiment: beating
// round-robin is easy, beating least-loaded is the claim worth making.
type LeastLoaded struct{}

// NewLeastLoaded builds a least-loaded router.
func NewLeastLoaded() *LeastLoaded { return &LeastLoaded{} }

func (l *LeastLoaded) Name() string { return "least-loaded" }

func (l *LeastLoaded) Pick(req Request, replicas []ReplicaState) (ReplicaID, error) {
	rs := ready(replicas)
	if len(rs) == 0 {
		return "", fmt.Errorf("%w for model %q", ErrNoReplica, req.Model)
	}
	best := rs[0]
	for _, r := range rs[1:] {
		if r.Load() < best.Load() ||
			(r.Load() == best.Load() && r.RecentTTFTMillis < best.RecentTTFTMillis) {
			best = r
		}
	}
	return best.ID, nil
}

func (l *LeastLoaded) Observe(Request, ReplicaID) {}

// --------------------------------------------------------------- prefix aware

// PrefixAware routes by remembered prefix affinity, optionally discounted by load.
//
// The idea: vLLM caches KV blocks for prompt prefixes it has already seen, so sending
// a request to a replica that recently served a prompt sharing its first N blocks
// turns the shared part of the prefill into a cache hit. Agent traffic — a long
// system prompt plus tool definitions resent on every turn — is almost entirely
// shared prefix, which is where this pays.
//
// The failure mode is equally important, and is why LoadWeight exists. Affinity alone
// sends every request for a popular prefix to one replica, which then queues while
// its peers idle. The score subtracts a load penalty so affinity wins when the
// replicas are comparably busy and loses when one is saturated. With LoadWeight at 0
// this is the pure prefix-affinity arm, which is the configuration that demonstrates
// the hot-spot problem.
type PrefixAware struct {
	// BlockWeight is the score per matching leading block.
	BlockWeight float64
	// LoadWeight scales the load penalty. Zero makes this pure affinity.
	LoadWeight float64
	// MaxEntries bounds the affinity table. Prefix tables grow without limit on real
	// traffic; this keeps one tenant's unique prompts from evicting everyone else's
	// useful entries, and bounds memory.
	MaxEntries int

	mu sync.Mutex
	// owner maps a block hash to the replica that most recently served a prompt
	// containing it, with an insertion counter for cheap approximate LRU eviction.
	owner map[uint64]prefixEntry
	clock uint64
}

type prefixEntry struct {
	replica ReplicaID
	seq     uint64
}

// NewPrefixAware builds a prefix-affinity router. loadWeight 0 is pure affinity;
// around 1.0 balances a matching block against one queued request.
func NewPrefixAware(loadWeight float64, maxEntries int) *PrefixAware {
	if maxEntries <= 0 {
		maxEntries = 100_000
	}
	return &PrefixAware{
		BlockWeight: 1.0, LoadWeight: loadWeight, MaxEntries: maxEntries,
		owner: make(map[uint64]prefixEntry),
	}
}

func (p *PrefixAware) Name() string {
	if p.LoadWeight == 0 {
		return "prefix-aware"
	}
	return "prefix-aware+load"
}

// matchLen counts how many leading blocks a replica is known to hold. It stops at the
// first miss on purpose: a KV cache hit requires an unbroken prefix, so blocks after
// a gap are worth nothing and counting them would overstate the benefit.
func (p *PrefixAware) matchLen(blocks []uint64, id ReplicaID) int {
	n := 0
	for _, b := range blocks {
		e, ok := p.owner[b]
		if !ok || e.replica != id {
			break
		}
		n++
	}
	return n
}

func (p *PrefixAware) Pick(req Request, replicas []ReplicaState) (ReplicaID, error) {
	rs := ready(replicas)
	if len(rs) == 0 {
		return "", fmt.Errorf("%w for model %q", ErrNoReplica, req.Model)
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	best, bestScore := rs[0].ID, -1e18
	for _, r := range rs {
		score := p.BlockWeight*float64(p.matchLen(req.PrefixBlocks, r.ID)) - p.LoadWeight*r.Load()
		if score > bestScore {
			best, bestScore = r.ID, score
		}
	}
	return best, nil
}

// Observe records that picked served this prompt, so later requests sharing its
// prefix are drawn to the same replica.
func (p *PrefixAware) Observe(req Request, picked ReplicaID) {
	if len(req.PrefixBlocks) == 0 {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, b := range req.PrefixBlocks {
		p.clock++
		p.owner[b] = prefixEntry{replica: picked, seq: p.clock}
	}
	if len(p.owner) > p.MaxEntries {
		p.evictOldest()
	}
}

// evictOldest drops the oldest quarter of the table.
//
// Batch eviction rather than one-at-a-time: exact LRU would need a linked list, and
// amortising a scan over a quarter of the entries is cheaper in both code and time
// for a table that is a hint, not a correctness requirement. A wrongly evicted entry
// costs one cache miss.
func (p *PrefixAware) evictOldest() {
	type kv struct {
		k uint64
		s uint64
	}
	all := make([]kv, 0, len(p.owner))
	for k, e := range p.owner {
		all = append(all, kv{k, e.seq})
	}
	sort.Slice(all, func(i, j int) bool { return all[i].s < all[j].s })
	for i := 0; i < len(all)/4; i++ {
		delete(p.owner, all[i].k)
	}
}

// Routers is the registry used by the --router flag and the benchmark harness.
var Routers = map[string]func() Router{
	"round-robin":        func() Router { return NewRoundRobin() },
	"least-loaded":       func() Router { return NewLeastLoaded() },
	"prefix-aware":       func() Router { return NewPrefixAware(0, 0) },
	"prefix-aware+load":  func() Router { return NewPrefixAware(1.0, 0) },
}

// RouterNames returns registry keys in a stable order.
func RouterNames() []string {
	out := make([]string, 0, len(Routers))
	for k := range Routers {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// NewRouter builds a router by name.
func NewRouter(name string) (Router, error) {
	f, ok := Routers[name]
	if !ok {
		return nil, fmt.Errorf("routing: unknown router %q (have %v)", name, RouterNames())
	}
	return f(), nil
}

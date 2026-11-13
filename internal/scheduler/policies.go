package scheduler

import (
	"fmt"
	"sort"
	"time"
)

// The five policies compared in experiment 3. They differ only in their victim score
// (and, for `cold`, in where victims land), so any benchmark gap between them is
// attributable to the scoring rule rather than to incidental implementation
// differences. See policy.go for the shared fitting loop.

// NewCold is the naive baseline: no sleep mode at all. Victims are stopped outright
// and the next request for them pays a full cold start. Choosing the victim by LRU
// keeps the comparison against `lru` honest — the only variable is whether sleep mode
// is used.
//
// This is the arm that answers "is tiering worth building?", so it has to be a fair
// implementation of the obvious alternative, not a strawman.
func NewCold() EvictionPolicy {
	return &scorer{
		name:  "cold",
		score: func(c *ClusterState, m *Model) float64 { return float64(m.LastRequest.UnixNano()) },
		reason: func(c *ClusterState, m *Model) string {
			return fmt.Sprintf("least recently used (idle %s)", m.Idle(c.Now).Round(time.Millisecond))
		},
		evictTo: func(c *ClusterState, m *Model) (Tier, string) {
			return TierNone, "cold baseline stops the process instead of sleeping it"
		},
		// No demotion: there is nothing in the host tier to demote.
		tick: nil,
	}
}

// NewLRU evicts whichever resident model has gone longest without a request.
//
// It is the right default and it is what most caches do, but it is blind to size and
// to wake cost: it will happily evict a 6GB model that takes 40s to wake in order to
// admit a 1GB one, when evicting two small cheap models would have been better.
func NewLRU() EvictionPolicy {
	return &scorer{
		name:  "lru",
		score: func(c *ClusterState, m *Model) float64 { return float64(m.LastRequest.UnixNano()) },
		reason: func(c *ClusterState, m *Model) string {
			return fmt.Sprintf("least recently used (idle %s)", m.Idle(c.Now).Round(time.Millisecond))
		},
		tick: defaultTick,
	}
}

// NewLFUDecay evicts the model with the lowest exponentially decayed arrival rate.
//
// Plain LFU is unusable in a serving system: a model that was hot last Tuesday keeps
// a high count forever and never gets evicted. The decay in Tracker gives recent
// requests more weight, so this behaves like LFU over a sliding window without
// storing one.
//
// Compared with LRU it is more robust to a scan — one request for each of twenty cold
// models will evict the hot model under LRU, but barely moves it under lfu-decay.
func NewLFUDecay() EvictionPolicy {
	return &scorer{
		name:  "lfu-decay",
		score: func(c *ClusterState, m *Model) float64 { return m.Rate },
		reason: func(c *ClusterState, m *Model) string {
			return fmt.Sprintf("lowest decayed arrival rate (%.4f req/s)", m.Rate)
		},
		tick: defaultTick,
	}
}

// NewCostAware scores a victim by the expected latency it will cost the system to
// have evicted it, per gigabyte of VRAM that evicting it frees:
//
//	score = (predicted arrival rate x wake cost from its destination tier) / GB freed
//
// The three terms are the three things LRU ignores:
//
//   - *rate* — how often we will have to pay the wake. A model nobody asks for is
//     free to evict however expensive its wake.
//   - *wake cost* — what the wake costs, measured on this GPU, from the tier the
//     victim would actually land in. Evicting a model to disk when host RAM is full
//     is scored with the disk wake cost, not the host one.
//   - *per GB* — a model that frees 6GB buys six times the headroom of one that frees
//     1GB, so it can justify six times the expected cost.
//
// This is the policy the project argues for, and the per-GB normalisation is the part
// that actually wins: on a mixed fleet it evicts one large cold model instead of four
// small warm ones.
func NewCostAware() EvictionPolicy {
	return &scorer{
		name:   "cost-aware",
		score:  costAwareScore,
		reason: costAwareReason,
		tick:   defaultTick,
	}
}

func wakeCostFromTier(c *ClusterState, m *Model) time.Duration {
	tier, _ := c.ChooseTier(m)
	switch tier {
	case TierHost:
		return m.WakeFromHost
	case TierDisk:
		return m.WakeFromDisk
	}
	return m.ColdStart
}

func costAwareScore(c *ClusterState, m *Model) float64 {
	gb := float64(m.VRAMBytes) / float64(1<<30)
	if gb <= 0 {
		gb = 1.0 / 1024 // never divide by zero; a zero-footprint model is free to evict
	}
	return m.Rate * wakeCostFromTier(c, m).Seconds() / gb
}

func costAwareReason(c *ClusterState, m *Model) string {
	return fmt.Sprintf("expected %.3f wake-seconds/s per GB (rate %.4f req/s, wake %s, frees %.2f GB)",
		costAwareScore(c, m), m.Rate,
		wakeCostFromTier(c, m).Round(time.Millisecond),
		float64(m.VRAMBytes)/float64(1<<30))
}

// NewPredictivePrewarm is cost-aware eviction plus a forward-looking Tick.
//
// Eviction scoring is identical to cost-aware on purpose — the experiment isolates
// the value of prewarming, so changing both at once would confound it. What differs
// is that each Tick projects every sleeper's arrival rate forward by
// Config.PrewarmHorizon using its measured trend, and wakes the ones whose projected
// demand clears PrewarmMinRate *while the VRAM is free anyway*.
//
// The "free anyway" condition is what keeps this from being a worse cost-aware: a
// speculative wake that forces an eviction is a bet the policy has no business making,
// so prewarming only ever consumes headroom that is currently idle.
func NewPredictivePrewarm() EvictionPolicy {
	return &scorer{
		name:   "predictive-prewarm",
		score:  costAwareScore,
		reason: costAwareReason,
		tick: func(c *ClusterState) []Action {
			out := defaultTick(c)

			type cand struct {
				m         *Model
				projected float64
			}
			var cands []cand
			for _, id := range c.SortedIDs() {
				m := c.Models[id]
				if m.State != StateSleepingL1 && m.State != StateSleepingL2 {
					continue
				}
				// Projected rate at the end of the horizon. Trend is req/s per
				// second, so this is a first-order extrapolation — deliberately
				// crude, because anything fancier would be fitting noise from a
				// handful of requests.
				projected := m.Rate + m.Trend*c.Config.PrewarmHorizon.Seconds()
				if projected < c.Config.PrewarmMinRate || m.Trend <= 0 {
					continue
				}
				cands = append(cands, cand{m, projected})
			}
			// Highest projected demand first, so scarce free VRAM goes to the model
			// most likely to need it.
			sort.SliceStable(cands, func(i, j int) bool {
				if cands[i].projected != cands[j].projected {
					return cands[i].projected > cands[j].projected
				}
				return cands[i].m.ID < cands[j].m.ID
			})

			free := c.VRAMFree()
			for _, cd := range cands {
				// Strictly opportunistic: prewarm only into headroom that already
				// exists. Never evict to make room for a guess.
				if cd.m.VRAMBytes > free {
					continue
				}
				free -= cd.m.VRAMBytes
				out = append(out, Action{
					Kind:  ActionPrewarm,
					Model: cd.m.ID,
					To:    TierGPU,
					Reason: fmt.Sprintf("rate %.4f rising at %.5f/s, projected %.4f req/s in %s; VRAM free",
						cd.m.Rate, cd.m.Trend, cd.projected, c.Config.PrewarmHorizon),
				})
			}
			return out
		},
	}
}

// Policies is the registry used by the --policy flag and by the benchmark harness, so
// adding an arm to the experiment means adding one line here.
var Policies = map[string]func() EvictionPolicy{
	"cold":               NewCold,
	"lru":                NewLRU,
	"lfu-decay":          NewLFUDecay,
	"cost-aware":         NewCostAware,
	"predictive-prewarm": NewPredictivePrewarm,
}

// PolicyNames returns the registry keys in a stable order.
func PolicyNames() []string {
	out := make([]string, 0, len(Policies))
	for k := range Policies {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// NewPolicy builds a policy by name.
func NewPolicy(name string) (EvictionPolicy, error) {
	f, ok := Policies[name]
	if !ok {
		return nil, fmt.Errorf("scheduler: unknown policy %q (have %v)", name, PolicyNames())
	}
	return f(), nil
}

package scheduler

import (
	"errors"
	"fmt"
	"sort"
	"time"
)

// ErrCannotFit is returned when no sequence of evictions can admit the target: the
// model is larger than the whole budget, or everything else is pinned. The caller
// must fail the request rather than queue it, because waiting will not help.
var ErrCannotFit = errors.New("scheduler: target cannot fit even after evicting everything evictable")

// ErrWouldThrash is returned when enough VRAM exists in principle but every candidate
// is inside its minimum residency window. The caller should queue the request and
// retry, because waiting *will* help — the window expires on its own.
//
// Distinguishing these two is the difference between a 503 and a slightly slow
// request, and it is the single most useful thing the anti-thrash guard produces.
var ErrWouldThrash = errors.New("scheduler: all candidates are inside their min-residency window")

// EvictionPolicy decides which models are paged out of VRAM and what background
// tiering work to do between requests.
//
// Implementations must be pure: no I/O, no clock reads (use state.Now), no mutation
// of the state passed in. That contract is what lets the whole policy suite run
// against recorded traces in a unit test, deterministically, with no GPU.
type EvictionPolicy interface {
	// Name is the identifier used in metrics labels, CLI flags and result tables.
	Name() string

	// PickVictims returns the models to evict, in the order they should be drained,
	// so that target can be admitted. It returns nil with no error when target
	// already fits.
	PickVictims(state *ClusterState, target ModelID) ([]Eviction, error)

	// Tick returns background work: prewarming sleepers whose demand is rising,
	// demoting idle host-tier sleepers to disk, stopping long-dead processes.
	// Called on a timer, not on the request path.
	Tick(state *ClusterState) []Action
}

// scorer is the shared shape of the four score-based policies. Each supplies a cost
// function over a candidate victim; lower scores are evicted first. Everything else —
// the fitting loop, tier choice, thrash detection — is identical across policies on
// purpose, so a benchmark difference between two policies is attributable to the
// score and nothing else.
type scorer struct {
	name string
	// score returns the victim's value; the lowest-valued victim is evicted first.
	score func(c *ClusterState, m *Model) float64
	// reason explains the score in one phrase for the decision log.
	reason func(c *ClusterState, m *Model) string
	// evictTo overrides tier selection. nil means use ClusterState.ChooseTier.
	evictTo func(c *ClusterState, m *Model) (Tier, string)
	// tick supplies the policy's background work. nil means none.
	tick func(c *ClusterState) []Action
}

func (s *scorer) Name() string { return s.name }

func (s *scorer) PickVictims(c *ClusterState, target ModelID) ([]Eviction, error) {
	m := c.Get(target)
	if m == nil {
		return nil, fmt.Errorf("scheduler: unknown target %q", target)
	}
	need := c.Need(target)
	if need <= 0 {
		return nil, nil
	}

	cands := c.EvictableSet(target)

	// Separate the two failure modes. If even evicting every candidate leaves the
	// target short, no amount of waiting fixes it.
	var reclaimable int64
	for _, cm := range cands {
		reclaimable += cm.VRAMBytes
	}
	if reclaimable < need {
		// Would waiting out the residency windows be enough? If so this is
		// temporary contention, not an impossibility.
		var potential int64
		for _, id := range c.SortedIDs() {
			cm := c.Models[id]
			if id == target || cm.Pinned || !cm.State.HoldsVRAM() {
				continue
			}
			potential += cm.VRAMBytes
		}
		if potential >= need {
			return nil, ErrWouldThrash
		}
		return nil, fmt.Errorf("%w: need %d bytes, at most %d reclaimable",
			ErrCannotFit, need, potential)
	}

	type scored struct {
		m *Model
		s float64
	}
	ranked := make([]scored, 0, len(cands))
	for _, cm := range cands {
		ranked = append(ranked, scored{cm, s.score(c, cm)})
	}
	// Ties break on ID so the result is reproducible across runs; without this two
	// identically-scored models swap places between runs and the benchmark becomes
	// noise.
	sort.SliceStable(ranked, func(i, j int) bool {
		if ranked[i].s != ranked[j].s {
			return ranked[i].s < ranked[j].s
		}
		return ranked[i].m.ID < ranked[j].m.ID
	})

	// Evicting into a working copy keeps the host-RAM budget honest: after the first
	// victim goes to the host tier, the second one sees the reduced budget and may
	// correctly be sent to disk instead.
	work := c.Clone()
	var freed int64
	var out []Eviction
	for _, r := range ranked {
		if freed >= need {
			break
		}
		tier, tierWhy := TierDisk, "cold baseline stops the process"
		if s.evictTo != nil {
			tier, tierWhy = s.evictTo(work, r.m)
		} else {
			tier, tierWhy = work.ChooseTier(r.m)
		}
		out = append(out, Eviction{
			Model:  r.m.ID,
			To:     tier,
			Reason: fmt.Sprintf("%s; %s", s.reason(c, r.m), tierWhy),
			Score:  r.s,
		})
		freed += r.m.VRAMBytes
		work.Models[r.m.ID].State = SleepStateFor(tier)
	}
	return out, nil
}

func (s *scorer) Tick(c *ClusterState) []Action {
	if s.tick == nil {
		return nil
	}
	return s.tick(c)
}

// defaultTick is shared by every policy except the cold baseline: demote host-tier
// sleepers that have gone quiet, so pinned RAM is not held hostage by a model nobody
// wants. It deliberately does not prewarm; only the predictive policy does that.
func defaultTick(c *ClusterState) []Action {
	if c.Config.IdleDemoteAfter <= 0 {
		return nil
	}
	var out []Action
	for _, id := range c.SortedIDs() {
		m := c.Models[id]
		if m.State != StateSleepingL1 || m.Pinned {
			continue
		}
		if m.Idle(c.Now) >= c.Config.IdleDemoteAfter {
			out = append(out, Action{
				Kind:   ActionDemote,
				Model:  id,
				To:     TierDisk,
				Reason: fmt.Sprintf("idle %s in host tier", m.Idle(c.Now).Round(time.Second)),
			})
		}
	}
	return out
}

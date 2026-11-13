package scheduler

import "sort"

// VRAMUsed is the total GPU memory currently claimed by models in VRAM-holding
// states. STARTING and WAKING models count, because their allocation is already in
// flight; ignoring them is how you admit a second model into memory the first one is
// halfway through claiming and get a CUDA OOM at the worst possible moment.
func (c *ClusterState) VRAMUsed() int64 {
	var total int64
	for _, m := range c.Models {
		if m.State.HoldsVRAM() {
			total += m.VRAMBytes
		}
	}
	return total
}

// VRAMFree is the budget left after accounting for everything currently resident.
func (c *ClusterState) VRAMFree() int64 {
	free := c.GPU.Budget() - c.VRAMUsed()
	if free < 0 {
		return 0
	}
	return free
}

// HostRAMUsed is the pinned host memory held by sleep-level-1 sleepers.
func (c *ClusterState) HostRAMUsed() int64 {
	var total int64
	for _, m := range c.Models {
		if m.State.HoldsHostRAM() {
			total += m.HostBytes
		}
	}
	return total
}

// HostRAMFree is how much of the host tier budget remains.
func (c *ClusterState) HostRAMFree() int64 {
	free := c.Config.HostRAMBudget - c.HostRAMUsed()
	if free < 0 {
		return 0
	}
	return free
}

// Get returns a model or nil.
func (c *ClusterState) Get(id ModelID) *Model { return c.Models[id] }

// SortedIDs returns model IDs in a deterministic order. Map iteration order in Go is
// randomised, and a policy that iterates a map produces different victims on
// identical input, which makes benchmark runs unreproducible and A/B comparisons
// meaningless. Every policy iterates through this.
func (c *ClusterState) SortedIDs() []ModelID {
	ids := make([]ModelID, 0, len(c.Models))
	for id := range c.Models {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	return ids
}

// Evictable reports whether a model may be chosen as a victim right now, and if not,
// why. The reason is returned for the decision log: "nothing was evictable" is a much
// less useful benchmark artefact than "everything was inside min-residency".
func (c *ClusterState) Evictable(id ModelID, target ModelID) (bool, string) {
	m := c.Models[id]
	switch {
	case m == nil:
		return false, "unknown model"
	case id == target:
		return false, "is the target"
	case m.Pinned:
		return false, "pinned"
	case !m.State.HoldsVRAM():
		return false, "not resident in VRAM"
	case m.State == StateStarting || m.State == StateWaking:
		// Evicting a model mid-load wastes the load and usually leaves the allocator
		// in a state nobody wants to reason about. Let it finish; min-residency will
		// make it evictable soon enough.
		return false, "transition in progress"
	case m.State == StateDraining:
		return false, "already draining"
	case c.Now.Sub(m.ResidentSince) < c.Config.MinResidency:
		return false, "inside min-residency window"
	}
	return true, ""
}

// EvictableSet returns every model that may currently be evicted to make room for
// target, in deterministic order.
func (c *ClusterState) EvictableSet(target ModelID) []*Model {
	var out []*Model
	for _, id := range c.SortedIDs() {
		if ok, _ := c.Evictable(id, target); ok {
			out = append(out, c.Models[id])
		}
	}
	return out
}

// Blockers explains, per model, why it could not be evicted. Used when a swap cannot
// be satisfied so the caller can distinguish "wait a moment" from "impossible".
func (c *ClusterState) Blockers(target ModelID) map[ModelID]string {
	out := make(map[ModelID]string)
	for _, id := range c.SortedIDs() {
		if ok, why := c.Evictable(id, target); !ok && why != "is the target" {
			out[id] = why
		}
	}
	return out
}

// Need is how much VRAM must be freed to admit target, including the hysteresis
// margin. Returns 0 when the model already fits.
//
// Hysteresis matters more than it looks: freeing exactly enough leaves the GPU at
// 100% occupancy, so the very next model — even one a few MiB larger — triggers
// another full eviction cycle. Over-freeing by a fixed margin converts a stream of
// single-victim evictions into occasional two-victim ones, which is cheaper overall.
func (c *ClusterState) Need(target ModelID) int64 {
	m := c.Models[target]
	if m == nil || m.State.HoldsVRAM() {
		return 0
	}
	deficit := m.VRAMBytes + c.Config.HysteresisBytes - c.VRAMFree()
	if deficit < 0 {
		return 0
	}
	// Never ask for more than is theoretically reclaimable, or callers see a demand
	// that can never be met even after evicting everything.
	return deficit
}

// Reclaimable is the total VRAM that could be freed by evicting everything currently
// evictable. If Need exceeds this, no sequence of evictions admits the target and the
// request must be rejected rather than queued forever.
func (c *ClusterState) Reclaimable(target ModelID) int64 {
	var total int64
	for _, m := range c.EvictableSet(target) {
		total += m.VRAMBytes
	}
	return total
}

// ChooseTier decides where a victim's weights should land.
//
// Host memory (sleep level 1) is fast to wake but scarce, so it is reserved for
// models likely to be asked for again soon. "Likely" is the decayed arrival rate
// crossing HostTierMinRate. Anything colder goes to disk (level 2), which costs a
// re-read on wake but no pinned RAM. If the host budget is exhausted, disk is the
// only option regardless of how hot the victim is.
func (c *ClusterState) ChooseTier(victim *Model) (Tier, string) {
	if victim.HostBytes > c.HostRAMFree() {
		return TierDisk, "host tier budget exhausted"
	}
	if victim.Rate < c.Config.HostTierMinRate {
		return TierDisk, "arrival rate below host-tier threshold"
	}
	return TierHost, "hot enough to keep in host memory"
}

// SleepStateFor maps a destination tier onto the model state that represents it.
func SleepStateFor(t Tier) State {
	switch t {
	case TierHost:
		return StateSleepingL1
	case TierDisk:
		return StateSleepingL2
	}
	return StateCold
}

// Package scheduler contains Kiln's GPU memory tiering logic: the model state
// machine, VRAM/host-RAM accounting, and the pluggable eviction policies that decide
// which model gets paged out of the GPU when a cold or sleeping model is requested.
//
// Everything in this package is pure. Nothing here opens a socket, shells out to
// nvidia-smi, or talks to vLLM. A policy receives a ClusterState snapshot and returns
// decisions; the caller (the control plane) is responsible for carrying them out and
// for feeding measured results back in. That split is what makes the interesting part
// of this project testable on a laptop with no GPU attached.
package scheduler

import (
	"fmt"
	"time"
)

// ModelID identifies a deployed model (not a replica). A model may have several
// replicas, but tiering decisions are made per model: sleeping a model sleeps all of
// its replicas on the GPU in question.
type ModelID string

// Tier is where a model's weights live.
type Tier int

const (
	// TierGPU means weights are resident in VRAM and the model can serve immediately.
	TierGPU Tier = iota
	// TierHost means weights have been offloaded to pinned CPU memory (vLLM sleep
	// level 1). The process and its CUDA context survive, so waking is a PCIe copy.
	TierHost
	// TierDisk means weights were discarded entirely (vLLM sleep level 2). The process
	// survives but the weights must be re-read from the page cache or disk on wake.
	TierDisk
	// TierNone means no process exists. Waking is a full cold start: spawn, load,
	// profile, capture CUDA graphs.
	TierNone
)

func (t Tier) String() string {
	switch t {
	case TierGPU:
		return "gpu"
	case TierHost:
		return "host"
	case TierDisk:
		return "disk"
	case TierNone:
		return "none"
	}
	return fmt.Sprintf("tier(%d)", int(t))
}

// State is the model lifecycle state. The transitions the control plane is allowed to
// drive are enumerated in Transitions; anything else is a bug and is rejected by
// ValidateTransition rather than silently applied.
//
//	COLD -> STARTING -> ACTIVE <-> DRAINING -> SLEEPING_L1 <-> WAKING -> ACTIVE
//	                                       \-> SLEEPING_L2 <-> WAKING
//	any -> FAILED (retried with backoff)
type State int

const (
	StateCold State = iota
	StateStarting
	StateActive
	StateDraining
	StateSleepingL1
	StateSleepingL2
	StateWaking
	StateFailed
)

var stateNames = map[State]string{
	StateCold:       "COLD",
	StateStarting:   "STARTING",
	StateActive:     "ACTIVE",
	StateDraining:   "DRAINING",
	StateSleepingL1: "SLEEPING_L1",
	StateSleepingL2: "SLEEPING_L2",
	StateWaking:     "WAKING",
	StateFailed:     "FAILED",
}

func (s State) String() string {
	if n, ok := stateNames[s]; ok {
		return n
	}
	return fmt.Sprintf("state(%d)", int(s))
}

// ParseState is the inverse of State.String, used when rehydrating from Postgres.
func ParseState(s string) (State, error) {
	for k, v := range stateNames {
		if v == s {
			return k, nil
		}
	}
	return StateFailed, fmt.Errorf("scheduler: unknown state %q", s)
}

// HoldsVRAM reports whether a model in this state is occupying GPU memory and so must
// be counted against the VRAM budget. DRAINING counts: the model is still resident and
// finishing its in-flight requests, it just is not accepting new ones. STARTING and
// WAKING count because their memory is already being claimed — not counting them is
// how you oversubscribe a GPU and get an allocator OOM mid-wake.
func (s State) HoldsVRAM() bool {
	switch s {
	case StateStarting, StateActive, StateDraining, StateWaking:
		return true
	}
	return false
}

// HoldsHostRAM reports whether the model is occupying pinned host memory, which only
// sleep level 1 does.
func (s State) HoldsHostRAM() bool {
	return s == StateSleepingL1
}

// Serving reports whether the gateway may route new requests to this model.
func (s State) Serving() bool { return s == StateActive }

// Transitions is the allowed state graph. FAILED is reachable from anywhere and so is
// handled separately in ValidateTransition.
var Transitions = map[State][]State{
	StateCold:       {StateStarting},
	StateStarting:   {StateActive},
	StateActive:     {StateDraining},
	StateDraining:   {StateSleepingL1, StateSleepingL2, StateCold, StateActive},
	StateSleepingL1: {StateWaking, StateSleepingL2, StateCold},
	StateSleepingL2: {StateWaking, StateCold},
	StateWaking:     {StateActive},
	StateFailed:     {StateCold, StateStarting},
}

// ValidateTransition returns nil if from -> to is a legal edge.
//
// DRAINING -> ACTIVE exists on purpose: if a drain is abandoned (say the request that
// triggered the eviction timed out and the victim is suddenly hot again) the model
// goes straight back to serving without a swap. That edge is what keeps a burst of
// cancelled requests from costing a pair of real swaps.
func ValidateTransition(from, to State) error {
	if to == StateFailed {
		return nil
	}
	for _, allowed := range Transitions[from] {
		if allowed == to {
			return nil
		}
	}
	return fmt.Errorf("scheduler: illegal transition %s -> %s", from, to)
}

// TierOf maps a state to the tier its weights currently occupy.
func TierOf(s State) Tier {
	switch {
	case s.HoldsVRAM():
		return TierGPU
	case s == StateSleepingL1:
		return TierHost
	case s == StateSleepingL2:
		return TierDisk
	}
	return TierNone
}

// Model is the scheduler's view of one deployed model. Fields fall into three groups:
// static facts measured at deploy time, live state owned by the control plane, and
// demand statistics maintained by the Tracker.
type Model struct {
	ID    ModelID
	State State

	// Pinned models are never chosen as eviction victims, whatever a policy scores
	// them. This is the escape hatch for a latency-critical model that must never
	// pay a wake.
	Pinned bool

	// VRAMBytes is the measured resident GPU footprint when ACTIVE: weights plus the
	// KV cache arena vLLM reserves from --gpu-memory-utilization. It is measured
	// during the deploy job rather than estimated from parameter count, because the
	// KV arena usually dominates and depends on flags we chose.
	VRAMBytes int64
	// HostBytes is the pinned host memory a sleep level 1 offload consumes. Close to
	// the weight size, excluding the KV arena, which is dropped rather than copied.
	HostBytes int64

	// Measured transition costs, in wall-clock. Seeded from the deploy-time
	// measurement and then updated by an EWMA over observed swaps, so a policy that
	// scores by cost uses this GPU's real numbers rather than a guess.
	ColdStart time.Duration
	WakeFromHost time.Duration
	WakeFromDisk time.Duration

	// Live load, supplied by the gateway.
	InFlight   int
	QueueDepth int

	// LastRequest is when a request for this model last arrived, used by LRU and to
	// decide whether a sleeper is worth keeping in host RAM.
	LastRequest time.Time
	// ResidentSince is when the model last entered ACTIVE. Anti-thrash uses it: a
	// model that has been serving for less than Config.MinResidency is not evictable.
	ResidentSince time.Time

	// Rate is the exponentially decayed arrival rate in requests per second, and
	// Trend its derivative (positive means demand is climbing). Both are maintained
	// by Tracker and consumed by the lfu-decay, cost-aware and prewarm policies.
	Rate  float64
	Trend float64
}

// WakeCost is how long it would take to make this model servable from where it is now.
func (m *Model) WakeCost() time.Duration {
	switch m.State {
	case StateActive:
		return 0
	case StateSleepingL1:
		return m.WakeFromHost
	case StateSleepingL2:
		return m.WakeFromDisk
	case StateStarting, StateWaking:
		// Already on its way; treat the remaining cost as the full one rather than
		// zero, so a policy does not prefer interrupting an in-progress load.
		return m.ColdStart
	}
	return m.ColdStart
}

// Idle is how long since the last request arrived for this model.
func (m *Model) Idle(now time.Time) time.Duration { return now.Sub(m.LastRequest) }

// GPU describes the device budget for one node.
type GPU struct {
	Node string
	// TotalBytes is the device's physical VRAM.
	TotalBytes int64
	// ReservedBytes is held back from the budget and never allocated to models: the
	// CUDA context, allocator fragmentation, and on a laptop the desktop compositor.
	// Without this the scheduler packs a model into memory the driver has already
	// spoken for and the load fails at the last moment.
	ReservedBytes int64
}

// Budget is TotalBytes minus the reservation: the VRAM the scheduler may hand out.
func (g GPU) Budget() int64 {
	b := g.TotalBytes - g.ReservedBytes
	if b < 0 {
		return 0
	}
	return b
}

// Config holds the tuning knobs shared by all policies.
type Config struct {
	// MinResidency is how long a model must have been ACTIVE before it may be
	// evicted. This is the primary anti-thrash guard: without it, two models that
	// each receive steady traffic on a GPU that fits one will evict each other on
	// every alternating request and spend all their time swapping.
	MinResidency time.Duration

	// HysteresisBytes is extra VRAM freed beyond what the incoming model strictly
	// needs. Freeing exactly enough means the next slightly larger model triggers
	// another eviction immediately; over-freeing a little amortises that.
	HysteresisBytes int64

	// DrainDeadline bounds how long the scheduler waits for in-flight requests on a
	// victim to finish before sleeping it anyway. Unbounded draining lets one slow
	// request block a swap forever; the deadline trades a few cancelled requests for
	// a bounded swap latency.
	DrainDeadline time.Duration

	// HostRAMBudget caps the pinned host memory sleep level 1 offloads may consume
	// in total. Host RAM is the real ceiling on how many sleepers you can keep warm,
	// and exceeding it gets the whole node OOM-killed rather than slowed down.
	HostRAMBudget int64

	// HostTierMinRate is the decayed arrival rate above which a victim is offloaded
	// to host RAM rather than disk. A model nobody is asking for does not deserve
	// scarce pinned memory.
	HostTierMinRate float64

	// IdleDemoteAfter is how long a host-tier sleeper may sit untouched before it is
	// demoted to disk to reclaim host RAM. Zero disables demotion.
	IdleDemoteAfter time.Duration

	// PrewarmHorizon is how far ahead the predictive policy looks: a sleeper is
	// prewarmed when its projected arrival rate within this window clears
	// PrewarmMinRate and the VRAM is free anyway.
	PrewarmHorizon time.Duration
	PrewarmMinRate float64
}

// DefaultConfig is tuned for the 8GB RTX 3070 development box. The Modal runs
// override MinResidency and HostRAMBudget via the node agent's flags.
func DefaultConfig() Config {
	return Config{
		MinResidency:    15 * time.Second,
		HysteresisBytes: 256 << 20,
		DrainDeadline:   10 * time.Second,
		HostRAMBudget:   16 << 30,
		HostTierMinRate: 0.02,
		IdleDemoteAfter: 5 * time.Minute,
		PrewarmHorizon:  30 * time.Second,
		PrewarmMinRate:  0.1,
	}
}

// ClusterState is the immutable snapshot a policy reasons over. Policies must not
// mutate it; the planner copies what it needs.
type ClusterState struct {
	Now    time.Time
	GPU    GPU
	Config Config
	Models map[ModelID]*Model
}

// Clone deep-copies the state. Tests and the simulator use it to fork timelines.
func (c *ClusterState) Clone() *ClusterState {
	out := &ClusterState{Now: c.Now, GPU: c.GPU, Config: c.Config, Models: make(map[ModelID]*Model, len(c.Models))}
	for id, m := range c.Models {
		cp := *m
		out.Models[id] = &cp
	}
	return out
}

// Eviction is a policy's decision to page one model out of VRAM.
type Eviction struct {
	Model ModelID
	// To is the tier the victim should land in: TierHost (sleep 1), TierDisk
	// (sleep 2) or TierNone (stop the process entirely, used by the cold baseline).
	To Tier
	// Reason is recorded on the swap metric and in the decision log so an odd
	// benchmark result can be traced back to the scoring that produced it.
	Reason string
	// Score is the policy's victim score, lowest-first. Exposed for debugging and
	// for the policy comparison tables.
	Score float64
}

// ActionKind enumerates the background work a policy may request from Tick.
type ActionKind int

const (
	// ActionPrewarm asks the control plane to wake a sleeper before demand arrives.
	ActionPrewarm ActionKind = iota
	// ActionDemote moves an idle host-tier sleeper down to disk to reclaim host RAM.
	ActionDemote
	// ActionStop terminates a disk-tier sleeper's process entirely.
	ActionStop
)

func (a ActionKind) String() string {
	switch a {
	case ActionPrewarm:
		return "prewarm"
	case ActionDemote:
		return "demote"
	case ActionStop:
		return "stop"
	}
	return fmt.Sprintf("action(%d)", int(a))
}

// Action is a unit of background work returned by EvictionPolicy.Tick.
type Action struct {
	Kind   ActionKind
	Model  ModelID
	To     Tier
	Reason string
}

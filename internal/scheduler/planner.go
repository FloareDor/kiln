package scheduler

import (
	"fmt"
	"time"
)

// StepKind is one executable operation in a swap plan.
type StepKind int

const (
	// StepDrain stops routing new requests to a model and waits for its in-flight
	// requests to finish, bounded by Config.DrainDeadline.
	StepDrain StepKind = iota
	// StepSleep offloads a drained model to the host or disk tier.
	StepSleep
	// StepStop terminates a model's process outright (cold baseline, or reclaiming a
	// long-dead sleeper).
	StepStop
	// StepWake restores a sleeping model to VRAM.
	StepWake
	// StepStart cold-starts a model that has no process at all.
	StepStart
)

func (k StepKind) String() string {
	switch k {
	case StepDrain:
		return "drain"
	case StepSleep:
		return "sleep"
	case StepStop:
		return "stop"
	case StepWake:
		return "wake"
	case StepStart:
		return "start"
	}
	return fmt.Sprintf("step(%d)", int(k))
}

// Step is one operation the control plane executes against a node agent.
type Step struct {
	Kind  StepKind
	Model ModelID
	// To is the destination tier for StepSleep.
	To Tier
	// Deadline bounds a StepDrain; zero for other kinds.
	Deadline time.Duration
	Reason   string
}

func (s Step) String() string {
	if s.Kind == StepSleep {
		return fmt.Sprintf("%s %s -> %s", s.Kind, s.Model, s.To)
	}
	return fmt.Sprintf("%s %s", s.Kind, s.Model)
}

// Plan is an ordered swap plan: everything that must happen for Target to serve.
type Plan struct {
	Target ModelID
	Steps  []Step
	// FreedBytes is how much VRAM the evictions release, and NeedBytes how much was
	// required including hysteresis. Recorded on the swap metric so a plan that
	// over-evicts shows up in the data rather than only in a hunch.
	FreedBytes int64
	NeedBytes  int64
	// EstimatedCost is how long the plan is expected to take, from measured per-model
	// transition costs. The gateway uses it to decide whether to queue the request or
	// fail fast against the client's deadline.
	EstimatedCost time.Duration
	Policy        string
}

// NoOp reports whether the target is already servable.
func (p *Plan) NoOp() bool { return len(p.Steps) == 0 }

// Planner turns a policy's victim choices into an executable plan. It owns no state;
// the caller supplies a fresh snapshot each time.
type Planner struct {
	Policy EvictionPolicy
}

// NewPlanner builds a planner over a policy.
func NewPlanner(p EvictionPolicy) *Planner { return &Planner{Policy: p} }

// Plan produces the ordered steps required to make target servable.
//
// Ordering matters and is not negotiable: every victim is drained before any victim
// is slept, and all sleeps complete before the target is woken. Interleaving a wake
// with an outstanding drain is how you get a CUDA OOM — the wake allocates while the
// victim still holds its arena. The cost of the strict ordering is that the slowest
// drain gates the whole swap, which is exactly what DrainDeadline bounds.
func (pl *Planner) Plan(c *ClusterState, target ModelID) (*Plan, error) {
	m := c.Get(target)
	if m == nil {
		return nil, fmt.Errorf("scheduler: unknown target %q", target)
	}

	plan := &Plan{Target: target, NeedBytes: c.Need(target), Policy: pl.Policy.Name()}

	// Already serving, or already on its way: nothing to plan. Returning an empty
	// plan rather than a duplicate wake is what makes swap coalescing work — fifty
	// concurrent requests for one sleeping model all land here after the first one
	// has moved it to WAKING, and produce exactly one wake between them.
	switch m.State {
	case StateActive, StateWaking, StateStarting:
		return plan, nil
	case StateDraining:
		// The model is being evicted but has not yet been slept. Cancelling the
		// drain is strictly cheaper than completing it and waking again.
		plan.Steps = append(plan.Steps, Step{
			Kind: StepWake, Model: target,
			Reason: "cancel in-progress drain; model is wanted again",
		})
		return plan, nil
	}

	victims, err := pl.Policy.PickVictims(c, target)
	if err != nil {
		return nil, err
	}

	for _, v := range victims {
		plan.Steps = append(plan.Steps, Step{
			Kind: StepDrain, Model: v.Model,
			Deadline: c.Config.DrainDeadline,
			Reason:   v.Reason,
		})
	}
	for _, v := range victims {
		vm := c.Models[v.Model]
		plan.FreedBytes += vm.VRAMBytes
		kind := StepSleep
		if v.To == TierNone {
			kind = StepStop
		}
		plan.Steps = append(plan.Steps, Step{
			Kind: kind, Model: v.Model, To: v.To, Reason: v.Reason,
		})
	}

	switch m.State {
	case StateSleepingL1, StateSleepingL2:
		plan.Steps = append(plan.Steps, Step{
			Kind: StepWake, Model: target,
			Reason: fmt.Sprintf("wake from %s tier", TierOf(m.State)),
		})
	default: // COLD or FAILED
		plan.Steps = append(plan.Steps, Step{
			Kind: StepStart, Model: target, Reason: "no process; cold start",
		})
	}

	plan.EstimatedCost = pl.estimate(c, plan)
	return plan, nil
}

// estimate sums the plan's expected wall-clock.
//
// Drains run concurrently once issued, so their contribution is the maximum rather
// than the sum, and a model with no in-flight requests drains instantly. Sleeps are
// fast enough relative to wakes to fold into the wake estimate.
func (pl *Planner) estimate(c *ClusterState, plan *Plan) time.Duration {
	var maxDrain, wake time.Duration
	for _, s := range plan.Steps {
		switch s.Kind {
		case StepDrain:
			vm := c.Models[s.Model]
			if vm != nil && vm.InFlight > 0 {
				// Without a per-model latency estimate, assume the worst bounded
				// case. Pessimistic on purpose: the gateway uses this to decide
				// whether to fail fast, and under-promising a swap is worse than
				// over-promising one.
				if d := c.Config.DrainDeadline; d > maxDrain {
					maxDrain = d
				}
			}
		case StepWake:
			if vm := c.Models[s.Model]; vm != nil {
				wake += vm.WakeCost()
			}
		case StepStart:
			if vm := c.Models[s.Model]; vm != nil {
				wake += vm.ColdStart
			}
		}
	}
	return maxDrain + wake
}

// Tick exposes the policy's background work.
func (pl *Planner) Tick(c *ClusterState) []Action { return pl.Policy.Tick(c) }

// Apply mutates a state as if the plan had executed successfully. It exists for the
// offline simulator and for tests: running a trace through Apply is how the policy
// comparison in experiment 3 is produced without a GPU, and how the swap counts in
// the results table are computed.
func (plan *Plan) Apply(c *ClusterState) error {
	for _, s := range plan.Steps {
		m := c.Models[s.Model]
		if m == nil {
			return fmt.Errorf("scheduler: plan references unknown model %q", s.Model)
		}

		// A step is a logical operation, not a single edge: waking a sleeper really
		// passes through WAKING, and starting a cold model through STARTING. Walking
		// the intermediate states rather than jumping to the end keeps Apply's result
		// identical to what the real executor produces, so the offline simulator and
		// the live control plane cannot disagree about what a plan did.
		var path []State
		switch s.Kind {
		case StepDrain:
			path = []State{StateDraining}
		case StepSleep:
			path = []State{SleepStateFor(s.To)}
		case StepStop:
			path = []State{StateCold}
		case StepWake:
			if m.State == StateDraining {
				path = []State{StateActive} // cancelled drain: no reload needed
			} else {
				path = []State{StateWaking, StateActive}
			}
		case StepStart:
			path = []State{StateStarting, StateActive}
		}

		for _, next := range path {
			if err := ValidateTransition(m.State, next); err != nil {
				return fmt.Errorf("applying %s: %w", s, err)
			}
			m.State = next
		}
		if m.State == StateActive {
			m.ResidentSince = c.Now
		}
	}
	return nil
}

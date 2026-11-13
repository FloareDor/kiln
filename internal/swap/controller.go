// Package swap executes GPU memory tiering: it turns the scheduler's pure decisions
// into sleep, wake and start calls against node agents.
//
// The controller runs inside the gateway process rather than the control plane. That
// is a deliberate placement, and it follows from what draining needs: a model may
// only be slept once its in-flight requests have finished, and the gateway is the
// only component that knows the in-flight count exactly and instantly. Putting the
// controller in the control plane would mean polling the gateway for that count,
// which is both a round trip on the eviction path and a number that is already stale
// when it arrives.
//
// The split of responsibility is therefore: the control plane owns deployments (what
// models exist, what they cost), the gateway owns residency (which of them are in
// VRAM right now).
package swap

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"sync"
	"time"

	"github.com/FloareDor/kiln/internal/agentclient"
	"github.com/FloareDor/kiln/internal/metrics"
	"github.com/FloareDor/kiln/internal/nodeagent"
	"github.com/FloareDor/kiln/internal/scheduler"
	"github.com/FloareDor/kiln/internal/store"
)

// Fleet is the controller's view of what is deployed and where. The gateway's
// registry supplies it; the controller never talks to the database directly on the
// request path.
type Fleet interface {
	// Models returns every deployed model with its measured costs.
	Models(ctx context.Context) ([]*store.Model, error)
	// Replicas returns a model's replicas and the agent that owns each.
	Replicas(ctx context.Context, modelID string) ([]*store.Replica, error)
	// InFlight is the gateway's live count of requests being served by a model.
	InFlight(modelID string) int
	// QueueDepth is the gateway's live count of requests waiting for a model.
	QueueDepth(modelID string) int
	// GPU reports the device's total and currently free memory, or ok=false if it
	// cannot be read. The scheduler uses it to account for memory it does not own.
	GPU(ctx context.Context) (total, free int64, ok bool)
}

// AgentPool resolves an agent client from a replica's recorded URL.
type AgentPool interface {
	For(agentURL string) *agentclient.Client
}

type defaultPool struct {
	mu sync.Mutex
	m  map[string]*agentclient.Client
}

// NewAgentPool returns a pool that reuses one client per agent URL, so connections
// are kept alive across swaps instead of being rebuilt per call.
func NewAgentPool() AgentPool { return &defaultPool{m: map[string]*agentclient.Client{}} }

func (p *defaultPool) For(url string) *agentclient.Client {
	p.mu.Lock()
	defer p.mu.Unlock()
	c, ok := p.m[url]
	if !ok {
		c = agentclient.New(url)
		p.m[url] = c
	}
	return c
}

// Controller executes swap plans.
type Controller struct {
	Fleet   Fleet
	Pool    AgentPool
	Policy  scheduler.EvictionPolicy
	Tracker *scheduler.Tracker
	Config  scheduler.Config
	GPU     scheduler.GPU
	Log     *slog.Logger
	// SwapTimeout bounds a detached swap. Zero means 15 minutes.
	SwapTimeout time.Duration

	// mu guards state. Every residency transition is recorded here, and the GPU is
	// mutated only while holding swapMu.
	mu    sync.RWMutex
	state map[scheduler.ModelID]*scheduler.Model

	// swapMu serialises swap execution. Two concurrent swaps would each compute a
	// plan against a VRAM budget the other is about to change, and both would commit
	// — the classic double-booking that ends in a CUDA OOM halfway through a load.
	// One swap at a time costs concurrency the GPU could not have used anyway.
	swapMu sync.Mutex

	// inflightSwaps coalesces concurrent requests for the same model. Fifty requests
	// arriving for one sleeping model must produce exactly one wake, and the other
	// forty-nine must wait for it rather than queueing forty-nine more.
	flightMu      sync.Mutex
	inflightSwaps map[scheduler.ModelID]*swapCall
}

// errTransitionInProgress means the model is mid-wake or mid-start, so the right
// response is to wait rather than to act or to fail.
var errTransitionInProgress = errors.New("swap: transition already in progress")

// swapCall is one in-progress swap that later callers can join.
type swapCall struct {
	done chan struct{}
	err  error

	// blocked records why the swap has not finished yet, so a caller that gives up
	// can say something more useful than "deadline exceeded". Without it, a request
	// that waited out its whole budget behind the anti-thrash guard reports a bare
	// timeout and the operator has no idea the GPU was contended rather than slow.
	mu      sync.Mutex
	blocked error
}

func (s *swapCall) setBlocked(err error) {
	s.mu.Lock()
	s.blocked = err
	s.mu.Unlock()
}

func (s *swapCall) blockedErr() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.blocked
}

// New builds a controller.
func New(fleet Fleet, pool AgentPool, policy scheduler.EvictionPolicy,
	cfg scheduler.Config, gpu scheduler.GPU, log *slog.Logger) *Controller {
	return &Controller{
		Fleet: fleet, Pool: pool, Policy: policy,
		Tracker: scheduler.NewTracker(60 * time.Second),
		Config:  cfg, GPU: gpu, Log: log,
		state:         map[scheduler.ModelID]*scheduler.Model{},
		inflightSwaps: map[scheduler.ModelID]*swapCall{},
	}
}

// Observe records a request arrival so the demand statistics stay current. Called by
// the gateway on every request, before Ensure.
func (c *Controller) Observe(model string) {
	c.Tracker.Observe(scheduler.ModelID(model), time.Now(), 1)
}

// Snapshot builds the ClusterState a policy reasons over.
func (c *Controller) Snapshot(ctx context.Context) (*scheduler.ClusterState, error) {
	models, err := c.Fleet.Models(ctx)
	if err != nil {
		return nil, err
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	now := time.Now()
	cs := &scheduler.ClusterState{
		Now: now, GPU: c.GPU, Config: c.Config,
		Models: make(map[scheduler.ModelID]*scheduler.Model, len(models)),
	}

	for _, m := range models {
		id := scheduler.ModelID(m.ID)
		known, ok := c.state[id]
		if !ok {
			known = &scheduler.Model{ID: id, State: scheduler.StateCold}
			c.state[id] = known
		}

		// Static facts are refreshed from the registry every snapshot, because a
		// redeploy can change a model's footprint and a stale footprint is how the
		// budget silently stops matching the device.
		known.VRAMBytes = m.FootprintVRAMBytes
		known.HostBytes = m.FootprintHostBytes
		known.ColdStart = time.Duration(m.ColdStartMS) * time.Millisecond
		known.WakeFromHost = time.Duration(m.WakeHostMS) * time.Millisecond
		known.WakeFromDisk = time.Duration(m.WakeDiskMS) * time.Millisecond
		known.Pinned = m.Pinned
		known.InFlight = c.Fleet.InFlight(m.ID)
		known.QueueDepth = c.Fleet.QueueDepth(m.ID)

		// Release a transition that never completed.
		//
		// A model left in WAKING or STARTING is the worst possible state to be stuck
		// in: it counts against the VRAM budget, it is never evictable ("transition
		// in progress"), and it is never servable -- so it holds memory hostage and
		// fails every request for itself, permanently. Any lost wake, agent restart
		// or dropped goroutine can produce it. Time-bounding the state means the
		// system recovers on its own instead of needing a restart.
		if c.transitionStuck(known, now) {
			c.Log.Warn("transition exceeded its deadline; releasing the model",
				"model", known.ID, "state", known.State.String(),
				"stuck_for", now.Sub(known.TransitionAt).Round(time.Second))
			known.State = scheduler.StateFailed
			metrics.SwapFailures.WithLabelValues(m.ID, "stuck_transition").Inc()
		}

		// A model whose footprint was never measured would be budgeted as free and
		// could be admitted without evicting anything, overcommitting the device.
		// Fall back to the utilisation ceiling, which is the one number that is
		// always known and is a true upper bound.
		if known.VRAMBytes <= 0 {
			known.VRAMBytes = int64(float64(c.GPU.TotalBytes) * m.GPUMemoryUtilization)
		}
		if known.HostBytes <= 0 {
			known.HostBytes = known.VRAMBytes / 2
		}

		cp := *known
		cs.Models[id] = &cp
	}

	// Budget against what the device actually reports, not against an assumed total.
	//
	// Plenty of VRAM belongs to things the scheduler does not track: the desktop
	// compositor on a workstation (~1.1 GB here), another tenant's process, and the
	// ~230 MiB each sleeping model keeps for its CUDA context. Budgeting on
	// total-minus-a-fixed-reservation over-commits by exactly that amount, and the
	// failure arrives as "No available memory for the cache blocks" a minute into a
	// cold start -- long after the decision that caused it.
	//
	// Treating everything we do not account for as reserved makes the budget
	// self-correcting, and makes the scheduler safe to run on a GPU it does not have
	// to itself.
	if total, free, ok := c.Fleet.GPU(ctx); ok && total > 0 {
		var ours int64
		for _, m := range cs.Models {
			if m.State.HoldsVRAM() {
				ours += m.VRAMBytes
			}
		}
		foreign := total - free - ours
		if foreign < 0 {
			foreign = 0
		}
		cs.GPU.TotalBytes = total
		// Keep the configured reservation on top as a safety margin for allocator
		// fragmentation, which no counter reports.
		cs.GPU.ReservedBytes = foreign + c.GPU.ReservedBytes
	}

	// Stamp the measured arrival rates and trends the scoring policies read.
	c.Tracker.Apply(cs)
	return cs, nil
}

// Ensure makes a model servable, waking or starting it if necessary.
//
// This is the gateway's hook on the request path. It blocks until the model is
// ACTIVE or the context expires, which is exactly what makes tiering invisible to
// clients: a request for a sleeping model is slow, not failed.
func (c *Controller) Ensure(ctx context.Context, model string) error {
	id := scheduler.ModelID(model)
	c.Observe(model)

	if c.isActive(id) {
		metrics.ResidencyHits.WithLabelValues(model).Inc()
		return nil
	}

	// Coalesce. The first caller performs the swap; everyone else waits on it.
	c.flightMu.Lock()
	if call, ok := c.inflightSwaps[id]; ok {
		c.flightMu.Unlock()
		metrics.ResidencyMisses.WithLabelValues(model, "coalesced").Inc()
		return c.await(ctx, call, model)
	}
	call := &swapCall{done: make(chan struct{})}
	c.inflightSwaps[id] = call
	c.flightMu.Unlock()

	// Run the swap on a context detached from the caller's.
	//
	// This is not a nicety. A swap is a sequence of GPU state transitions with real
	// side effects, and it is shared: fifty requests coalesce onto one wake. Binding
	// it to the triggering request's context means that whenever that one client
	// times out or disconnects, the wake is aborted part-way through -- leaving a
	// model neither asleep nor resident, failing every request that was waiting on
	// it, and losing the work. A benchmark run ending was enough to cancel every
	// swap in flight, which is how this was found.
	//
	// So the swap runs to completion or fails on its own terms, and the caller
	// merely waits for it. A client that gives up stops waiting; the model still
	// becomes resident, which is what the next request wants anyway.
	swapCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), c.swapTimeout())
	go func() {
		defer cancel()
		defer close(call.done)
		defer func() {
			c.flightMu.Lock()
			delete(c.inflightSwaps, id)
			c.flightMu.Unlock()
		}()
		call.err = c.swapInWithRetry(swapCtx, id, c.thrashBudget(ctx), call)
	}()

	return c.await(ctx, call, model)
}

// await blocks until the swap finishes or the caller gives up.
func (c *Controller) await(ctx context.Context, call *swapCall, model string) error {
	select {
	case <-call.done:
		return call.err
	case <-ctx.Done():
		// The swap keeps running; only this caller stops waiting. Report what it was
		// waiting on, which is the actionable part.
		if blocked := call.blockedErr(); blocked != nil {
			return fmt.Errorf("gave up waiting for %q to become servable: %w", model, blocked)
		}
		return fmt.Errorf("gave up waiting for %q to become servable: %w", model, ctx.Err())
	}
}

// swapTimeout bounds a detached swap. Generous, because a cold start of a large model
// legitimately takes minutes, and abandoning one half-done is worse than waiting.
func (c *Controller) swapTimeout() time.Duration {
	if c.SwapTimeout > 0 {
		return c.SwapTimeout
	}
	return 15 * time.Minute
}

// thrashBudget is how long to keep retrying a swap that is blocked only by the
// anti-thrash guard.
//
// Bounded by the caller's own deadline rather than by the detached swap timeout:
// retrying for fifteen minutes to satisfy a request that gave up after three would
// be pure waste, and unlike a wake in progress there is no work to protect — nothing
// has touched the GPU yet.
func (c *Controller) thrashBudget(ctx context.Context) time.Duration {
	const (
		floor = 5 * time.Second
		ceil  = 2 * time.Minute
	)
	budget := ceil
	if deadline, ok := ctx.Deadline(); ok {
		budget = time.Until(deadline)
	}
	if budget > ceil {
		budget = ceil
	}
	if budget < floor {
		budget = floor
	}
	return budget
}

// swapInWithRetry retries while the only obstacle is the anti-thrash guard.
//
// ErrWouldThrash means enough VRAM exists in principle but every candidate is inside
// its minimum-residency window -- a condition that resolves on its own as those
// windows expire. Returning it to the client would turn a momentarily contended GPU
// into a 503, which is precisely the behaviour the guard exists to avoid; the point
// of distinguishing it from ErrCannotFit is that one is worth waiting out and the
// other is not.
//
// The wait is bounded by the caller's context, which the gateway sets from its queue
// timeout, so a request still fails eventually rather than hanging forever.
func (c *Controller) swapInWithRetry(ctx context.Context, id scheduler.ModelID,
	thrashBudget time.Duration, call *swapCall) error {
	const (
		minBackoff = 150 * time.Millisecond
		maxBackoff = 2 * time.Second
	)
	backoff := minBackoff
	attempts := 0
	giveUp := time.After(thrashBudget)

	for {
		err := c.swapIn(ctx, id)
		if !errors.Is(err, scheduler.ErrWouldThrash) && !errors.Is(err, errTransitionInProgress) {
			if call != nil {
				call.setBlocked(nil)
			}
			if attempts > 0 && err == nil {
				c.Log.Info("swap succeeded after waiting out a residency window",
					"model", id, "attempts", attempts+1)
			}
			return err
		}
		attempts++
		if call != nil {
			call.setBlocked(err)
		}

		select {
		case <-ctx.Done():
			return fmt.Errorf("model %q could not be admitted: %w", id, scheduler.ErrWouldThrash)
		case <-giveUp:
			if errors.Is(err, errTransitionInProgress) {
				return fmt.Errorf("model %q was still loading after %s", id, thrashBudget)
			}
			// Report the thrash condition, not a context error: "every candidate was
			// inside its residency window for the whole budget" is actionable,
			// "deadline exceeded" is not.
			return fmt.Errorf("model %q could not be admitted within %s: %w",
				id, thrashBudget, scheduler.ErrWouldThrash)
		case <-time.After(backoff):
		}
		if backoff < maxBackoff {
			backoff *= 2
		}
	}
}

func (c *Controller) isActive(id scheduler.ModelID) bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	m, ok := c.state[id]
	return ok && m.State == scheduler.StateActive
}

// swapIn plans and executes everything needed to make id servable.
func (c *Controller) swapIn(ctx context.Context, id scheduler.ModelID) error {
	// Serialise against other swaps. Taken before planning, not after: a plan
	// computed against a budget that another swap then changes is a plan that
	// overcommits the device.
	c.swapMu.Lock()
	defer c.swapMu.Unlock()

	// Re-check under the lock. Another swap may have brought this model up while
	// this one waited, which is the common case under a burst.
	if c.isActive(id) {
		metrics.ResidencyHits.WithLabelValues(string(id)).Inc()
		return nil
	}

	cs, err := c.Snapshot(ctx)
	if err != nil {
		return err
	}
	target := cs.Models[id]
	if target == nil {
		return fmt.Errorf("swap: model %q is not deployed", id)
	}
	metrics.ResidencyMisses.WithLabelValues(string(id), scheduler.TierOf(target.State).String()).Inc()

	// One line per swap decision, with the budget the controller believes it has.
	// Without this, a benchmark that reports zero evictions on an oversubscribed
	// fleet is unexplainable after the fact: there is no way to tell whether the
	// scheduler declined to evict or never saw a reason to.
	c.Log.Info("planning swap",
		"target", id, "target_state", target.State.String(),
		"target_bytes", target.VRAMBytes,
		"vram_used", cs.VRAMUsed(), "vram_free", cs.VRAMFree(),
		"budget", cs.GPU.Budget(), "need", cs.Need(id),
		"resident", residentIDs(cs), "evictable", len(cs.EvictableSet(id)),
		// Why each resident model could not be evicted. "Nothing was evictable" is a
		// useless benchmark artefact; "everything was inside its residency window" or
		// "a drain is still in progress" is a diagnosis.
		"blockers", cs.Blockers(id))

	// A model already mid-transition produces a no-op plan: the planner's job is to
	// avoid issuing a second wake for something already waking, which is what makes
	// coalescing work. But a no-op plan is not the same as a servable model, and
	// returning success here told the gateway to route to a replica that was still
	// loading -- which then failed to find a ready one and answered 503. Signal that
	// the caller should wait instead.
	if target.State == scheduler.StateWaking || target.State == scheduler.StateStarting {
		return errTransitionInProgress
	}

	plan, err := scheduler.NewPlanner(c.Policy).Plan(cs, id)
	if err != nil {
		if errors.Is(err, scheduler.ErrWouldThrash) {
			// Temporary: every candidate is inside its minimum residency window, so
			// waiting genuinely helps. Surfaced as a distinct error so the gateway
			// can retry rather than failing the request.
			metrics.Thrash.WithLabelValues(string(id), c.Policy.Name()).Inc()
		}
		return err
	}
	if plan.NoOp() {
		return nil
	}

	c.Log.Info("executing swap plan",
		"target", id, "policy", plan.Policy, "steps", len(plan.Steps),
		"need_bytes", plan.NeedBytes, "freed_bytes", plan.FreedBytes,
		"estimated", plan.EstimatedCost)

	return c.execute(ctx, cs, plan)
}

// transitionStuck reports whether a model has sat in a transitional state past the
// deadline for it. The allowance is generous because a cold start of a large model
// legitimately takes minutes; the point is that it is bounded at all.
func (c *Controller) transitionStuck(m *scheduler.Model, now time.Time) bool {
	var limit time.Duration
	switch m.State {
	case scheduler.StateWaking:
		limit = 3 * time.Minute
	case scheduler.StateStarting:
		limit = c.swapTimeout()
	case scheduler.StateDraining:
		// Draining is already bounded by its own deadline inside drain(); this only
		// catches a drain whose executor disappeared entirely.
		limit = c.Config.DrainDeadline + time.Minute
	default:
		return false
	}
	return !m.TransitionAt.IsZero() && now.Sub(m.TransitionAt) > limit
}

// residentIDs lists the models the controller believes are holding VRAM, for the
// swap decision log.
func residentIDs(cs *scheduler.ClusterState) []string {
	var out []string
	for _, id := range cs.SortedIDs() {
		if m := cs.Models[id]; m.State.HoldsVRAM() {
			out = append(out, fmt.Sprintf("%s=%s", id, m.State))
		}
	}
	return out
}

// execute runs a plan's steps in order.
//
// The ordering the planner produced is not an implementation detail that can be
// relaxed here: every victim drains before any victim sleeps, and all sleeps finish
// before the target wakes. Starting the wake while a victim still holds its arena is
// how a swap turns into a CUDA OOM, and the failure arrives a minute into a load.
func (c *Controller) execute(ctx context.Context, cs *scheduler.ClusterState, plan *scheduler.Plan) error {
	for _, step := range plan.Steps {
		var err error
		switch step.Kind {
		case scheduler.StepDrain:
			err = c.drain(ctx, step.Model, step.Deadline)
		case scheduler.StepSleep:
			err = c.sleep(ctx, step.Model, step.To, plan.Policy)
		case scheduler.StepStop:
			err = c.stop(ctx, step.Model)
		case scheduler.StepWake:
			err = c.wake(ctx, step.Model)
		case scheduler.StepStart:
			err = c.start(ctx, step.Model)
		}
		if err != nil {
			c.Log.Error("swap step failed", "step", step.String(), "err", err)
			c.setState(step.Model, scheduler.StateFailed)
			return fmt.Errorf("swap step %s: %w", step, err)
		}
	}
	_ = cs
	return nil
}

// drain stops routing to a model and waits for its in-flight requests to finish.
//
// Bounded by a deadline on purpose. Unbounded draining lets one slow request block a
// swap indefinitely, which would make eviction latency unbounded and therefore make
// the whole tiering system unusable under load. Hitting the deadline is recorded, not
// hidden: the vLLM sleep call that follows uses mode=wait, so requests still in
// flight get a chance to finish rather than being aborted outright.
func (c *Controller) drain(ctx context.Context, id scheduler.ModelID, deadline time.Duration) error {
	// Flipping to DRAINING is what removes the model from the gateway's routable set,
	// so no new requests join while the existing ones finish.
	c.setState(id, scheduler.StateDraining)

	if deadline <= 0 {
		deadline = c.Config.DrainDeadline
	}
	ctx, cancel := context.WithTimeout(ctx, deadline)
	defer cancel()

	t := time.NewTicker(25 * time.Millisecond)
	defer t.Stop()
	for {
		if n := c.Fleet.InFlight(string(id)); n == 0 {
			return nil
		}
		select {
		case <-ctx.Done():
			n := c.Fleet.InFlight(string(id))
			metrics.DrainTimeouts.WithLabelValues(string(id)).Inc()
			c.Log.Warn("drain deadline hit; sleeping with requests still in flight",
				"model", id, "in_flight", n, "deadline", deadline)
			return nil
		case <-t.C:
		}
	}
}

func (c *Controller) sleep(ctx context.Context, id scheduler.ModelID, tier scheduler.Tier, policy string) error {
	level := 1
	if tier == scheduler.TierDisk {
		level = 2
	}
	start := time.Now()
	err := c.forEachReplica(ctx, id, func(cl *agentclient.Client, replicaID string) error {
		// mode=wait rather than abort: the scheduler has already drained, so this is
		// the backstop for a request that arrived in the gap between the drain
		// finishing and the sleep being issued. Aborting there would kill a request
		// that was admitted legitimately.
		_, err := cl.Sleep(ctx, replicaID, level, "wait")
		return err
	})
	if err != nil {
		metrics.SwapFailures.WithLabelValues(string(id), "sleep").Inc()
		return err
	}

	c.setState(id, scheduler.SleepStateFor(tier))
	metrics.SwapDuration.WithLabelValues(string(id), "gpu", tier.String()).Observe(time.Since(start).Seconds())
	metrics.Evictions.WithLabelValues(string(id), tier.String(), policy).Inc()
	c.Log.Info("model slept", "model", id, "tier", tier, "took", time.Since(start).Round(time.Millisecond))
	return nil
}

func (c *Controller) wake(ctx context.Context, id scheduler.ModelID) error {
	from := scheduler.TierOf(c.stateOf(id))
	c.setState(id, scheduler.StateWaking)

	start := time.Now()
	err := c.forEachReplica(ctx, id, func(cl *agentclient.Client, replicaID string) error {
		// The agent's Wake performs the reload_weights step a level-2 wake needs and
		// then generates a token before declaring the replica servable. Skipping
		// either produces a replica that is healthy and silently wrong (see M0).
		_, err := cl.Wake(ctx, replicaID)
		return err
	})
	if err != nil {
		metrics.SwapFailures.WithLabelValues(string(id), "wake").Inc()
		return err
	}

	c.setStateResident(id)
	metrics.SwapDuration.WithLabelValues(string(id), from.String(), "gpu").Observe(time.Since(start).Seconds())
	metrics.Wakes.WithLabelValues(string(id), from.String()).Inc()
	c.Log.Info("model woken", "model", id, "from", from, "took", time.Since(start).Round(time.Millisecond))
	return nil
}

func (c *Controller) start(ctx context.Context, id scheduler.ModelID) error {
	// A cold start means no process exists, which is a different operation from a
	// wake and cannot be served by one. The `cold` eviction policy stops a victim's
	// process outright -- that is what makes it the no-tiering baseline -- and the
	// node agent forgets a stopped replica entirely. Asking it to wake one then fails
	// with "unknown replica", and the model is unreachable for the rest of the run.
	//
	// So a cold start re-creates the process from the registry's spec. The spec is
	// the deployment's own, so a restarted replica is configured identically to the
	// one that was stopped; anything else would silently change a benchmark arm's
	// conditions mid-run.
	spec, err := c.specFor(ctx, id)
	if err != nil {
		metrics.SwapFailures.WithLabelValues(string(id), "start").Inc()
		return err
	}

	c.setState(id, scheduler.StateStarting)
	start := time.Now()

	err = c.forEachReplica(ctx, id, func(cl *agentclient.Client, replicaID string) error {
		s := spec
		s.Replica = replicaID
		// Idempotent on the agent side: an already-ACTIVE replica is returned as-is
		// without touching the GPU, so this is also the safe recovery path when the
		// controller's view has drifted.
		_, err := cl.Start(ctx, s)
		return err
	})
	if err != nil {
		metrics.SwapFailures.WithLabelValues(string(id), "start").Inc()
		return err
	}

	c.setStateResident(id)
	metrics.ColdStarts.WithLabelValues(string(id)).Inc()
	metrics.SwapDuration.WithLabelValues(string(id), "none", "gpu").Observe(time.Since(start).Seconds())
	return nil
}

// specFor rebuilds the vLLM spec for a model from the registry.
func (c *Controller) specFor(ctx context.Context, id scheduler.ModelID) (nodeagent.Spec, error) {
	models, err := c.Fleet.Models(ctx)
	if err != nil {
		return nodeagent.Spec{}, err
	}
	for _, m := range models {
		if m.ID != string(id) {
			continue
		}
		return nodeagent.Spec{
			Model: m.ID, HFRepo: m.HFRepo, Revision: m.Revision,
			GPUMemoryUtilization: m.GPUMemoryUtilization,
			MaxModelLen:          m.MaxModelLen,
			EnforceEager:         m.EnforceEager,
		}, nil
	}
	return nodeagent.Spec{}, fmt.Errorf("swap: model %q is not in the registry", id)
}

func (c *Controller) stop(ctx context.Context, id scheduler.ModelID) error {
	start := time.Now()
	err := c.forEachReplica(ctx, id, func(cl *agentclient.Client, replicaID string) error {
		return cl.Stop(ctx, replicaID)
	})
	if err != nil {
		metrics.SwapFailures.WithLabelValues(string(id), "stop").Inc()
		return err
	}
	c.setState(id, scheduler.StateCold)
	metrics.SwapDuration.WithLabelValues(string(id), "gpu", "none").Observe(time.Since(start).Seconds())
	metrics.Evictions.WithLabelValues(string(id), "none", c.Policy.Name()).Inc()
	return nil
}

// forEachReplica applies an operation to every replica of a model.
//
// Sequential, not concurrent. Replicas of one model on one GPU contend for the same
// device memory, and waking three at once triples the peak allocation during the
// transition — the moment when memory is tightest.
func (c *Controller) forEachReplica(ctx context.Context, id scheduler.ModelID,
	op func(*agentclient.Client, string) error) error {

	reps, err := c.Fleet.Replicas(ctx, string(id))
	if err != nil {
		return err
	}
	if len(reps) == 0 {
		return fmt.Errorf("swap: model %q has no replicas", id)
	}
	sort.Slice(reps, func(i, j int) bool { return reps[i].ID < reps[j].ID })

	for _, r := range reps {
		if err := op(c.Pool.For(r.AgentURL), r.ID); err != nil {
			return fmt.Errorf("replica %s on %s: %w", r.ID, r.AgentURL, err)
		}
	}
	return nil
}

func (c *Controller) setState(id scheduler.ModelID, s scheduler.State) {
	c.mu.Lock()
	defer c.mu.Unlock()
	m, ok := c.state[id]
	if !ok {
		m = &scheduler.Model{ID: id}
		c.state[id] = m
	}
	m.State = s
	m.TransitionAt = time.Now()
}

// setStateResident marks a model ACTIVE and starts its minimum-residency clock. The
// clock starts here, at the moment it becomes servable, rather than when the swap was
// requested — otherwise a slow wake would eat into the window that exists to stop the
// model being evicted again immediately.
func (c *Controller) setStateResident(id scheduler.ModelID) {
	c.mu.Lock()
	defer c.mu.Unlock()
	m, ok := c.state[id]
	if !ok {
		m = &scheduler.Model{ID: id}
		c.state[id] = m
	}
	m.State = scheduler.StateActive
	m.ResidentSince = time.Now()
	m.TransitionAt = time.Time{}
}

func (c *Controller) stateOf(id scheduler.ModelID) scheduler.State {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if m, ok := c.state[id]; ok {
		return m.State
	}
	return scheduler.StateCold
}

// States returns a snapshot of every model's residency, for the gateway's routing
// decisions and for /v1/models.
func (c *Controller) States() map[string]string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make(map[string]string, len(c.state))
	for id, m := range c.state {
		out[string(id)] = m.State.String()
	}
	return out
}

// Sync reconciles the controller's view with what the agents actually report.
//
// The controller's own map is authoritative for decisions, but it can drift: an agent
// restarts, a replica dies, a swap fails halfway. Reconciling against the agents means
// the budget is eventually corrected by observation rather than trusted indefinitely.
func (c *Controller) Sync(ctx context.Context, observed map[string]string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for id, state := range observed {
		m, ok := c.state[scheduler.ModelID(id)]
		if !ok {
			parsed, err := scheduler.ParseState(state)
			if err != nil {
				continue
			}
			c.state[scheduler.ModelID(id)] = &scheduler.Model{
				ID: scheduler.ModelID(id), State: parsed, ResidentSince: time.Now(),
			}
			continue
		}
		// Never overwrite a transition the controller is driving. The agent's view is
		// a poll and is behind; believing it would make a wake in progress look cold
		// and trigger a second one.
		switch m.State {
		case scheduler.StateDraining, scheduler.StateWaking, scheduler.StateStarting:
			continue
		}
		if parsed, err := scheduler.ParseState(state); err == nil && parsed != m.State {
			if parsed == scheduler.StateActive && m.State != scheduler.StateActive {
				m.ResidentSince = time.Now()
			}
			m.State = parsed
		}
	}
}

// RunTick performs the policy's background work: demoting idle host-tier sleepers and
// prewarming models whose demand is rising.
func (c *Controller) RunTick(ctx context.Context, every time.Duration) {
	t := time.NewTicker(every)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			c.tickOnce(ctx)
		}
	}
}

func (c *Controller) tickOnce(ctx context.Context) {
	cs, err := c.Snapshot(ctx)
	if err != nil {
		c.Log.Warn("tick snapshot failed", "err", err)
		return
	}
	actions := c.Policy.Tick(cs)
	if len(actions) == 0 {
		return
	}

	// Background work must never block or delay a request-path swap, so it competes
	// for the same lock and gives up rather than waiting.
	if !c.swapMu.TryLock() {
		return
	}
	defer c.swapMu.Unlock()

	for _, a := range actions {
		c.Log.Info("background tiering action", "kind", a.Kind, "model", a.Model, "reason", a.Reason)
		var err error
		switch a.Kind {
		case scheduler.ActionPrewarm:
			err = c.wake(ctx, a.Model)
		case scheduler.ActionDemote:
			// Demotion is host -> disk, which needs no drain: the model is already
			// asleep and serving nothing.
			err = c.sleep(ctx, a.Model, scheduler.TierDisk, c.Policy.Name())
		case scheduler.ActionStop:
			err = c.stop(ctx, a.Model)
		}
		if err != nil {
			c.Log.Warn("background tiering action failed", "kind", a.Kind, "model", a.Model, "err", err)
		}
	}
}

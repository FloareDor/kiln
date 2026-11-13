package scheduler

import (
	"errors"
	"testing"
	"time"
)

const (
	gb  = int64(1) << 30
	mib = int64(1) << 20
)

var t0 = time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

// fixture builds a cluster with a 24GB budget and the named models, all ACTIVE and
// long past their residency window unless a test says otherwise.
type fixture struct{ c *ClusterState }

func newFixture(budgetGB int64) *fixture {
	return &fixture{c: &ClusterState{
		Now:    t0,
		GPU:    GPU{Node: "n1", TotalBytes: budgetGB * gb, ReservedBytes: 0},
		Config: Config{MinResidency: 15 * time.Second, HysteresisBytes: 0, DrainDeadline: 10 * time.Second, HostRAMBudget: 64 * gb, HostTierMinRate: 0.02, IdleDemoteAfter: 5 * time.Minute},
		Models: map[ModelID]*Model{},
	}}
}

type opt func(*Model)

func state(s State) opt          { return func(m *Model) { m.State = s } }
func pinned() opt                { return func(m *Model) { m.Pinned = true } }
func rate(r float64) opt         { return func(m *Model) { m.Rate = r } }
func trend(tr float64) opt       { return func(m *Model) { m.Trend = tr } }
func idle(d time.Duration) opt   { return func(m *Model) { m.LastRequest = t0.Add(-d) } }
func inflight(n int) opt         { return func(m *Model) { m.InFlight = n } }
func resident(d time.Duration) opt { return func(m *Model) { m.ResidentSince = t0.Add(-d) } }
func wakeHost(d time.Duration) opt { return func(m *Model) { m.WakeFromHost = d } }
func wakeDisk(d time.Duration) opt { return func(m *Model) { m.WakeFromDisk = d } }
func host(b int64) opt           { return func(m *Model) { m.HostBytes = b } }

func (f *fixture) add(id ModelID, vramGB float64, opts ...opt) *Model {
	m := &Model{
		ID: id, State: StateActive,
		VRAMBytes: int64(vramGB * float64(gb)),
		HostBytes: int64(vramGB * float64(gb) * 0.7),
		ColdStart: 40 * time.Second, WakeFromHost: 1 * time.Second, WakeFromDisk: 6 * time.Second,
		LastRequest: t0.Add(-time.Minute), ResidentSince: t0.Add(-time.Hour),
	}
	for _, o := range opts {
		o(m)
	}
	f.c.Models[id] = m
	return m
}

func ids(evs []Eviction) []ModelID {
	out := make([]ModelID, len(evs))
	for i, e := range evs {
		out[i] = e.Model
	}
	return out
}

func eq(t *testing.T, got, want []ModelID) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("victims: got %v, want %v", got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("victims: got %v, want %v", got, want)
		}
	}
}

// --------------------------------------------------------------- accounting

func TestVRAMAccountingCountsInFlightTransitions(t *testing.T) {
	f := newFixture(24)
	f.add("a", 8)
	f.add("b", 6, state(StateWaking))   // allocation already in flight
	f.add("c", 4, state(StateDraining)) // still resident while finishing requests
	f.add("d", 5, state(StateSleepingL1))
	f.add("e", 5, state(StateSleepingL2))

	// A WAKING model's memory is already being claimed. Not counting it is how you
	// admit a second model into memory the first is halfway through taking.
	if got, want := f.c.VRAMUsed(), 18*gb; got != want {
		t.Fatalf("VRAMUsed = %d GB, want %d GB", got/gb, want/gb)
	}
	if got, want := f.c.VRAMFree(), 6*gb; got != want {
		t.Fatalf("VRAMFree = %d GB, want %d GB", got/gb, want/gb)
	}
	// Only the L1 sleeper holds pinned host memory; the L2 one dropped its weights.
	if got, want := f.c.HostRAMUsed(), int64(3.5*float64(gb)); got != want {
		t.Fatalf("HostRAMUsed = %d, want %d", got, want)
	}
}

func TestReservedVRAMIsNotHandedOut(t *testing.T) {
	f := newFixture(8)
	// 1GB held back for the CUDA context, allocator fragmentation and, on a laptop,
	// the desktop compositor.
	f.c.GPU.ReservedBytes = 1 * gb
	f.add("a", 6)
	f.add("target", 2, state(StateCold))

	// Naive accounting says 2GB free and the 2GB model fits. With the reservation it
	// does not, and something must be evicted.
	if got := f.c.VRAMFree(); got != 1*gb {
		t.Fatalf("VRAMFree = %d GB, want 1 GB", got/gb)
	}
	if need := f.c.Need("target"); need != 1*gb {
		t.Fatalf("Need = %d, want 1 GB", need)
	}
}

func TestNeedIncludesHysteresis(t *testing.T) {
	f := newFixture(10)
	f.c.Config.HysteresisBytes = 512 * mib
	f.add("a", 8)
	f.add("target", 2, state(StateSleepingL1))

	// Exactly 2GB free for a 2GB model: it fits, but leaves the GPU at 100% so the
	// next request evicts again. Hysteresis makes the planner free a little more.
	if need := f.c.Need("target"); need != 512*mib {
		t.Fatalf("Need = %d MiB, want 512 MiB", need/mib)
	}
}

// ---------------------------------------------------------------- state machine

func TestIllegalTransitionsRejected(t *testing.T) {
	legal := [][2]State{
		{StateCold, StateStarting},
		{StateActive, StateDraining},
		{StateDraining, StateSleepingL1},
		{StateDraining, StateSleepingL2},
		{StateDraining, StateActive}, // abandoned drain: cheaper than completing it
		{StateSleepingL1, StateWaking},
		{StateSleepingL1, StateSleepingL2}, // demotion
		{StateWaking, StateActive},
		{StateActive, StateFailed},
	}
	for _, p := range legal {
		if err := ValidateTransition(p[0], p[1]); err != nil {
			t.Errorf("expected %s -> %s to be legal: %v", p[0], p[1], err)
		}
	}
	illegal := [][2]State{
		{StateActive, StateSleepingL1},  // must drain first, or in-flight requests die
		{StateSleepingL2, StateActive},  // must pass through WAKING
		{StateCold, StateActive},        // must pass through STARTING
		{StateSleepingL2, StateSleepingL1},
	}
	for _, p := range illegal {
		if err := ValidateTransition(p[0], p[1]); err == nil {
			t.Errorf("expected %s -> %s to be rejected", p[0], p[1])
		}
	}
}

// ------------------------------------------------------------------- anti-thrash

func TestPinnedModelIsNeverEvicted(t *testing.T) {
	f := newFixture(10)
	f.add("critical", 8, pinned(), idle(time.Hour)) // coldest thing on the box
	f.add("target", 4, state(StateSleepingL1))

	for _, name := range PolicyNames() {
		p, _ := NewPolicy(name)
		_, err := p.PickVictims(f.c, "target")
		if !errors.Is(err, ErrCannotFit) {
			t.Errorf("%s: expected ErrCannotFit with only a pinned victim, got %v", name, err)
		}
	}
}

func TestMinResidencyBlocksThrashAndIsDistinguishedFromImpossible(t *testing.T) {
	f := newFixture(10)
	// Just woken: evicting it now is the textbook thrash, two models ping-ponging on
	// a GPU that fits one.
	f.add("hot", 8, resident(2*time.Second))
	f.add("target", 8, state(StateSleepingL1))

	p := NewLRU()
	_, err := p.PickVictims(f.c, "target")
	if !errors.Is(err, ErrWouldThrash) {
		t.Fatalf("expected ErrWouldThrash inside the residency window, got %v", err)
	}

	// The distinction matters: a thrash block is temporary and the caller should
	// queue, whereas ErrCannotFit means waiting will never help.
	f.c.Now = t0.Add(20 * time.Second)
	evs, err := p.PickVictims(f.c, "target")
	if err != nil {
		t.Fatalf("after the window expired: %v", err)
	}
	eq(t, ids(evs), []ModelID{"hot"})
}

func TestModelLargerThanBudgetCannotFit(t *testing.T) {
	f := newFixture(8)
	f.add("a", 4)
	f.add("huge", 16, state(StateCold))

	_, err := NewLRU().PickVictims(f.c, "huge")
	if !errors.Is(err, ErrCannotFit) {
		t.Fatalf("expected ErrCannotFit for a model bigger than the GPU, got %v", err)
	}
}

// --------------------------------------------------------------------- policies

func TestLRUEvictsLeastRecentlyUsed(t *testing.T) {
	f := newFixture(12)
	f.add("recent", 4, idle(5*time.Second))
	f.add("stale", 4, idle(30*time.Minute))
	f.add("middling", 4, idle(2*time.Minute))
	f.add("target", 4, state(StateSleepingL1))

	evs, err := NewLRU().PickVictims(f.c, "target")
	if err != nil {
		t.Fatal(err)
	}
	eq(t, ids(evs), []ModelID{"stale"})
}

func TestLFUDecayResistsAScanThatDefeatsLRU(t *testing.T) {
	// The classic cache-pollution case: a steadily-busy model that happens not to
	// have been touched in the last few seconds, against models that each received a
	// single recent request. LRU evicts the busy one; lfu-decay keeps it.
	f := newFixture(12)
	f.add("busy", 4, idle(10*time.Second), rate(5.0))
	f.add("scan1", 4, idle(1*time.Second), rate(0.01))
	f.add("scan2", 4, idle(2*time.Second), rate(0.01))
	f.add("target", 4, state(StateSleepingL1))

	lru, _ := NewLRU().PickVictims(f.c, "target")
	if ids(lru)[0] != "busy" {
		t.Fatalf("precondition: LRU should fall for the scan, picked %v", ids(lru))
	}

	lfu, err := NewLFUDecay().PickVictims(f.c, "target")
	if err != nil {
		t.Fatal(err)
	}
	if got := ids(lfu)[0]; got == "busy" {
		t.Fatalf("lfu-decay evicted the busy model: %v", ids(lfu))
	}
}

func TestCostAwarePrefersFreeingOneLargeCheapModelOverManySmallHotOnes(t *testing.T) {
	// This is the case the per-GB normalisation exists for. The GPU is exactly full,
	// so admitting the 6GB target needs either the one big cold model or all three
	// small warm ones.
	f := newFixture(12)
	f.add("big-cold", 6, rate(0.01), wakeHost(2*time.Second), host(1*gb))
	f.add("small-hot-1", 2, rate(4.0), wakeHost(1*time.Second), host(1*gb))
	f.add("small-hot-2", 2, rate(4.0), wakeHost(1*time.Second), host(1*gb))
	f.add("small-hot-3", 2, rate(4.0), wakeHost(1*time.Second), host(1*gb))
	f.add("target", 6, state(StateSleepingL1))

	evs, err := NewCostAware().PickVictims(f.c, "target")
	if err != nil {
		t.Fatal(err)
	}
	eq(t, ids(evs), []ModelID{"big-cold"})

	// LRU is blind to all of this: with every model equally stale it picks on
	// recency alone and needs three evictions to free the same 6GB.
	for _, m := range f.c.Models {
		m.LastRequest = t0.Add(-time.Minute)
	}
	f.c.Models["big-cold"].LastRequest = t0.Add(-time.Second)
	lru, _ := NewLRU().PickVictims(f.c, "target")
	if len(lru) != 3 {
		t.Fatalf("expected LRU to need 3 small victims here, got %v", ids(lru))
	}
}

func TestCostAwareScoresAgainstTheTierTheVictimActuallyLandsIn(t *testing.T) {
	// Two identical models except for wake cost from disk. With the host tier full,
	// both would go to disk, so the disk cost is what should drive the choice —
	// scoring against the host cost would pick the wrong victim.
	f := newFixture(12)
	f.c.Config.HostRAMBudget = 0 // host tier exhausted: everything goes to disk
	f.c.GPU.TotalBytes = 10 * gb // only 2GB free, so one victim must go
	f.add("cheap-disk", 4, rate(1.0), wakeHost(1*time.Second), wakeDisk(2*time.Second))
	f.add("dear-disk", 4, rate(1.0), wakeHost(1*time.Second), wakeDisk(60*time.Second))
	f.add("target", 4, state(StateSleepingL2))

	evs, err := NewCostAware().PickVictims(f.c, "target")
	if err != nil {
		t.Fatal(err)
	}
	eq(t, ids(evs), []ModelID{"cheap-disk"})
	if evs[0].To != TierDisk {
		t.Fatalf("expected the victim to go to disk when host RAM is exhausted, got %s", evs[0].To)
	}
}

func TestColdBaselineStopsProcessesRatherThanSleepingThem(t *testing.T) {
	f := newFixture(10)
	f.add("victim", 6, idle(time.Hour), rate(10.0)) // hot, but the baseline has no tiers
	f.add("target", 6, state(StateCold))

	evs, err := NewCold().PickVictims(f.c, "target")
	if err != nil {
		t.Fatal(err)
	}
	if evs[0].To != TierNone {
		t.Fatalf("cold baseline must stop the process, got tier %s", evs[0].To)
	}
}

// ------------------------------------------------------------------ tier choice

func TestTierChoiceFallsBackToDiskWhenHostBudgetIsExhausted(t *testing.T) {
	f := newFixture(20)
	f.c.Config.HostRAMBudget = 5 * gb
	f.add("sleeper", 6, state(StateSleepingL1), host(4*gb)) // already holds 4GB of it
	victim := f.add("victim", 6, rate(10.0), host(4*gb))    // hot enough to deserve host

	tier, why := f.c.ChooseTier(victim)
	if tier != TierDisk {
		t.Fatalf("expected disk when only 1GB of host budget remains, got %s (%s)", tier, why)
	}
}

func TestColdModelsAreNotGivenScarceHostMemory(t *testing.T) {
	f := newFixture(20)
	victim := f.add("victim", 6, rate(0.001), host(1*gb)) // nobody is asking for it

	tier, _ := f.c.ChooseTier(victim)
	if tier != TierDisk {
		t.Fatalf("a model below the host-tier rate threshold should go to disk, got %s", tier)
	}
}

// ---------------------------------------------------------------------- planner

func TestPlanDrainsEveryVictimBeforeSleepingAny(t *testing.T) {
	// Interleaving a wake with an outstanding drain is how you get a CUDA OOM: the
	// wake allocates while the victim still holds its arena.
	f := newFixture(8) // full: admitting the 8GB target needs both victims gone
	f.add("v1", 4, idle(time.Hour), inflight(2))
	f.add("v2", 4, idle(50*time.Minute), inflight(1))
	f.add("target", 8, state(StateSleepingL1))

	plan, err := NewPlanner(NewLRU()).Plan(f.c, "target")
	if err != nil {
		t.Fatal(err)
	}

	seenSleep := false
	for _, s := range plan.Steps {
		switch s.Kind {
		case StepDrain:
			if seenSleep {
				t.Fatalf("drain issued after a sleep: %v", plan.Steps)
			}
		case StepSleep, StepStop:
			seenSleep = true
		case StepWake, StepStart:
			if !seenSleep {
				t.Fatalf("wake issued before any victim was slept: %v", plan.Steps)
			}
		}
	}
	last := plan.Steps[len(plan.Steps)-1]
	if last.Kind != StepWake || last.Model != "target" {
		t.Fatalf("plan must end by waking the target, ended with %v", last)
	}
	if plan.FreedBytes != 8*gb {
		t.Fatalf("FreedBytes = %d GB, want 8", plan.FreedBytes/gb)
	}
}

func TestPlanForAnAlreadyActiveModelIsANoOp(t *testing.T) {
	f := newFixture(12)
	f.add("target", 4)
	plan, err := NewPlanner(NewLRU()).Plan(f.c, "target")
	if err != nil {
		t.Fatal(err)
	}
	if !plan.NoOp() {
		t.Fatalf("expected a no-op plan, got %v", plan.Steps)
	}
}

func TestSwapCoalescing(t *testing.T) {
	// Fifty concurrent requests for one sleeping model must produce exactly one wake.
	// The second and subsequent planners see WAKING and return an empty plan; without
	// that, each request issues its own wake and the GPU is asked to load the same
	// weights fifty times.
	f := newFixture(12)
	f.add("target", 4, state(StateSleepingL1))
	pl := NewPlanner(NewLRU())

	first, err := pl.Plan(f.c, "target")
	if err != nil {
		t.Fatal(err)
	}
	if len(first.Steps) != 1 || first.Steps[0].Kind != StepWake {
		t.Fatalf("first request should produce one wake, got %v", first.Steps)
	}

	f.c.Models["target"].State = StateWaking // the control plane has begun the wake
	wakes := 0
	for i := 0; i < 49; i++ {
		p, err := pl.Plan(f.c, "target")
		if err != nil {
			t.Fatal(err)
		}
		for _, s := range p.Steps {
			if s.Kind == StepWake {
				wakes++
			}
		}
	}
	if wakes != 0 {
		t.Fatalf("49 piggybacking requests issued %d extra wakes, want 0", wakes)
	}
}

func TestPlanCancelsAnAbandonedDrainInsteadOfCompletingIt(t *testing.T) {
	// A model being evicted suddenly becomes wanted again. Finishing the eviction and
	// then waking costs two swaps; cancelling costs none.
	f := newFixture(12)
	f.add("target", 4, state(StateDraining))

	plan, err := NewPlanner(NewLRU()).Plan(f.c, "target")
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Steps) != 1 || plan.Steps[0].Kind != StepWake {
		t.Fatalf("expected a single cancelling wake, got %v", plan.Steps)
	}
}

func TestPlanEstimateUsesMeasuredCostsAndParallelDrains(t *testing.T) {
	f := newFixture(12)
	f.add("v1", 4, idle(time.Hour), inflight(3))
	f.add("v2", 4, idle(50*time.Minute), inflight(3))
	f.add("target", 8, state(StateSleepingL2), wakeDisk(6*time.Second))

	plan, err := NewPlanner(NewLRU()).Plan(f.c, "target")
	if err != nil {
		t.Fatal(err)
	}
	// Drains run concurrently, so two draining victims cost one deadline, not two.
	want := f.c.Config.DrainDeadline + 6*time.Second
	if plan.EstimatedCost != want {
		t.Fatalf("EstimatedCost = %s, want %s", plan.EstimatedCost, want)
	}
}

func TestPlanApplyKeepsAccountingConsistent(t *testing.T) {
	f := newFixture(12)
	f.add("v", 8, idle(time.Hour))
	f.add("target", 8, state(StateSleepingL1))

	plan, err := NewPlanner(NewLRU()).Plan(f.c, "target")
	if err != nil {
		t.Fatal(err)
	}
	if err := plan.Apply(f.c); err != nil {
		t.Fatal(err)
	}
	if f.c.Models["target"].State != StateActive {
		t.Fatalf("target state = %s, want ACTIVE", f.c.Models["target"].State)
	}
	if got := f.c.VRAMUsed(); got != 8*gb {
		t.Fatalf("after the swap VRAMUsed = %d GB, want 8", got/gb)
	}
}

// -------------------------------------------------------------------- prewarming

func TestPredictivePrewarmOnlyUsesIdleHeadroom(t *testing.T) {
	f := newFixture(12)
	f.c.Config.PrewarmHorizon = 30 * time.Second
	f.c.Config.PrewarmMinRate = 0.1
	f.add("resident", 8)
	f.add("rising", 3, state(StateSleepingL1), rate(0.05), trend(0.01))

	// 4GB free, the sleeper needs 3GB: prewarm it.
	acts := NewPredictivePrewarm().Tick(f.c)
	if len(acts) != 1 || acts[0].Kind != ActionPrewarm || acts[0].Model != "rising" {
		t.Fatalf("expected one prewarm of 'rising', got %v", acts)
	}

	// Now the GPU is full. A speculative wake that forces an eviction is a bet the
	// policy has no business making, so it must do nothing.
	f.c.Models["resident"].VRAMBytes = 11 * gb
	if acts := NewPredictivePrewarm().Tick(f.c); len(acts) != 0 {
		t.Fatalf("prewarm must not evict to make room for a guess, got %v", acts)
	}
}

func TestPredictivePrewarmIgnoresFallingDemand(t *testing.T) {
	f := newFixture(12)
	f.c.Config.PrewarmMinRate = 0.1
	f.add("falling", 3, state(StateSleepingL1), rate(0.5), trend(-0.02))

	if acts := NewPredictivePrewarm().Tick(f.c); len(acts) != 0 {
		t.Fatalf("expected no prewarm for falling demand, got %v", acts)
	}
}

func TestIdleHostSleepersAreDemotedToDisk(t *testing.T) {
	f := newFixture(12)
	f.c.Config.IdleDemoteAfter = 5 * time.Minute
	f.add("stale-sleeper", 4, state(StateSleepingL1), idle(10*time.Minute))
	f.add("fresh-sleeper", 4, state(StateSleepingL1), idle(1*time.Minute))

	acts := NewLRU().Tick(f.c)
	if len(acts) != 1 || acts[0].Kind != ActionDemote || acts[0].Model != "stale-sleeper" {
		t.Fatalf("expected one demotion of the stale sleeper, got %v", acts)
	}
}

func TestColdBaselineHasNoBackgroundWork(t *testing.T) {
	f := newFixture(12)
	f.add("x", 4, state(StateSleepingL1), idle(time.Hour))
	if acts := NewCold().Tick(f.c); len(acts) != 0 {
		t.Fatalf("the cold baseline has no tiers to manage, got %v", acts)
	}
}

// ---------------------------------------------------------------- determinism

func TestVictimChoiceIsDeterministic(t *testing.T) {
	// Go randomises map iteration order. A policy that iterates a map directly
	// returns different victims on identical input, which makes the A/B comparison
	// in experiment 3 meaningless.
	for _, name := range PolicyNames() {
		var first []ModelID
		for run := 0; run < 50; run++ {
			f := newFixture(12)
			for _, id := range []ModelID{"a", "b", "c", "d"} {
				f.add(id, 3, idle(time.Hour), rate(1.0)) // deliberately all identical
			}
			f.add("target", 3, state(StateSleepingL1))

			p, _ := NewPolicy(name)
			evs, err := p.PickVictims(f.c, "target")
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			got := ids(evs)
			if run == 0 {
				first = got
				continue
			}
			eq(t, got, first)
		}
	}
}

// ------------------------------------------------------------------- tracker

func TestTrackerRateConvergesToArrivalsPerSecond(t *testing.T) {
	// The EWMA increment is scaled so a model receiving r req/s settles near r. If it
	// did not, HostTierMinRate and PrewarmMinRate would be in arbitrary units and
	// impossible to set.
	tr := NewTracker(30 * time.Second)
	now := t0
	const perSec = 4
	for i := 0; i < 30*perSec*6; i++ {
		now = now.Add(time.Second / perSec)
		tr.Observe("m", now, 1)
	}
	got := tr.Rate("m", now)
	if got < perSec*0.85 || got > perSec*1.15 {
		t.Fatalf("rate converged to %.3f req/s, want ~%d", got, perSec)
	}
}

func TestTrackerRateDecaysByHalfPerHalfLife(t *testing.T) {
	tr := NewTracker(time.Minute)
	now := t0
	for i := 0; i < 600; i++ {
		now = now.Add(100 * time.Millisecond)
		tr.Observe("m", now, 1)
	}
	before := tr.Rate("m", now)
	after := tr.Rate("m", now.Add(time.Minute))
	ratio := after / before
	if ratio < 0.45 || ratio > 0.55 {
		t.Fatalf("after one half-life the rate was %.3f of its peak, want ~0.5", ratio)
	}
}

func TestTrackerTrendIsPositiveWhileDemandClimbs(t *testing.T) {
	tr := NewTracker(30 * time.Second)
	now := t0
	// Quiet, then a sustained ramp.
	for i := 0; i < 60; i++ {
		now = now.Add(time.Second)
		if i%20 == 0 {
			tr.Observe("m", now, 1)
		}
	}
	quiet := tr.Trend("m", now)
	for i := 0; i < 120; i++ {
		now = now.Add(250 * time.Millisecond)
		tr.Observe("m", now, 1)
	}
	climbing := tr.Trend("m", now)
	if climbing <= quiet {
		t.Fatalf("trend did not rise with demand: quiet %.5f, climbing %.5f", quiet, climbing)
	}
	if climbing <= 0 {
		t.Fatalf("trend during a ramp = %.5f, want > 0", climbing)
	}

	// And it must go negative once demand stops, or the predictive policy would keep
	// prewarming models nobody wants.
	falling := tr.Trend("m", now.Add(2*time.Minute))
	if falling >= 0 {
		t.Fatalf("trend after demand stopped = %.5f, want < 0", falling)
	}
}

func TestTrackerApplyStampsStateForPolicies(t *testing.T) {
	tr := NewTracker(30 * time.Second)
	now := t0
	for i := 0; i < 100; i++ {
		now = now.Add(100 * time.Millisecond)
		tr.Observe("hot", now, 1)
	}
	f := newFixture(12)
	f.c.Now = now
	f.add("hot", 4)
	f.add("cold", 4)

	tr.Apply(f.c)
	if f.c.Models["hot"].Rate <= 0 {
		t.Fatal("Apply did not stamp the observed rate")
	}
	if f.c.Models["cold"].Rate != 0 {
		t.Fatal("Apply invented a rate for an unobserved model")
	}
	if !f.c.Models["hot"].LastRequest.Equal(now) {
		t.Fatalf("LastRequest = %s, want %s", f.c.Models["hot"].LastRequest, now)
	}
}

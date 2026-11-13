package swap

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sairavig/kiln/internal/agentclient"
	"github.com/sairavig/kiln/internal/nodeagent"
	"github.com/sairavig/kiln/internal/scheduler"
	"github.com/sairavig/kiln/internal/store"
)

const gb = int64(1) << 30

func discardLog() *slog.Logger { return slog.New(slog.NewTextHandler(io.Discard, nil)) }

// fakeAgent is a node agent that records the operations it is asked to perform and
// how they interleave. The controller's correctness lives almost entirely in
// ordering and concurrency, so the fake records a global sequence rather than just
// per-replica counts.
type fakeAgent struct {
	*httptest.Server

	mu    sync.Mutex
	ops   []string
	state map[string]string // replica -> state

	// opDelay makes every operation take time, so a test can observe two swaps
	// overlapping if the controller fails to serialise them.
	opDelay time.Duration
	// concurrent tracks the peak number of simultaneous GPU operations.
	live, peak atomic.Int64
	failWake   atomic.Bool
}

func newFakeAgent(delay time.Duration) *fakeAgent {
	f := &fakeAgent{state: map[string]string{}, opDelay: delay}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /status", f.status)
	mux.HandleFunc("POST /replicas", f.start)
	mux.HandleFunc("POST /replicas/{id}/sleep", f.sleep)
	mux.HandleFunc("POST /replicas/{id}/wake", f.wake)
	mux.HandleFunc("DELETE /replicas/{id}", f.stop)
	f.Server = httptest.NewServer(mux)
	return f
}

func (f *fakeAgent) enter() {
	n := f.live.Add(1)
	for {
		p := f.peak.Load()
		if n <= p || f.peak.CompareAndSwap(p, n) {
			break
		}
	}
	time.Sleep(f.opDelay)
}

func (f *fakeAgent) leave() { f.live.Add(-1) }

func (f *fakeAgent) record(op string) {
	f.mu.Lock()
	f.ops = append(f.ops, op)
	f.mu.Unlock()
}

func (f *fakeAgent) opLog() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.ops...)
}

func (f *fakeAgent) count(prefix string) int {
	n := 0
	for _, op := range f.opLog() {
		if strings.HasPrefix(op, prefix) {
			n++
		}
	}
	return n
}

func (f *fakeAgent) status(w http.ResponseWriter, r *http.Request) {
	_ = json.NewEncoder(w).Encode(nodeagent.NodeStatus{Node: "fake"})
}

func (f *fakeAgent) sleep(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Level int    `json:"level"`
		Mode  string `json:"mode"`
	}
	_ = json.NewDecoder(r.Body).Decode(&body)
	id := r.PathValue("id")
	f.record(fmt.Sprintf("sleep:%s:L%d:%s", id, body.Level, body.Mode))
	f.enter()
	defer f.leave()
	f.mu.Lock()
	f.state[id] = fmt.Sprintf("SLEEPING_L%d", body.Level)
	f.mu.Unlock()
	_ = json.NewEncoder(w).Encode(nodeagent.Status{State: f.state[id]})
}

func (f *fakeAgent) wake(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	f.record("wake:" + id)
	if f.failWake.Load() {
		http.Error(w, `{"error":"wake failed"}`, http.StatusInternalServerError)
		return
	}
	// Mirror the real agent: a stopped replica is forgotten, so waking it is an
	// error rather than a silent no-op. Without this the fake would hide the very
	// bug that broke the cold baseline.
	f.mu.Lock()
	_, known := f.state[id]
	f.mu.Unlock()
	if !known {
		http.Error(w, `{"error":"nodeagent: unknown replica"}`, http.StatusInternalServerError)
		return
	}
	f.enter()
	defer f.leave()
	f.mu.Lock()
	f.state[id] = "ACTIVE"
	f.mu.Unlock()
	_ = json.NewEncoder(w).Encode(nodeagent.Status{State: "ACTIVE", WakeHostMS: 100})
}

func (f *fakeAgent) stop(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	f.record("stop:" + id)
	f.enter()
	defer f.leave()
	f.mu.Lock()
	delete(f.state, id) // the real agent forgets a stopped replica entirely
	f.mu.Unlock()
	w.WriteHeader(http.StatusNoContent)
}

func (f *fakeAgent) start(w http.ResponseWriter, r *http.Request) {
	var spec nodeagent.Spec
	_ = json.NewDecoder(r.Body).Decode(&spec)
	f.record("start:" + spec.Replica)
	f.enter()
	defer f.leave()
	f.mu.Lock()
	f.state[spec.Replica] = "ACTIVE"
	f.mu.Unlock()
	_ = json.NewEncoder(w).Encode(nodeagent.Status{
		Spec: spec, State: "ACTIVE", ColdStartMS: 35000})
}

// staticFleet is a deployed fleet with controllable in-flight counts.
type staticFleet struct {
	models   []*store.Model
	replicas map[string][]*store.Replica

	mu       sync.Mutex
	inflight map[string]int
	gpuTotal int64
	gpuFree  int64
}

func (s *staticFleet) Models(context.Context) ([]*store.Model, error) { return s.models, nil }

func (s *staticFleet) Replicas(_ context.Context, id string) ([]*store.Replica, error) {
	return s.replicas[id], nil
}

func (s *staticFleet) InFlight(id string) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.inflight[id]
}

func (s *staticFleet) setInFlight(id string, n int) {
	s.mu.Lock()
	s.inflight[id] = n
	s.mu.Unlock()
}

func (s *staticFleet) QueueDepth(string) int { return 0 }

// GPU reports no reading by default, so the budget tests keep using the configured
// total and reservation. Tests that care about foreign memory set gpuTotal/gpuFree.
func (s *staticFleet) GPU(context.Context) (int64, int64, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.gpuTotal == 0 {
		return 0, 0, false
	}
	return s.gpuTotal, s.gpuFree, true
}

type fixedPool struct{ c *agentclient.Client }

func (p fixedPool) For(string) *agentclient.Client { return p.c }

// harness wires a controller to one fake agent with the given models.
type harness struct {
	ctrl  *Controller
	agent *fakeAgent
	fleet *staticFleet
}

func newHarness(t *testing.T, budgetGB int64, opDelay time.Duration,
	policy scheduler.EvictionPolicy, models ...*store.Model) *harness {
	t.Helper()

	agent := newFakeAgent(opDelay)
	t.Cleanup(agent.Close)

	fleet := &staticFleet{
		models:   models,
		replicas: map[string][]*store.Replica{},
		inflight: map[string]int{},
	}
	for _, m := range models {
		fleet.replicas[m.ID] = []*store.Replica{
			{ID: m.ID + "--0", ModelID: m.ID, AgentURL: agent.URL, State: "ACTIVE"},
		}
		agent.state[m.ID+"--0"] = "ACTIVE" // deployed, as after a real deploy job
	}

	cfg := scheduler.DefaultConfig()
	cfg.MinResidency = 0 // overridden per test where it is the subject
	cfg.DrainDeadline = time.Second
	cfg.HysteresisBytes = 0

	ctrl := New(fleet, fixedPool{agentclient.New(agent.URL)}, policy, cfg,
		scheduler.GPU{Node: "n1", TotalBytes: budgetGB * gb}, discardLog())
	return &harness{ctrl: ctrl, agent: agent, fleet: fleet}
}

func model(id string, vramGB float64) *store.Model {
	return &store.Model{
		ID: id, HFRepo: "repo/" + id, Replicas: 1,
		FootprintVRAMBytes: int64(vramGB * float64(gb)),
		FootprintHostBytes: int64(vramGB * float64(gb) * 0.7),
		ColdStartMS:        40000, WakeHostMS: 600, WakeDiskMS: 1800,
		GPUMemoryUtilization: 0.35, Status: store.ModelReady,
	}
}

func setState(c *Controller, id string, s scheduler.State) {
	c.setState(scheduler.ModelID(id), s)
	if s == scheduler.StateActive {
		c.setStateResident(scheduler.ModelID(id))
	}
}

// ------------------------------------------------------------------- basics

func TestEnsureOnAnActiveModelDoesNothing(t *testing.T) {
	h := newHarness(t, 8, 0, scheduler.NewLRU(), model("a", 3))
	setState(h.ctrl, "a", scheduler.StateActive)

	if err := h.ctrl.Ensure(context.Background(), "a"); err != nil {
		t.Fatal(err)
	}
	if ops := h.agent.opLog(); len(ops) != 0 {
		t.Fatalf("a resident model triggered GPU work: %v", ops)
	}
}

func TestEnsureWakesASleepingModel(t *testing.T) {
	h := newHarness(t, 8, 0, scheduler.NewLRU(), model("a", 3))
	setState(h.ctrl, "a", scheduler.StateSleepingL1)

	if err := h.ctrl.Ensure(context.Background(), "a"); err != nil {
		t.Fatal(err)
	}
	if h.agent.count("wake:") != 1 {
		t.Fatalf("expected exactly one wake, got %v", h.agent.opLog())
	}
	if got := h.ctrl.States()["a"]; got != "ACTIVE" {
		t.Fatalf("state after wake = %s, want ACTIVE", got)
	}
}

func TestEnsureEvictsToMakeRoom(t *testing.T) {
	// 8GB budget, two 5GB models: only one fits, so admitting the second must evict
	// the first.
	h := newHarness(t, 8, 0, scheduler.NewLRU(), model("hot", 5), model("cold", 5))
	setState(h.ctrl, "hot", scheduler.StateActive)
	setState(h.ctrl, "cold", scheduler.StateSleepingL1)

	if err := h.ctrl.Ensure(context.Background(), "cold"); err != nil {
		t.Fatal(err)
	}

	ops := h.agent.opLog()
	if h.agent.count("sleep:hot") != 1 {
		t.Fatalf("the resident model was not evicted: %v", ops)
	}
	if h.agent.count("wake:cold") != 1 {
		t.Fatalf("the target was not woken: %v", ops)
	}
	// Ordering is the whole point: waking while the victim still holds its arena is
	// how a swap becomes a CUDA OOM a minute into a load.
	sleepAt, wakeAt := -1, -1
	for i, op := range ops {
		if strings.HasPrefix(op, "sleep:hot") {
			sleepAt = i
		}
		if strings.HasPrefix(op, "wake:cold") {
			wakeAt = i
		}
	}
	if sleepAt > wakeAt {
		t.Fatalf("woke the target before sleeping the victim: %v", ops)
	}
	states := h.ctrl.States()
	if states["hot"] != "SLEEPING_L1" && states["hot"] != "SLEEPING_L2" {
		t.Fatalf("victim state = %s, want a sleeping state", states["hot"])
	}
	if states["cold"] != "ACTIVE" {
		t.Fatalf("target state = %s, want ACTIVE", states["cold"])
	}
}

func TestSleepUsesWaitModeNotAbort(t *testing.T) {
	// The scheduler drains first, so vLLM's own wait is the backstop for a request
	// that arrived in the gap. Aborting there would kill a request that had been
	// admitted legitimately.
	h := newHarness(t, 8, 0, scheduler.NewLRU(), model("hot", 5), model("cold", 5))
	setState(h.ctrl, "hot", scheduler.StateActive)
	setState(h.ctrl, "cold", scheduler.StateSleepingL1)

	if err := h.ctrl.Ensure(context.Background(), "cold"); err != nil {
		t.Fatal(err)
	}
	for _, op := range h.agent.opLog() {
		if strings.HasPrefix(op, "sleep:") && !strings.HasSuffix(op, ":wait") {
			t.Fatalf("sleep issued with the wrong mode: %s", op)
		}
	}
}

// -------------------------------------------------------------- coalescing

func TestFiftyConcurrentRequestsProduceExactlyOneWake(t *testing.T) {
	// The headline coalescing property. Without it, a burst for one sleeping model
	// asks the GPU to load the same weights fifty times.
	h := newHarness(t, 8, 60*time.Millisecond, scheduler.NewLRU(), model("a", 3))
	setState(h.ctrl, "a", scheduler.StateSleepingL1)

	var wg sync.WaitGroup
	errs := make([]error, 50)
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			errs[i] = h.ctrl.Ensure(context.Background(), "a")
		}(i)
	}
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Fatalf("request %d failed: %v", i, err)
		}
	}
	if n := h.agent.count("wake:"); n != 1 {
		t.Fatalf("50 concurrent requests produced %d wakes, want 1", n)
	}
}

func TestCoalescedWaitersSeeTheFailure(t *testing.T) {
	// A piggybacking request must not be told the model is ready when the wake it
	// waited on failed.
	h := newHarness(t, 8, 40*time.Millisecond, scheduler.NewLRU(), model("a", 3))
	setState(h.ctrl, "a", scheduler.StateSleepingL1)
	h.agent.failWake.Store(true)

	var wg sync.WaitGroup
	errs := make([]error, 20)
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			errs[i] = h.ctrl.Ensure(context.Background(), "a")
		}(i)
	}
	wg.Wait()

	for i, err := range errs {
		if err == nil {
			t.Fatalf("request %d reported success despite a failed wake", i)
		}
	}
}

func TestSwapsAreSerialisedAgainstEachOther(t *testing.T) {
	// Two swaps planned concurrently each compute against a VRAM budget the other is
	// about to change, and both commit. The peak-concurrency counter in the fake
	// agent is what catches it.
	h := newHarness(t, 12, 50*time.Millisecond, scheduler.NewLRU(),
		model("a", 5), model("b", 5), model("c", 5))
	setState(h.ctrl, "a", scheduler.StateActive)
	setState(h.ctrl, "b", scheduler.StateSleepingL1)
	setState(h.ctrl, "c", scheduler.StateSleepingL1)

	var wg sync.WaitGroup
	for _, id := range []string{"b", "c"} {
		wg.Add(1)
		go func(id string) {
			defer wg.Done()
			_ = h.ctrl.Ensure(context.Background(), id)
		}(id)
	}
	wg.Wait()

	if peak := h.agent.peak.Load(); peak > 1 {
		t.Fatalf("peak concurrent GPU operations = %d, want 1; swaps were not serialised", peak)
	}
}

// ------------------------------------------------------------------ draining

func TestDrainWaitsForInFlightRequests(t *testing.T) {
	h := newHarness(t, 8, 0, scheduler.NewLRU(), model("hot", 5), model("cold", 5))
	h.ctrl.Config.DrainDeadline = 5 * time.Second
	setState(h.ctrl, "hot", scheduler.StateActive)
	setState(h.ctrl, "cold", scheduler.StateSleepingL1)

	// Two requests in flight on the victim. Sleeping now would kill them.
	h.fleet.setInFlight("hot", 2)

	done := make(chan error, 1)
	go func() { done <- h.ctrl.Ensure(context.Background(), "cold") }()

	// The swap must not have slept the victim yet.
	time.Sleep(150 * time.Millisecond)
	if h.agent.count("sleep:hot") != 0 {
		t.Fatal("the victim was slept while requests were still in flight")
	}

	h.fleet.setInFlight("hot", 0)
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the swap did not complete after the victim drained")
	}
	if h.agent.count("sleep:hot") != 1 {
		t.Fatalf("expected one sleep after draining, got %v", h.agent.opLog())
	}
}

func TestDrainGivesUpAtItsDeadline(t *testing.T) {
	// Unbounded draining lets one slow request block a swap forever, which makes
	// eviction latency unbounded and the whole system unusable under load.
	h := newHarness(t, 8, 0, scheduler.NewLRU(), model("stuck", 5), model("cold", 5))
	h.ctrl.Config.DrainDeadline = 250 * time.Millisecond
	setState(h.ctrl, "stuck", scheduler.StateActive)
	setState(h.ctrl, "cold", scheduler.StateSleepingL1)
	h.fleet.setInFlight("stuck", 1) // never finishes

	start := time.Now()
	if err := h.ctrl.Ensure(context.Background(), "cold"); err != nil {
		t.Fatal(err)
	}
	elapsed := time.Since(start)

	if elapsed > 3*time.Second {
		t.Fatalf("the swap took %s; the drain deadline did not bound it", elapsed)
	}
	if h.agent.count("sleep:stuck") != 1 {
		t.Fatalf("the victim was never slept: %v", h.agent.opLog())
	}
	if h.ctrl.States()["cold"] != "ACTIVE" {
		t.Fatal("the target never became active")
	}
}

func TestDrainingModelIsImmediatelyUnroutable(t *testing.T) {
	// Flipping to DRAINING is what stops new requests joining. If the state change
	// came after the drain finished, requests would keep arriving throughout and the
	// drain would never converge.
	h := newHarness(t, 8, 0, scheduler.NewLRU(), model("hot", 5), model("cold", 5))
	h.ctrl.Config.DrainDeadline = 3 * time.Second
	setState(h.ctrl, "hot", scheduler.StateActive)
	setState(h.ctrl, "cold", scheduler.StateSleepingL1)
	h.fleet.setInFlight("hot", 1)

	go func() { _ = h.ctrl.Ensure(context.Background(), "cold") }()

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if h.ctrl.States()["hot"] == "DRAINING" {
			h.fleet.setInFlight("hot", 0)
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("the victim never entered DRAINING: %v", h.ctrl.States())
}

// --------------------------------------------------------------- anti-thrash

func TestNothingIsEvictedInsideItsResidencyWindow(t *testing.T) {
	// The guard itself, observed one level down from Ensure: planning reports the
	// condition rather than picking a victim. Ensure's job is to wait it out, which
	// TestThrashBlockIsWaitedOutRatherThanFailed covers; here the question is only
	// whether anything gets evicted too early.
	h := newHarness(t, 8, 0, scheduler.NewLRU(), model("justwoke", 5), model("want", 5))
	h.ctrl.Config.MinResidency = 10 * time.Second
	setState(h.ctrl, "justwoke", scheduler.StateActive) // resident as of now
	setState(h.ctrl, "want", scheduler.StateSleepingL1)

	if err := h.ctrl.swapIn(context.Background(), "want"); !errors.Is(err, scheduler.ErrWouldThrash) {
		t.Fatalf("got %v, want ErrWouldThrash", err)
	}
	if h.agent.count("sleep:") != 0 {
		t.Fatalf("something was evicted inside its residency window: %v", h.agent.opLog())
	}
}

func TestThrashBlockIsWaitedOutRatherThanFailed(t *testing.T) {
	// ErrWouldThrash means "wait and it will clear", so it must not reach the client.
	// Returning it would turn a momentarily contended GPU into a 503 -- the exact
	// outcome distinguishing it from ErrCannotFit exists to prevent.
	h := newHarness(t, 8, 0, scheduler.NewLRU(), model("justwoke", 5), model("want", 5))
	h.ctrl.Config.MinResidency = 400 * time.Millisecond
	setState(h.ctrl, "justwoke", scheduler.StateActive)
	setState(h.ctrl, "want", scheduler.StateSleepingL1)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	start := time.Now()
	if err := h.ctrl.Ensure(ctx, "want"); err != nil {
		t.Fatalf("expected the swap to succeed once the window expired: %v", err)
	}
	elapsed := time.Since(start)

	if elapsed < 300*time.Millisecond {
		t.Fatalf("succeeded in %s, before the residency window could expire", elapsed)
	}
	if h.agent.count("sleep:justwoke") != 1 {
		t.Fatalf("victim was not evicted after the window expired: %v", h.agent.opLog())
	}
	if h.ctrl.States()["want"] != "ACTIVE" {
		t.Fatalf("target is %s, want ACTIVE", h.ctrl.States()["want"])
	}
}

func TestThrashBlockStillFailsAtTheDeadline(t *testing.T) {
	// Waiting is bounded: a request must not hang forever behind a window that keeps
	// being renewed, and the error must name the real cause.
	h := newHarness(t, 8, 0, scheduler.NewLRU(), model("justwoke", 5), model("want", 5))
	h.ctrl.Config.MinResidency = time.Hour
	setState(h.ctrl, "justwoke", scheduler.StateActive)
	setState(h.ctrl, "want", scheduler.StateSleepingL1)

	ctx, cancel := context.WithTimeout(context.Background(), 600*time.Millisecond)
	defer cancel()

	err := h.ctrl.Ensure(ctx, "want")
	if !errors.Is(err, scheduler.ErrWouldThrash) {
		t.Fatalf("got %v, want an error wrapping ErrWouldThrash", err)
	}
}

func TestCannotFitFailsImmediatelyWithoutWaiting(t *testing.T) {
	// The other half of the distinction: waiting cannot help when everything is
	// pinned, so the request must fail at once rather than burning the deadline.
	pinned := model("critical", 5)
	pinned.Pinned = true
	h := newHarness(t, 8, 0, scheduler.NewLRU(), pinned, model("want", 5))
	setState(h.ctrl, "critical", scheduler.StateActive)
	setState(h.ctrl, "want", scheduler.StateSleepingL1)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	start := time.Now()
	err := h.ctrl.Ensure(ctx, "want")
	if !errors.Is(err, scheduler.ErrCannotFit) {
		t.Fatalf("got %v, want ErrCannotFit", err)
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Fatalf("took %s to report an impossible swap; it should fail fast", elapsed)
	}
}

func TestPinnedModelsAreNeverEvicted(t *testing.T) {
	pinned := model("critical", 5)
	pinned.Pinned = true
	h := newHarness(t, 8, 0, scheduler.NewLRU(), pinned, model("want", 5))
	setState(h.ctrl, "critical", scheduler.StateActive)
	setState(h.ctrl, "want", scheduler.StateSleepingL1)

	err := h.ctrl.Ensure(context.Background(), "want")
	if err == nil {
		t.Fatal("a pinned model was evicted")
	}
	if !errors.Is(err, scheduler.ErrCannotFit) {
		t.Fatalf("got %v, want ErrCannotFit", err)
	}
	if h.agent.count("sleep:critical") != 0 {
		t.Fatalf("the pinned model was slept: %v", h.agent.opLog())
	}
}

// ---------------------------------------------------------------- accounting

func TestBudgetAccountsForMemoryTheSchedulerDoesNotOwn(t *testing.T) {
	// A workstation GPU is never empty: the desktop compositor holds about a
	// gigabyte here, and every sleeping model keeps its CUDA context. Budgeting on
	// total-minus-a-fixed-reservation over-commits by exactly that much, and the
	// failure surfaces as "No available memory for the cache blocks" a minute into a
	// cold start, long after the decision that caused it.
	h := newHarness(t, 8, 0, scheduler.NewLRU(), model("a", 3))
	setState(h.ctrl, "a", scheduler.StateActive)

	// 8 GB device: 3 GB is ours, 2 GB belongs to something else, 3 GB free.
	h.fleet.mu.Lock()
	h.fleet.gpuTotal = 8 * gb
	h.fleet.gpuFree = 3 * gb
	h.fleet.mu.Unlock()

	cs, err := h.ctrl.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	// The 2 GB we do not account for must come out of the budget, not be handed to
	// the next model.
	if got, want := cs.GPU.ReservedBytes, 2*gb; got < want {
		t.Fatalf("reserved = %d bytes, want at least %d (foreign memory ignored)", got, want)
	}
	if free := cs.VRAMFree(); free > 3*gb {
		t.Fatalf("VRAMFree = %.2f GB, but only 3 GB is actually free", float64(free)/float64(gb))
	}
}

func TestUnmeasuredFootprintFallsBackToTheUtilisationCeiling(t *testing.T) {
	// A model whose footprint was never measured would otherwise be budgeted as
	// free, admitted without evicting anything, and overcommit the device.
	m := model("unmeasured", 0)
	m.FootprintVRAMBytes = 0
	m.GPUMemoryUtilization = 0.5

	h := newHarness(t, 8, 0, scheduler.NewLRU(), m)
	cs, err := h.ctrl.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	got := cs.Models["unmeasured"].VRAMBytes
	if want := int64(0.5 * float64(8*gb)); got != want {
		t.Fatalf("fallback footprint = %d, want %d (utilisation ceiling)", got, want)
	}
}

func TestSyncDoesNotOverwriteATransitionInProgress(t *testing.T) {
	// Agent polls are behind. Believing one mid-wake would make the model look cold
	// and trigger a second wake of something already loading.
	h := newHarness(t, 8, 0, scheduler.NewLRU(), model("a", 3))
	setState(h.ctrl, "a", scheduler.StateWaking)

	h.ctrl.Sync(context.Background(), map[string]string{"a": "SLEEPING_L1"})
	if got := h.ctrl.States()["a"]; got != "WAKING" {
		t.Fatalf("a stale poll overwrote an in-progress transition: %s", got)
	}

	// Once the transition settles, observation is authoritative again.
	setState(h.ctrl, "a", scheduler.StateActive)
	h.ctrl.Sync(context.Background(), map[string]string{"a": "FAILED"})
	if got := h.ctrl.States()["a"]; got != "FAILED" {
		t.Fatalf("a settled model ignored observation: %s", got)
	}
}

func TestResidencyClockStartsWhenTheModelBecomesServable(t *testing.T) {
	// If the clock started when the swap was requested, a slow wake would eat into
	// the window that exists to stop the model being evicted again immediately.
	h := newHarness(t, 8, 300*time.Millisecond, scheduler.NewLRU(), model("a", 3))
	setState(h.ctrl, "a", scheduler.StateSleepingL1)

	before := time.Now()
	if err := h.ctrl.Ensure(context.Background(), "a"); err != nil {
		t.Fatal(err)
	}
	h.ctrl.mu.RLock()
	resident := h.ctrl.state["a"].ResidentSince
	h.ctrl.mu.RUnlock()

	if resident.Sub(before) < 250*time.Millisecond {
		t.Fatalf("residency clock started %s after the request, before the wake finished",
			resident.Sub(before))
	}
}

// ------------------------------------------------------------------- policies

func TestEveryPolicyProducesAWorkingSwap(t *testing.T) {
	// A policy that compiles but produces an unexecutable plan would only show up in
	// a benchmark as a mysterious failure, so every registered arm is exercised.
	for _, name := range scheduler.PolicyNames() {
		t.Run(name, func(t *testing.T) {
			p, err := scheduler.NewPolicy(name)
			if err != nil {
				t.Fatal(err)
			}
			h := newHarness(t, 8, 0, p, model("victim", 5), model("target", 5))
			setState(h.ctrl, "victim", scheduler.StateActive)
			setState(h.ctrl, "target", scheduler.StateSleepingL1)

			if err := h.ctrl.Ensure(context.Background(), "target"); err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			if h.ctrl.States()["target"] != "ACTIVE" {
				t.Fatalf("%s: target is %s", name, h.ctrl.States()["target"])
			}
			// The cold baseline stops processes instead of sleeping them; everything
			// else uses a sleep tier. Both are valid evictions.
			if h.agent.count("sleep:victim")+h.agent.count("stop:victim") != 1 {
				t.Fatalf("%s: victim was not evicted exactly once: %v", name, h.agent.opLog())
			}
		})
	}
}

func TestColdPolicyStopsRatherThanSleeps(t *testing.T) {
	h := newHarness(t, 8, 0, scheduler.NewCold(), model("victim", 5), model("target", 5))
	setState(h.ctrl, "victim", scheduler.StateActive)
	setState(h.ctrl, "target", scheduler.StateSleepingL1)

	if err := h.ctrl.Ensure(context.Background(), "target"); err != nil {
		t.Fatal(err)
	}
	if h.agent.count("stop:victim") != 1 {
		t.Fatalf("the cold baseline did not stop the victim's process: %v", h.agent.opLog())
	}
	if h.agent.count("sleep:victim") != 0 {
		t.Fatalf("the cold baseline used sleep mode: %v", h.agent.opLog())
	}
}

func TestACancelledRequestDoesNotAbortTheSwap(t *testing.T) {
	// A swap is shared work with real GPU side effects: fifty requests coalesce onto
	// one wake. If the triggering request's cancellation killed it, one client timing
	// out would abort the wake, fail everyone waiting, and leave the model neither
	// asleep nor resident. A whole benchmark run's worth of swaps was lost to this.
	h := newHarness(t, 8, 400*time.Millisecond, scheduler.NewLRU(), model("a", 3))
	setState(h.ctrl, "a", scheduler.StateSleepingL1)

	ctx, cancel := context.WithTimeout(context.Background(), 80*time.Millisecond)
	defer cancel()

	// The caller gives up long before the wake can finish.
	if err := h.ctrl.Ensure(ctx, "a"); err == nil {
		t.Fatal("expected the caller to stop waiting")
	}

	// The wake must nonetheless complete, because the next request wants it.
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if h.ctrl.States()["a"] == "ACTIVE" {
			if h.agent.count("wake:") != 1 {
				t.Fatalf("expected exactly one wake, got %v", h.agent.opLog())
			}
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("the swap was aborted when its caller gave up; state is %s", h.ctrl.States()["a"])
}

func TestColdBaselineCanRestartWhatItStopped(t *testing.T) {
	// The cold baseline is only a fair arm if the model it killed can come back. The
	// agent forgets a stopped replica, so a cold start has to re-create the process
	// from the registry's spec; treating it as a wake fails with "unknown replica"
	// and the model is unreachable for the rest of the run -- which silently poisons
	// every later arm, because they share the agent.
	h := newHarness(t, 8, 0, scheduler.NewCold(), model("victim", 5), model("target", 5))
	setState(h.ctrl, "victim", scheduler.StateActive)
	setState(h.ctrl, "target", scheduler.StateSleepingL1)

	// First swap: the baseline stops the victim's process.
	if err := h.ctrl.Ensure(context.Background(), "target"); err != nil {
		t.Fatal(err)
	}
	if h.agent.count("stop:victim--0") != 1 {
		t.Fatalf("victim's process was not stopped: %v", h.agent.opLog())
	}
	if got := h.ctrl.States()["victim"]; got != "COLD" {
		t.Fatalf("victim state = %s, want COLD", got)
	}

	// Second swap: the stopped model is wanted again and must cold-start.
	h.ctrl.Config.MinResidency = 0
	if err := h.ctrl.Ensure(context.Background(), "victim"); err != nil {
		t.Fatalf("could not restart a model the cold baseline stopped: %v", err)
	}
	if h.agent.count("start:victim--0") != 1 {
		t.Fatalf("expected a cold start, got %v", h.agent.opLog())
	}
	if got := h.ctrl.States()["victim"]; got != "ACTIVE" {
		t.Fatalf("victim state after restart = %s, want ACTIVE", got)
	}
}

func TestColdStartUsesTheRegistrySpec(t *testing.T) {
	// A restarted replica must be configured exactly as the one that was stopped, or
	// a benchmark arm's conditions change silently halfway through.
	m := model("a", 3)
	m.HFRepo = "Qwen/Qwen2.5-0.5B-Instruct"
	m.MaxModelLen = 2048
	m.EnforceEager = true
	m.GPUMemoryUtilization = 0.31

	h := newHarness(t, 8, 0, scheduler.NewLRU(), m)
	setState(h.ctrl, "a", scheduler.StateCold)
	delete(h.agent.state, "a--0") // as if its process had been stopped

	if err := h.ctrl.Ensure(context.Background(), "a"); err != nil {
		t.Fatal(err)
	}
	spec, err := h.ctrl.specFor(context.Background(), "a")
	if err != nil {
		t.Fatal(err)
	}
	if spec.HFRepo != m.HFRepo || spec.MaxModelLen != 2048 ||
		!spec.EnforceEager || spec.GPUMemoryUtilization != 0.31 {
		t.Fatalf("cold start spec does not match the deployment: %+v", spec)
	}
}

func TestEnsureRespectsContextCancellation(t *testing.T) {
	// A client that gives up must not leave its request blocked forever behind a
	// swap it no longer cares about.
	h := newHarness(t, 8, 2*time.Second, scheduler.NewLRU(), model("a", 3), model("b", 3))
	setState(h.ctrl, "a", scheduler.StateSleepingL1)
	setState(h.ctrl, "b", scheduler.StateSleepingL1)

	go func() { _ = h.ctrl.Ensure(context.Background(), "a") }()
	time.Sleep(100 * time.Millisecond)

	ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
	defer cancel()
	start := time.Now()
	err := h.ctrl.Ensure(ctx, "a")
	if err == nil {
		t.Fatal("expected the cancelled request to return an error")
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Fatalf("cancellation took %s to take effect", elapsed)
	}
}

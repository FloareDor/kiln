package jobs

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/FloareDor/kiln/internal/store"
)

func discardLog() *slog.Logger { return slog.New(slog.NewTextHandler(io.Discard, nil)) }

func testStore(t *testing.T) *store.Store {
	t.Helper()
	dsn := os.Getenv("KILN_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("KILN_TEST_DATABASE_URL not set; skipping Postgres integration tests")
	}
	// Each integration suite gets its own schema. `go test ./...` runs packages in
	// parallel, so two suites sharing one set of tables would truncate each other's
	// rows mid-test and fail in ways that look like product bugs.
	dsn = withSchema(dsn, "jobs_test_schema")
	ctx := context.Background()
	s, err := store.Open(ctx, dsn)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if err := s.Migrate(ctx); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	if _, err := s.Pool.Exec(ctx,
		`TRUNCATE deploy_jobs, replicas, models, usage, quotas, api_keys, teams CASCADE`); err != nil {
		t.Fatalf("truncate: %v", err)
	}
	t.Cleanup(s.Close)
	return s
}

// scriptedExec stands in for a Deployer so the worker's claim, lease, retry and
// recovery behaviour can be tested with no node agent and no GPU. The worker's
// correctness is in that control flow, not in the HTTP calls.
type scriptedExec struct {
	mu       sync.Mutex
	calls    atomic.Int64
	deploy   func(ctx context.Context, m *store.Model) error
	deployed []string
	deleted  []string
}

func (s *scriptedExec) Deploy(ctx context.Context, m *store.Model, _ json.RawMessage) error {
	s.calls.Add(1)
	s.mu.Lock()
	s.deployed = append(s.deployed, m.ID)
	s.mu.Unlock()
	if s.deploy != nil {
		return s.deploy(ctx, m)
	}
	return nil
}

func (s *scriptedExec) Scale(ctx context.Context, m *store.Model, p json.RawMessage) error {
	return s.Deploy(ctx, m, p)
}

func (s *scriptedExec) Delete(_ context.Context, m *store.Model, _ json.RawMessage) error {
	s.mu.Lock()
	s.deleted = append(s.deleted, m.ID)
	s.mu.Unlock()
	return nil
}

func seed(t *testing.T, s *store.Store, ctx context.Context, model, job, kind string) {
	t.Helper()
	if _, err := s.UpsertModel(ctx, &store.Model{
		ID: model, HFRepo: "repo/" + model, Replicas: 1, GPUMemoryUtilization: 0.35,
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := s.EnqueueJob(ctx, job, model, kind, nil); err != nil {
		t.Fatal(err)
	}
}

func fastCfg(id string) Config {
	c := DefaultConfig(id)
	c.Lease = 600 * time.Millisecond
	c.RenewEvery = 150 * time.Millisecond
	c.PollEvery = 20 * time.Millisecond
	c.BaseBackoff = 10 * time.Millisecond
	c.MaxBackoff = 50 * time.Millisecond
	return c
}

// ------------------------------------------------------------------- happy path

func TestWorkerRunsAJobAndMarksTheModelReady(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	seed(t, s, ctx, "qwen", "j1", store.JobDeploy)

	ex := &scriptedExec{}
	w := NewWorker(s, ex, fastCfg("w1"), discardLog())
	if worked, err := w.RunOnce(ctx); err != nil || !worked {
		t.Fatalf("RunOnce: worked=%v err=%v", worked, err)
	}

	job, _ := s.Job(ctx, "j1")
	if job.Status != store.JobDone {
		t.Fatalf("job status = %s, want DONE (error: %q)", job.Status, job.Error)
	}
	m, _ := s.Model(ctx, "qwen")
	if m.Status != store.ModelReady {
		t.Fatalf("model status = %s, want READY", m.Status)
	}
}

func TestWorkerOnAnEmptyQueueDoesNothingQuietly(t *testing.T) {
	s := testStore(t)
	w := NewWorker(s, &scriptedExec{}, fastCfg("w1"), discardLog())
	worked, err := w.RunOnce(context.Background())
	if err != nil {
		t.Fatalf("an empty queue is the normal case, not an error: %v", err)
	}
	if worked {
		t.Fatal("reported work on an empty queue")
	}
}

func TestDeleteJobRemovesTheModel(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	seed(t, s, ctx, "qwen", "j1", store.JobDelete)

	ex := &scriptedExec{}
	w := NewWorker(s, ex, fastCfg("w1"), discardLog())
	if _, err := w.RunOnce(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Model(ctx, "qwen"); !errors.Is(err, store.ErrNotFound) {
		t.Fatalf("model survived a delete job: %v", err)
	}
	if len(ex.deleted) != 1 {
		t.Fatalf("executor Delete called %d times, want 1", len(ex.deleted))
	}
}

// ----------------------------------------------------------------- reliability

func TestASlowJobRenewsItsLeaseAndIsNotStolen(t *testing.T) {
	// A genuine 7B cold start outlives any sane lease. Renewal is what distinguishes
	// a slow worker from a dead one; without it, a second worker would claim the job
	// and two workers would drive the same GPU.
	s := testStore(t)
	ctx := context.Background()
	seed(t, s, ctx, "qwen", "j1", store.JobDeploy)

	release := make(chan struct{})
	ex := &scriptedExec{deploy: func(ctx context.Context, m *store.Model) error {
		<-release
		return nil
	}}

	w := NewWorker(s, ex, fastCfg("slow"), discardLog())
	done := make(chan error, 1)
	go func() { _, err := w.RunOnce(ctx); done <- err }()

	// Wait out several lease periods while the executor is blocked.
	time.Sleep(1500 * time.Millisecond)
	if _, err := s.ClaimJob(ctx, "thief", time.Minute); !errors.Is(err, store.ErrNotFound) {
		t.Fatal("a job held by a live worker was stolen despite renewal")
	}

	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	job, _ := s.Job(ctx, "j1")
	if job.Status != store.JobDone {
		t.Fatalf("job status = %s, want DONE", job.Status)
	}
}

func TestACrashedWorkersJobIsPickedUpByAnother(t *testing.T) {
	// The M2 acceptance criterion. The "crash" is modelled exactly as a crash
	// behaves: the worker stops without failing, completing or renewing anything.
	s := testStore(t)
	ctx := context.Background()
	seed(t, s, ctx, "qwen", "j1", store.JobDeploy)

	// Claim directly, then abandon it. Calling RunOnce and killing the goroutine
	// would still run its deferred bookkeeping, which a real crash does not.
	if _, err := s.ClaimJob(ctx, "doomed", 300*time.Millisecond); err != nil {
		t.Fatal(err)
	}
	time.Sleep(500 * time.Millisecond)

	ex := &scriptedExec{}
	rescue := NewWorker(s, ex, fastCfg("rescuer"), discardLog())
	worked, err := rescue.RunOnce(ctx)
	if err != nil || !worked {
		t.Fatalf("the abandoned job was not picked up: worked=%v err=%v", worked, err)
	}
	job, _ := s.Job(ctx, "j1")
	if job.Status != store.JobDone {
		t.Fatalf("job status = %s, want DONE", job.Status)
	}
	if ex.calls.Load() != 1 {
		t.Fatalf("executor ran %d times, want 1", ex.calls.Load())
	}
}

func TestAWorkerThatLosesItsLeaseDoesNotClobberTheNewOwner(t *testing.T) {
	// The dangerous case: a stalled worker finishes late, after another worker has
	// taken the job. It must discover it lost the lease and write nothing, rather
	// than marking the new owner's in-progress work as DONE.
	s := testStore(t)
	ctx := context.Background()
	seed(t, s, ctx, "qwen", "j1", store.JobDeploy)

	release := make(chan struct{})
	stalled := &scriptedExec{deploy: func(ctx context.Context, m *store.Model) error {
		<-release
		return nil
	}}

	cfg := fastCfg("stalled")
	cfg.Lease = 300 * time.Millisecond
	// Renewal interval longer than the lease: this worker cannot keep its claim,
	// which is precisely the stall being modelled.
	cfg.RenewEvery = 5 * time.Second
	w := NewWorker(s, stalled, cfg, discardLog())

	done := make(chan struct{})
	go func() { _, _ = w.RunOnce(ctx); close(done) }()

	time.Sleep(500 * time.Millisecond)
	newOwner, err := s.ClaimJob(ctx, "newowner", 10*time.Second)
	if err != nil {
		t.Fatalf("the lapsed job was not reclaimable: %v", err)
	}

	close(release)
	<-done

	job, _ := s.Job(ctx, "j1")
	if job.Status != store.JobRunning {
		t.Fatalf("the stalled worker overwrote the new owner's state: status = %s", job.Status)
	}
	if job.LockedBy == nil || *job.LockedBy != "newowner" {
		t.Fatalf("locked_by = %v, want newowner", job.LockedBy)
	}
	_ = newOwner
}

func TestAFailedJobIsRetriedAndTheModelIsMarkedFailed(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	seed(t, s, ctx, "qwen", "j1", store.JobDeploy)

	var attempts atomic.Int64
	ex := &scriptedExec{deploy: func(ctx context.Context, m *store.Model) error {
		if attempts.Add(1) == 1 {
			return errors.New("weights not found")
		}
		return nil
	}}

	w := NewWorker(s, ex, fastCfg("w1"), discardLog())
	if _, err := w.RunOnce(ctx); err != nil {
		t.Fatal(err)
	}

	job, _ := s.Job(ctx, "j1")
	if job.Status != store.JobPending {
		t.Fatalf("after one failure status = %s, want PENDING for retry", job.Status)
	}
	// An operator debugging a failed deploy has only this field.
	if job.Error == "" {
		t.Fatal("the failure recorded no error text")
	}
	m, _ := s.Model(ctx, "qwen")
	if m.Status != store.ModelFailed {
		t.Fatalf("model status = %s, want FAILED while the deploy is broken", m.Status)
	}

	// Wait out the backoff and let the retry succeed.
	time.Sleep(120 * time.Millisecond)
	if _, err := w.RunOnce(ctx); err != nil {
		t.Fatal(err)
	}
	job, _ = s.Job(ctx, "j1")
	if job.Status != store.JobDone {
		t.Fatalf("the retry did not complete: status = %s error = %q", job.Status, job.Error)
	}
	m, _ = s.Model(ctx, "qwen")
	if m.Status != store.ModelReady {
		t.Fatalf("model status = %s after a successful retry, want READY", m.Status)
	}
}

func TestConcurrentWorkersExecuteEachJobExactlyOnce(t *testing.T) {
	// The whole point of SKIP LOCKED plus leases: several control-plane replicas
	// draining one queue must not deploy the same model twice.
	s := testStore(t)
	ctx := context.Background()
	const jobs = 10
	for i := 0; i < jobs; i++ {
		id := string(rune('a'+i)) + "-model"
		seed(t, s, ctx, id, "j-"+id, store.JobDeploy)
	}

	ex := &scriptedExec{deploy: func(ctx context.Context, m *store.Model) error {
		time.Sleep(20 * time.Millisecond) // overlap the workers
		return nil
	}}

	runCtx, cancel := context.WithCancel(ctx)
	var wg sync.WaitGroup
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			NewWorker(s, ex, fastCfg("w"+string(rune('0'+i))), discardLog()).Run(runCtx)
		}(i)
	}

	deadline := time.Now().Add(20 * time.Second)
	for time.Now().Before(deadline) {
		all, err := s.Jobs(ctx, 100)
		if err != nil {
			t.Fatal(err)
		}
		done := 0
		for _, j := range all {
			if j.Status == store.JobDone {
				done++
			}
		}
		if done == jobs {
			break
		}
		time.Sleep(25 * time.Millisecond)
	}
	cancel()
	wg.Wait()

	all, _ := s.Jobs(ctx, 100)
	for _, j := range all {
		if j.Status != store.JobDone {
			t.Fatalf("job %s ended as %s (%s)", j.ID, j.Status, j.Error)
		}
	}
	if got := ex.calls.Load(); got != jobs {
		t.Fatalf("executor ran %d times for %d jobs; work was duplicated or dropped", got, jobs)
	}
}

// --------------------------------------------------------------------- backoff

func TestBackoffGrowsAndIsJittered(t *testing.T) {
	w := NewWorker(nil, nil, Config{
		WorkerID: "w", Lease: time.Minute, RenewEvery: time.Second, PollEvery: time.Second,
		BaseBackoff: time.Second, MaxBackoff: time.Minute,
	}, discardLog())

	var prevMax time.Duration
	for attempt := 1; attempt <= 5; attempt++ {
		seen := map[time.Duration]bool{}
		var max time.Duration
		for i := 0; i < 200; i++ {
			d := w.backoff(attempt)
			seen[d] = true
			if d > max {
				max = d
			}
			if d > time.Minute {
				t.Fatalf("attempt %d produced %s, above MaxBackoff", attempt, d)
			}
		}
		// Jitter matters: workers that fail together — a shared dependency going
		// down, the common case — would otherwise retry in perfect lockstep and
		// hammer it again at exactly the same instant.
		if len(seen) < 10 {
			t.Fatalf("attempt %d produced only %d distinct delays; jitter is absent", attempt, len(seen))
		}
		if attempt > 1 && max <= prevMax {
			t.Fatalf("backoff did not grow: attempt %d max %s vs previous %s", attempt, max, prevMax)
		}
		prevMax = max
	}
}

func TestReplicaIDsAreURLSafe(t *testing.T) {
	// Replica ids are interpolated into node-agent URL paths. '#' begins a fragment,
	// so POST /replicas/model#0/sleep arrives as POST /replicas/model and 405s. This
	// cost an afternoon in M1 and is pinned here so nobody reintroduces it.
	for _, bad := range []string{"#", "/", "?", " ", "%"} {
		if id := ReplicaID("qwen-0.5b", 0); contains(id, bad) {
			t.Fatalf("replica id %q contains the URL-unsafe character %q", id, bad)
		}
	}
	if got := ReplicaID("qwen-0.5b", 3); got != "qwen-0.5b--3" {
		t.Fatalf("ReplicaID = %q", got)
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

// withSchema points a DSN at a dedicated Postgres schema.
func withSchema(dsn, schema string) string {
	sep := "?"
	if strings.Contains(dsn, "?") {
		sep = "&"
	}
	return dsn + sep + "search_path=" + schema
}

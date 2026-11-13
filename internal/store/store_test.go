package store

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sairavig/kiln/internal/auth"
)

// These are integration tests against a real PostgreSQL.
//
// They are not faked, on purpose. The behaviour under test — FOR UPDATE SKIP LOCKED,
// lease expiry, partial unique indexes — is behaviour of Postgres, not of the Go
// code, and a fake would only assert that the fake does what I assumed Postgres does.
// That is the assumption most likely to be wrong, so it is the one worth testing for
// real.
//
// They skip when KILN_TEST_DATABASE_URL is unset so the default `go test ./...`
// remains offline and GPU-free.
func testStore(t *testing.T) *Store {
	t.Helper()
	dsn := os.Getenv("KILN_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("KILN_TEST_DATABASE_URL not set; skipping Postgres integration tests")
	}
	// Each integration suite gets its own schema. `go test ./...` runs packages in
	// parallel, so two suites sharing one set of tables would truncate each other's
	// rows mid-test and fail in ways that look like product bugs.
	dsn = withSchema(dsn, "store_test_schema")
	ctx := context.Background()
	s, err := Open(ctx, dsn)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if err := s.Migrate(ctx); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	// Truncate rather than drop: migrations are idempotent and re-running themevery  test
	// would dominate the runtime. CASCADE follows the foreign keys.
	_, err = s.Pool.Exec(ctx,
		`TRUNCATE deploy_jobs, replicas, models, usage, quotas, api_keys, teams CASCADE`)
	if err != nil {
		t.Fatalf("truncate: %v", err)
	}
	t.Cleanup(s.Close)
	return s
}

func ctxT(t *testing.T) context.Context {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	t.Cleanup(cancel)
	return ctx
}

// --------------------------------------------------------------- migrations

func TestMigrationsAreIdempotent(t *testing.T) {
	s := testStore(t)
	// A control plane restarting in a crash loop runs migrations on every start.
	for i := 0; i < 3; i++ {
		if err := s.Migrate(ctxT(t)); err != nil {
			t.Fatalf("migrate run %d: %v", i, err)
		}
	}
}

// ------------------------------------------------------------- tenants & keys

func TestTeamAndQuotaAreCreatedAtomically(t *testing.T) {
	s, ctx := testStore(t), ctxT(t)
	q := auth.Quota{RPM: 30, TokensPerDay: 100000, MaxConcurrency: 4}
	if _, err := s.CreateTeam(ctx, "acme", "Acme", q); err != nil {
		t.Fatal(err)
	}
	// A team without a quota row is unlimited, so the two must never be separable.
	qs, err := s.Quotas(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if got := qs["acme"]; got != q {
		t.Fatalf("quota = %+v, want %+v", got, q)
	}
}

func TestAPIKeyRoundTripAndRevocation(t *testing.T) {
	s, ctx := testStore(t), ctxT(t)
	if _, err := s.CreateTeam(ctx, "acme", "Acme", auth.Quota{RPM: 10}); err != nil {
		t.Fatal(err)
	}
	pt, err := s.CreateAPIKey(ctx, "key1", "acme")
	if err != nil {
		t.Fatal(err)
	}

	k, err := s.LookupKey(ctx, pt)
	if err != nil {
		t.Fatalf("lookup of a fresh key: %v", err)
	}
	if k.TeamID != "acme" {
		t.Fatalf("key resolved to team %q", k.TeamID)
	}

	if _, err := s.LookupKey(ctx, pt+"tampered"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("a tampered key resolved: %v", err)
	}

	if err := s.RevokeKey(ctx, "key1"); err != nil {
		t.Fatal(err)
	}
	if _, err := s.LookupKey(ctx, pt); !errors.Is(err, ErrNotFound) {
		t.Fatalf("a revoked key still resolves: %v", err)
	}
	// Revocation is idempotent; an operator reacting to a leak should not be made to
	// care whether they already clicked the button.
	if err := s.RevokeKey(ctx, "key1"); err != nil {
		t.Fatalf("second revoke: %v", err)
	}
	if err := s.RevokeKey(ctx, "nope"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("revoking an unknown key: %v", err)
	}
}

func TestPlaintextKeysAreNotStored(t *testing.T) {
	s, ctx := testStore(t), ctxT(t)
	_, _ = s.CreateTeam(ctx, "acme", "Acme", auth.Quota{RPM: 10})
	pt, err := s.CreateAPIKey(ctx, "key1", "acme")
	if err != nil {
		t.Fatal(err)
	}
	// The property a database dump must satisfy.
	var count int
	if err := s.Pool.QueryRow(ctx,
		`SELECT COUNT(*) FROM api_keys WHERE key_prefix = $1 AND encode(key_hash,'escape') = $1`,
		pt).Scan(&count); err != nil {
		t.Fatal(err)
	}
	if count != 0 {
		t.Fatal("the plaintext key is recoverable from the table")
	}
}

func TestUsageAggregation(t *testing.T) {
	s, ctx := testStore(t), ctxT(t)
	_, _ = s.CreateTeam(ctx, "acme", "Acme", auth.Quota{RPM: 10})
	_, _ = s.CreateTeam(ctx, "beta", "Beta", auth.Quota{RPM: 10})

	recs := []UsageRecord{
		{TeamID: "acme", ModelID: "m1", PromptTokens: 10, CompletionTokens: 5, StatusCode: 200},
		{TeamID: "acme", ModelID: "m1", PromptTokens: 20, CompletionTokens: 7, StatusCode: 200},
		{TeamID: "acme", ModelID: "m2", PromptTokens: 1, CompletionTokens: 1, StatusCode: 200},
		{TeamID: "beta", ModelID: "m1", PromptTokens: 100, CompletionTokens: 50, StatusCode: 200},
	}
	if err := s.RecordUsage(ctx, recs); err != nil {
		t.Fatal(err)
	}

	total, err := s.TokensSince(ctx, "acme", time.Now().Add(-time.Hour))
	if err != nil {
		t.Fatal(err)
	}
	if total != 44 {
		t.Fatalf("acme tokens = %d, want 44", total)
	}
	// Team isolation: one tenant's usage must never be billed to another.
	betaTotal, _ := s.TokensSince(ctx, "beta", time.Now().Add(-time.Hour))
	if betaTotal != 150 {
		t.Fatalf("beta tokens = %d, want 150", betaTotal)
	}

	sum, err := s.UsageBreakdown(ctx, time.Now().Add(-time.Hour))
	if err != nil {
		t.Fatal(err)
	}
	if len(sum) != 3 {
		t.Fatalf("breakdown rows = %d, want 3", len(sum))
	}
}

func TestTokensSinceIsZeroNotNullForAQuietTeam(t *testing.T) {
	// SUM over no rows returns NULL, which scans into a nil pointer and would panic
	// or seed the limiter with garbage if not handled.
	s, ctx := testStore(t), ctxT(t)
	_, _ = s.CreateTeam(ctx, "quiet", "Quiet", auth.Quota{RPM: 10})
	got, err := s.TokensSince(ctx, "quiet", time.Now().Add(-time.Hour))
	if err != nil {
		t.Fatal(err)
	}
	if got != 0 {
		t.Fatalf("got %d, want 0", got)
	}
}

// ------------------------------------------------------------------- models

func TestUpsertModelUpdatesShapeInPlace(t *testing.T) {
	s, ctx := testStore(t), ctxT(t)
	m := &Model{ID: "qwen", HFRepo: "Qwen/Qwen2.5-0.5B-Instruct", Replicas: 1,
		GPUMemoryUtilization: 0.35, MaxModelLen: 2048}
	if _, err := s.UpsertModel(ctx, m); err != nil {
		t.Fatal(err)
	}
	// Re-POSTing the same id is how a caller scales or repoints a deployment.
	m.Replicas = 3
	out, err := s.UpsertModel(ctx, m)
	if err != nil {
		t.Fatal(err)
	}
	if out.Replicas != 3 {
		t.Fatalf("replicas = %d, want 3", out.Replicas)
	}
	all, _ := s.Models(ctx)
	if len(all) != 1 {
		t.Fatalf("upsert created a duplicate row: %d models", len(all))
	}
}

func TestMeasurementsAreNotWipedByARedeploy(t *testing.T) {
	// A zero means "not measured this time", not "measured as zero". Wiping the wake
	// costs would make the cost-aware policy score every wake as free and stop
	// distinguishing models at all.
	s, ctx := testStore(t), ctxT(t)
	if _, err := s.UpsertModel(ctx, &Model{ID: "qwen", HFRepo: "r", Replicas: 1}); err != nil {
		t.Fatal(err)
	}
	if err := s.RecordMeasurements(ctx, "qwen", Measurements{
		FootprintVRAMBytes: 3 << 30, ColdStartMS: 40000, WakeHostMS: 880, WakeDiskMS: 1870,
	}); err != nil {
		t.Fatal(err)
	}
	// A later run that only cold-started: wake costs are left unmeasured.
	if err := s.RecordMeasurements(ctx, "qwen", Measurements{ColdStartMS: 35000}); err != nil {
		t.Fatal(err)
	}

	m, err := s.Model(ctx, "qwen")
	if err != nil {
		t.Fatal(err)
	}
	if m.ColdStartMS != 35000 {
		t.Fatalf("cold start = %d, want the fresh 35000", m.ColdStartMS)
	}
	if m.WakeHostMS != 880 || m.WakeDiskMS != 1870 {
		t.Fatalf("wake costs were wiped: host=%d disk=%d", m.WakeHostMS, m.WakeDiskMS)
	}
	if m.FootprintVRAMBytes != 3<<30 {
		t.Fatalf("footprint was wiped: %d", m.FootprintVRAMBytes)
	}
}

func TestDeletingAModelCascades(t *testing.T) {
	s, ctx := testStore(t), ctxT(t)
	_, _ = s.UpsertModel(ctx, &Model{ID: "qwen", HFRepo: "r", Replicas: 1})
	_ = s.UpsertReplica(ctx, &Replica{ID: "qwen--0", ModelID: "qwen", Node: "n1",
		AgentURL: "http://a", Port: 19000, State: "ACTIVE"})
	if _, err := s.EnqueueJob(ctx, "j1", "qwen", JobDeploy, nil); err != nil {
		t.Fatal(err)
	}

	if err := s.DeleteModel(ctx, "qwen"); err != nil {
		t.Fatal(err)
	}
	// Orphaned replica rows would make the scheduler budget against models that do
	// not exist.
	reps, _ := s.Replicas(ctx, "qwen")
	if len(reps) != 0 {
		t.Fatalf("replicas survived the delete: %d", len(reps))
	}
	jobs, _ := s.Jobs(ctx, 10)
	if len(jobs) != 0 {
		t.Fatalf("jobs survived the delete: %d", len(jobs))
	}
}

// --------------------------------------------------------------------- jobs

func seedModel(t *testing.T, s *Store, ctx context.Context, id string) {
	t.Helper()
	if _, err := s.UpsertModel(ctx, &Model{ID: id, HFRepo: "repo/" + id, Replicas: 1}); err != nil {
		t.Fatal(err)
	}
}

func TestOnlyOneLiveJobPerModel(t *testing.T) {
	// Two concurrent deploys of one model would race to start replicas and
	// double-allocate the GPU. The database refuses rather than trusting every
	// caller to check first.
	s, ctx := testStore(t), ctxT(t)
	seedModel(t, s, ctx, "qwen")

	if _, err := s.EnqueueJob(ctx, "j1", "qwen", JobDeploy, nil); err != nil {
		t.Fatal(err)
	}
	_, err := s.EnqueueJob(ctx, "j2", "qwen", JobDeploy, nil)
	if !errors.Is(err, ErrConflict) {
		t.Fatalf("second live job was accepted: %v", err)
	}

	// Once the first finishes, a new one is allowed.
	j, _ := s.ClaimJob(ctx, "w1", time.Minute)
	if err := s.CompleteJob(ctx, j.ID, "w1"); err != nil {
		t.Fatal(err)
	}
	if _, err := s.EnqueueJob(ctx, "j2", "qwen", JobDeploy, nil); err != nil {
		t.Fatalf("job after completion rejected: %v", err)
	}
}

func TestConcurrentWorkersEachClaimADistinctJob(t *testing.T) {
	// The SKIP LOCKED property: N workers claiming simultaneously must produce N
	// distinct jobs with no duplicates and no deadlock.
	s, ctx := testStore(t), ctxT(t)
	const n = 12
	for i := 0; i < n; i++ {
		id := fmt.Sprintf("m%02d", i)
		seedModel(t, s, ctx, id)
		if _, err := s.EnqueueJob(ctx, "j"+id, id, JobDeploy, nil); err != nil {
			t.Fatal(err)
		}
	}

	var mu sync.Mutex
	claimed := map[string]string{}
	var wg sync.WaitGroup
	for w := 0; w < n; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			worker := fmt.Sprintf("worker-%d", w)
			j, err := s.ClaimJob(ctx, worker, time.Minute)
			if err != nil {
				return
			}
			mu.Lock()
			defer mu.Unlock()
			if prev, dup := claimed[j.ID]; dup {
				t.Errorf("job %s claimed by both %s and %s", j.ID, prev, worker)
			}
			claimed[j.ID] = worker
		}(w)
	}
	wg.Wait()

	if len(claimed) != n {
		t.Fatalf("claimed %d of %d jobs; SKIP LOCKED should let every worker get one", len(claimed), n)
	}
}

func TestClaimedJobsAreInvisibleToOtherWorkers(t *testing.T) {
	s, ctx := testStore(t), ctxT(t)
	seedModel(t, s, ctx, "qwen")
	_, _ = s.EnqueueJob(ctx, "j1", "qwen", JobDeploy, nil)

	if _, err := s.ClaimJob(ctx, "w1", time.Minute); err != nil {
		t.Fatal(err)
	}
	if _, err := s.ClaimJob(ctx, "w2", time.Minute); !errors.Is(err, ErrNotFound) {
		t.Fatalf("a live claim was stolen: %v", err)
	}
}

func TestACrashedWorkersJobIsReclaimedAfterItsLeaseExpires(t *testing.T) {
	// The acceptance criterion for M2: a job survives a worker crash.
	//
	// A row lock cannot express this on its own, because it dies with the
	// transaction while the work takes minutes. The lease is what makes a claim
	// outlive its transaction and still be recoverable.
	s, ctx := testStore(t), ctxT(t)
	seedModel(t, s, ctx, "qwen")
	_, _ = s.EnqueueJob(ctx, "j1", "qwen", JobDeploy, nil)

	first, err := s.ClaimJob(ctx, "doomed-worker", 300*time.Millisecond)
	if err != nil {
		t.Fatal(err)
	}
	if first.Attempts != 1 {
		t.Fatalf("attempts = %d after first claim, want 1", first.Attempts)
	}

	// The worker dies here: no FailJob, no CompleteJob, no renewal. Nothing at all.
	if _, err := s.ClaimJob(ctx, "rescuer", time.Minute); !errors.Is(err, ErrNotFound) {
		t.Fatal("the job was reclaimed before its lease expired")
	}
	time.Sleep(500 * time.Millisecond)

	second, err := s.ClaimJob(ctx, "rescuer", time.Minute)
	if err != nil {
		t.Fatalf("the job was not reclaimed after the lease expired: %v", err)
	}
	if second.ID != first.ID {
		t.Fatalf("reclaimed the wrong job: %s", second.ID)
	}
	// Attempts increments at claim time, not at failure time: a worker killed mid-job
	// never reports a failure, and counting only reported failures would let a job
	// that reliably crashes its worker retry forever.
	if second.Attempts != 2 {
		t.Fatalf("attempts = %d after reclaim, want 2", second.Attempts)
	}
	if err := s.CompleteJob(ctx, second.ID, "rescuer"); err != nil {
		t.Fatal(err)
	}
}

func TestAZombieWorkerCannotOverwriteTheNewOwner(t *testing.T) {
	// A worker that stalls past its lease and then wakes up must discover it lost the
	// job, not silently mark the new owner's in-progress work as DONE.
	s, ctx := testStore(t), ctxT(t)
	seedModel(t, s, ctx, "qwen")
	_, _ = s.EnqueueJob(ctx, "j1", "qwen", JobDeploy, nil)

	if _, err := s.ClaimJob(ctx, "stalled", 200*time.Millisecond); err != nil {
		t.Fatal(err)
	}
	time.Sleep(350 * time.Millisecond)
	if _, err := s.ClaimJob(ctx, "newowner", time.Minute); err != nil {
		t.Fatal(err)
	}

	for name, err := range map[string]error{
		"complete": s.CompleteJob(ctx, "j1", "stalled"),
		"fail":     s.FailJob(ctx, "j1", "stalled", "boom", 0),
		"renew":    s.RenewLease(ctx, "j1", "stalled", time.Minute),
	} {
		if !errors.Is(err, ErrLeaseLost) {
			t.Errorf("%s by the stalled worker returned %v, want ErrLeaseLost", name, err)
		}
	}

	j, _ := s.Job(ctx, "j1")
	if j.Status != JobRunning || j.LockedBy == nil || *j.LockedBy != "newowner" {
		t.Fatalf("the new owner's claim was disturbed: %+v", j)
	}
}

func TestRenewingALeaseKeepsASlowJobAlive(t *testing.T) {
	s, ctx := testStore(t), ctxT(t)
	seedModel(t, s, ctx, "qwen")
	_, _ = s.EnqueueJob(ctx, "j1", "qwen", JobDeploy, nil)

	if _, err := s.ClaimJob(ctx, "slow", 400*time.Millisecond); err != nil {
		t.Fatal(err)
	}
	// A genuine deploy outlives its initial lease; renewing is what distinguishes
	// slow from dead.
	for i := 0; i < 3; i++ {
		time.Sleep(200 * time.Millisecond)
		if err := s.RenewLease(ctx, "j1", "slow", 400*time.Millisecond); err != nil {
			t.Fatalf("renew %d: %v", i, err)
		}
		if _, err := s.ClaimJob(ctx, "thief", time.Minute); !errors.Is(err, ErrNotFound) {
			t.Fatal("a renewed job was stolen")
		}
	}
}

func TestFailureRetriesWithBackoffThenGivesUp(t *testing.T) {
	s, ctx := testStore(t), ctxT(t)
	seedModel(t, s, ctx, "qwen")
	_, _ = s.EnqueueJob(ctx, "j1", "qwen", JobDeploy, nil)

	// Backoff is applied through run_after, not by sleeping in the worker: a job
	// whose weights 404 would otherwise spin a worker at full speed.
	j, _ := s.ClaimJob(ctx, "w1", time.Minute)
	if err := s.FailJob(ctx, j.ID, "w1", "weights 404", 2*time.Second); err != nil {
		t.Fatal(err)
	}
	if _, err := s.ClaimJob(ctx, "w1", time.Minute); !errors.Is(err, ErrNotFound) {
		t.Fatal("a job in backoff was claimed immediately")
	}

	// Exhaust the attempts with no backoff. The first failure above deliberately set
	// a 2s backoff, so this uses a separate job rather than waiting it out.
	seedModel(t, s, ctx, "other")
	if _, err := s.EnqueueJob(ctx, "j2", "other", JobDeploy, nil); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 10; i++ {
		j, err := s.ClaimJob(ctx, "w1", time.Minute)
		if errors.Is(err, ErrNotFound) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		if err := s.FailJob(ctx, j.ID, "w1", "still broken", 0); err != nil {
			t.Fatal(err)
		}
	}

	final, err := s.Job(ctx, "j2")
	if err != nil {
		t.Fatal(err)
	}
	if final.Status != JobFailed {
		t.Fatalf("status = %s after exhausting attempts, want FAILED", final.Status)
	}
	if final.Attempts < final.MaxAttempts {
		t.Fatalf("gave up at %d of %d attempts", final.Attempts, final.MaxAttempts)
	}
	if final.Error == "" {
		t.Fatal("a failed job recorded no error; the operator has nothing to debug with")
	}
	// A terminally failed job must not be claimable, or a worker loops on it forever.
	// (j1 is still in its 2s backoff, so an empty claim here covers both.)
	if _, err := s.ClaimJob(ctx, "w2", time.Minute); !errors.Is(err, ErrNotFound) {
		t.Fatalf("a FAILED job was claimed again: %v", err)
	}
}

func TestJobsAreClaimedOldestFirst(t *testing.T) {
	s, ctx := testStore(t), ctxT(t)
	for i := 0; i < 3; i++ {
		id := fmt.Sprintf("m%d", i)
		seedModel(t, s, ctx, id)
		if _, err := s.EnqueueJob(ctx, "j"+id, id, JobDeploy, nil); err != nil {
			t.Fatal(err)
		}
		// Postgres's now() is transaction-time, so distinct created_at values need a
		// real gap.
		time.Sleep(10 * time.Millisecond)
	}
	for i := 0; i < 3; i++ {
		j, err := s.ClaimJob(ctx, "w1", time.Minute)
		if err != nil {
			t.Fatal(err)
		}
		if want := fmt.Sprintf("jm%d", i); j.ID != want {
			t.Fatalf("claim %d returned %s, want %s", i, j.ID, want)
		}
		_ = s.CompleteJob(ctx, j.ID, "w1")
	}
}

func TestJobPayloadRoundTrips(t *testing.T) {
	s, ctx := testStore(t), ctxT(t)
	seedModel(t, s, ctx, "qwen")
	type payload struct {
		Replicas int    `json:"replicas"`
		Reason   string `json:"reason"`
	}
	if _, err := s.EnqueueJob(ctx, "j1", "qwen", JobScale, payload{Replicas: 4, Reason: "load"}); err != nil {
		t.Fatal(err)
	}
	j, err := s.Job(ctx, "j1")
	if err != nil {
		t.Fatal(err)
	}
	if string(j.Payload) == "" || string(j.Payload) == "{}" {
		t.Fatalf("payload lost: %q", j.Payload)
	}
}

func TestClaimOnAnEmptyQueueIsNotAnError(t *testing.T) {
	// Workers poll constantly; an empty queue is the normal case, not a failure.
	s := testStore(t)
	if _, err := s.ClaimJob(ctxT(t), "w1", time.Minute); !errors.Is(err, ErrNotFound) {
		t.Fatalf("empty-queue claim returned %v, want ErrNotFound", err)
	}
}

// withSchema points a DSN at a dedicated Postgres schema.
func withSchema(dsn, schema string) string {
	sep := "?"
	if strings.Contains(dsn, "?") {
		sep = "&"
	}
	return dsn + sep + "search_path=" + schema
}

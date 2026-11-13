package auth

import (
	"strings"
	"sync"
	"testing"
	"time"
)

var base = time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)

// clock is a manually advanced clock so quota behaviour over time is tested without
// sleeping. A quota suite that sleeps is a quota suite nobody runs.
type clock struct {
	mu sync.Mutex
	t  time.Time
}

func newClock() *clock { return &clock{t: base} }

func (c *clock) now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.t
}

func (c *clock) advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.t = c.t.Add(d)
}

// ---------------------------------------------------------------------- keys

func TestGeneratedKeysAreUniqueAndVerifiable(t *testing.T) {
	seen := map[string]bool{}
	for i := 0; i < 500; i++ {
		pt, prefix, hash, err := GenerateKey()
		if err != nil {
			t.Fatal(err)
		}
		if seen[pt] {
			t.Fatal("generated a duplicate key")
		}
		seen[pt] = true

		if !strings.HasPrefix(pt, "sk-") {
			t.Fatalf("key %q lacks the sk- prefix secret scanners look for", pt)
		}
		if prefix != pt[:KeyPrefixLen] {
			t.Fatalf("prefix %q does not match the key", prefix)
		}
		k := &Key{Hash: hash}
		if !k.Matches(pt) {
			t.Fatal("a freshly generated key failed to verify")
		}
		if k.Matches(pt + "x") {
			t.Fatal("a modified key verified")
		}
	}
}

func TestPlaintextIsNotRecoverableFromStorage(t *testing.T) {
	// The property the whole scheme rests on: a database dump must not contain
	// working credentials.
	pt, prefix, hash, _ := GenerateKey()
	if strings.Contains(string(hash), pt) {
		t.Fatal("the stored hash contains the plaintext")
	}
	if len(prefix) >= len(pt) {
		t.Fatal("the stored prefix is the whole key")
	}
	// The prefix must be short enough to be useless on its own.
	if len(prefix) > 16 {
		t.Fatalf("prefix is %d chars, long enough to narrow a brute force", len(prefix))
	}
}

func TestRevokedKeysNeverMatch(t *testing.T) {
	pt, _, hash, _ := GenerateKey()
	k := &Key{Hash: hash, Revoked: true}
	if k.Matches(pt) {
		t.Fatal("a revoked key still authenticates")
	}
}

func TestBearerTokenParsing(t *testing.T) {
	for _, tc := range []struct {
		header string
		want   string
		err    error
	}{
		{"Bearer sk-kiln-abc", "sk-kiln-abc", nil},
		{"bearer sk-kiln-abc", "sk-kiln-abc", nil}, // case-insensitive per RFC 7235
		{"BEARER sk-kiln-abc", "sk-kiln-abc", nil},
		{"Bearer  sk-kiln-abc ", "sk-kiln-abc", nil},
		{"", "", ErrNoKey},
		{"sk-kiln-abc", "", ErrUnauthorized}, // no scheme
		{"Basic abc", "", ErrUnauthorized},
	} {
		got, err := BearerToken(tc.header)
		if err != tc.err {
			t.Errorf("BearerToken(%q) err = %v, want %v", tc.header, err, tc.err)
		}
		if got != tc.want {
			t.Errorf("BearerToken(%q) = %q, want %q", tc.header, got, tc.want)
		}
	}
}

// -------------------------------------------------------------------- quotas

func TestRPMIsEnforcedAndRefillsContinuously(t *testing.T) {
	c := newClock()
	l := NewLimiter(c.now)
	l.SetQuota("t1", Quota{RPM: 60, MaxConcurrency: 1000})

	// A full bucket allows a burst of exactly RPM.
	for i := 0; i < 60; i++ {
		if d := l.Allow("t1"); !d.Allowed {
			t.Fatalf("request %d rejected: %s", i, d.Reason)
		}
		l.Release("t1")
	}
	d := l.Allow("t1")
	if d.Allowed {
		t.Fatal("request 61 allowed within the same instant")
	}
	if d.Reason != "rpm" {
		t.Fatalf("reason = %q, want rpm", d.Reason)
	}
	if d.RetryAfter <= 0 || d.RetryAfter > time.Minute {
		t.Fatalf("RetryAfter = %s, want a sane sub-minute value", d.RetryAfter)
	}

	// Continuous refill: one second at 60rpm restores exactly one token, so a client
	// pacing itself evenly is never throttled.
	c.advance(time.Second)
	if d := l.Allow("t1"); !d.Allowed {
		t.Fatalf("after 1s of refill the request was still rejected: %s", d.Reason)
	}
	l.Release("t1")
	if d := l.Allow("t1"); d.Allowed {
		t.Fatal("refill produced more than one token in one second")
	}
}

func TestTokenBucketPreventsTheFixedWindowBoundaryBurst(t *testing.T) {
	// A fixed window lets a team spend a full minute's budget at the end of one
	// window and again at the start of the next: 2x the nominal limit in an instant,
	// which is exactly when a GPU can least absorb it. A token bucket cannot.
	c := newClock()
	l := NewLimiter(c.now)
	l.SetQuota("t1", Quota{RPM: 60, MaxConcurrency: 1000})

	allowed := 0
	for i := 0; i < 60; i++ {
		if l.Allow("t1").Allowed {
			allowed++
			l.Release("t1")
		}
	}
	// Step to what would be the next fixed window and immediately try another burst.
	c.advance(time.Second)
	for i := 0; i < 60; i++ {
		if l.Allow("t1").Allowed {
			allowed++
			l.Release("t1")
		}
	}
	if allowed > 62 {
		t.Fatalf("allowed %d requests across a window boundary; a token bucket should cap near 61", allowed)
	}
}

func TestMaxConcurrencyIsEnforcedAndReleased(t *testing.T) {
	l := NewLimiter(newClock().now)
	l.SetQuota("t1", Quota{RPM: 10000, MaxConcurrency: 3})

	for i := 0; i < 3; i++ {
		if !l.Allow("t1").Allowed {
			t.Fatalf("concurrent request %d rejected", i)
		}
	}
	d := l.Allow("t1")
	if d.Allowed || d.Reason != "max_concurrency" {
		t.Fatalf("4th concurrent request: allowed=%v reason=%q", d.Allowed, d.Reason)
	}

	l.Release("t1")
	if !l.Allow("t1").Allowed {
		t.Fatal("a freed slot was not reusable")
	}
}

func TestConcurrencyCheckAndReserveAreAtomic(t *testing.T) {
	// With a separate check and reserve, N concurrent requests all observe "below the
	// limit" and all proceed — exactly the overload the limit exists to prevent.
	l := NewLimiter(newClock().now)
	const cap = 5
	l.SetQuota("t1", Quota{RPM: 1000000, MaxConcurrency: cap})

	var wg sync.WaitGroup
	var mu sync.Mutex
	admitted := 0
	for i := 0; i < 200; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if l.Allow("t1").Allowed {
				mu.Lock()
				admitted++
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	if admitted != cap {
		t.Fatalf("admitted %d concurrent requests, cap is %d", admitted, cap)
	}
}

func TestDailyTokenBudgetBlocksAndResetsAfterADay(t *testing.T) {
	c := newClock()
	l := NewLimiter(c.now)
	l.SetQuota("t1", Quota{RPM: 100000, TokensPerDay: 1000, MaxConcurrency: 100})

	if !l.Allow("t1").Allowed {
		t.Fatal("first request rejected")
	}
	l.Release("t1")
	l.RecordTokens("t1", 1000)

	d := l.Allow("t1")
	if d.Allowed || d.Reason != "tokens_per_day" {
		t.Fatalf("over budget: allowed=%v reason=%q", d.Allowed, d.Reason)
	}
	// Retry-After should point at the window reset, not at some fixed guess.
	if d.RetryAfter < 23*time.Hour || d.RetryAfter > 24*time.Hour {
		t.Fatalf("RetryAfter = %s, want ~24h", d.RetryAfter)
	}

	c.advance(24 * time.Hour)
	if !l.Allow("t1").Allowed {
		t.Fatal("the daily window did not reset")
	}
}

func TestTokensAreCountedAfterTheFactByDesign(t *testing.T) {
	// A request is admitted on the budget *before* it runs, so a team at 99% can push
	// slightly over. Pre-reserving a worst-case token count instead would reject
	// requests that would have fit, which is the worse error — this pins the chosen
	// trade-off rather than leaving it to drift.
	c := newClock()
	l := NewLimiter(c.now)
	l.SetQuota("t1", Quota{RPM: 100000, TokensPerDay: 1000, MaxConcurrency: 100})

	l.RecordTokens("t1", 999)
	if !l.Allow("t1").Allowed {
		t.Fatal("a request at 99.9% of budget was rejected before running")
	}
	l.Release("t1")
	l.RecordTokens("t1", 5000) // a long generation

	if l.DailyTokens("t1") != 5999 {
		t.Fatalf("daily tokens = %d, want 5999", l.DailyTokens("t1"))
	}
	if l.Allow("t1").Allowed {
		t.Fatal("the next request was admitted despite being far over budget")
	}
}

func TestSeedingSurvivesAGatewayRestart(t *testing.T) {
	// Without seeding, restarting the gateway hands every team a fresh daily budget,
	// which makes the quota trivially bypassable by waiting for a deploy.
	c := newClock()
	l := NewLimiter(c.now)
	l.SetQuota("t1", Quota{RPM: 100000, TokensPerDay: 1000, MaxConcurrency: 100})
	l.SeedDailyTokens("t1", 1000, c.now().Add(-time.Hour))

	if d := l.Allow("t1"); d.Allowed {
		t.Fatal("a restarted gateway granted a fresh daily budget")
	}
}

func TestSeparateTeamsDoNotShareLimits(t *testing.T) {
	l := NewLimiter(newClock().now)
	l.SetQuota("small", Quota{RPM: 2, MaxConcurrency: 10})
	l.SetQuota("large", Quota{RPM: 1000, MaxConcurrency: 10})

	for i := 0; i < 2; i++ {
		if !l.Allow("small").Allowed {
			t.Fatalf("small team request %d rejected", i)
		}
		l.Release("small")
	}
	if l.Allow("small").Allowed {
		t.Fatal("small team exceeded its own RPM")
	}
	// The throttled team must not affect the other one.
	for i := 0; i < 50; i++ {
		if !l.Allow("large").Allowed {
			t.Fatalf("large team request %d rejected because of the small team", i)
		}
		l.Release("large")
	}
}

func TestNoQuotaMeansUnlimited(t *testing.T) {
	l := NewLimiter(newClock().now)
	for i := 0; i < 1000; i++ {
		if !l.Allow("internal").Allowed {
			t.Fatal("an unconfigured team was throttled")
		}
	}
}

func TestRetryAfterIsNeverZero(t *testing.T) {
	// "Retry-After: 0" tells a client to retry immediately, turning a throttle into a
	// hot loop against an already-overloaded gateway.
	for _, d := range []time.Duration{0, time.Nanosecond, 400 * time.Millisecond, 90 * time.Second} {
		if got := RetryAfterSeconds(d); got == "0" {
			t.Fatalf("RetryAfterSeconds(%s) = %q", d, got)
		}
	}
	if got := RetryAfterSeconds(90 * time.Second); got != "90" {
		t.Fatalf("RetryAfterSeconds(90s) = %q, want 90", got)
	}
}

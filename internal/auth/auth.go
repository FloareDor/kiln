// Package auth implements API keys, per-team quotas and usage accounting.
//
// The quota logic is pure and in-memory by design: a token-bucket check that needed a
// database round trip would add milliseconds to every request and would fail open or
// closed during a database blip, neither of which is acceptable on the data path.
// Durable counters live in Postgres and are loaded in; the enforcement decision is
// made locally.
package auth

import (
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"
)

// KeyPrefixLen is how many characters of a key are stored in plaintext so a human can
// identify a key without seeing it, and so verification is a single indexed lookup
// rather than a scan over every key in the table.
const KeyPrefixLen = 12

// Errors returned by the authenticator.
var (
	ErrUnauthorized = errors.New("auth: invalid or revoked API key")
	ErrNoKey        = errors.New("auth: missing Authorization header")
)

// GenerateKey returns a new plaintext key and its storage form. The plaintext is
// shown to the user exactly once and never persisted.
func GenerateKey() (plaintext, prefix string, hash []byte, err error) {
	buf := make([]byte, 24)
	if _, err = rand.Read(buf); err != nil {
		return "", "", nil, err
	}
	// The sk- prefix is what every OpenAI SDK and most secret scanners expect, which
	// makes a leaked key more likely to be caught by tooling the user already runs.
	plaintext = "sk-kiln-" + hex.EncodeToString(buf)
	prefix = plaintext[:KeyPrefixLen]
	sum := HashKey(plaintext)
	return plaintext, prefix, sum, nil
}

// HashKey hashes a plaintext key for storage and comparison.
//
// SHA-256, not bcrypt or argon2. The usual reason to use a slow hash is that the
// input is a low-entropy human password; an API key here is 192 bits of CSPRNG
// output, so there is no dictionary to attack and a slow hash buys nothing but
// latency — and this runs on every single request.
func HashKey(plaintext string) []byte {
	sum := sha256.Sum256([]byte(plaintext))
	return sum[:]
}

// PrefixOf extracts the stored prefix from a plaintext key.
func PrefixOf(plaintext string) string {
	if len(plaintext) < KeyPrefixLen {
		return plaintext
	}
	return plaintext[:KeyPrefixLen]
}

// BearerToken pulls the key out of an Authorization header.
func BearerToken(header string) (string, error) {
	if header == "" {
		return "", ErrNoKey
	}
	const p = "bearer "
	if len(header) <= len(p) || !strings.EqualFold(header[:len(p)], p) {
		return "", ErrUnauthorized
	}
	return strings.TrimSpace(header[len(p):]), nil
}

// Key is a stored credential.
type Key struct {
	ID      string
	TeamID  string
	Prefix  string
	Hash    []byte
	Revoked bool
}

// Matches compares a plaintext key against this record in constant time.
//
// Constant-time comparison even though the stored value is a hash: a short-circuiting
// compare leaks how many leading bytes matched, and with an online oracle that is
// enough to reconstruct the hash byte by byte. It costs nothing to avoid.
func (k *Key) Matches(plaintext string) bool {
	if k.Revoked {
		return false
	}
	return subtle.ConstantTimeCompare(k.Hash, HashKey(plaintext)) == 1
}

// Quota is a team's limits.
type Quota struct {
	// RPM is requests per minute.
	RPM int
	// TokensPerDay caps total prompt plus completion tokens in a rolling day.
	TokensPerDay int64
	// MaxConcurrency caps simultaneous in-flight requests.
	MaxConcurrency int
}

// Decision is the outcome of a quota check.
type Decision struct {
	Allowed bool
	// Reason names the limit that was hit, used as a metric label so a 429 spike can
	// be attributed to the right limit rather than guessed at.
	Reason string
	// RetryAfter is how long the client should wait, sent in the header. Computing
	// this properly matters: a fixed value makes every throttled client retry in
	// lockstep and produces a thundering herd on the exact second it expires.
	RetryAfter time.Duration
}

// Limiter enforces per-team quotas.
//
// Requests per minute use a token bucket rather than a fixed window. A fixed window
// lets a team spend its entire minute's budget in the last instant of one window and
// again in the first instant of the next, producing a burst of twice the nominal
// limit at the boundary — which is exactly when a GPU is least able to absorb it.
//
// The daily token budget uses a rolling counter seeded from the database, since a
// day-long bucket's refill granularity is too coarse to be worth the complexity.
type Limiter struct {
	mu      sync.Mutex
	buckets map[string]*bucket
	day     map[string]*dayCounter
	conc    map[string]int
	quotas  map[string]Quota
	now     func() time.Time
}

type bucket struct {
	tokens float64
	last   time.Time
}

type dayCounter struct {
	tokens     int64
	windowOpen time.Time
}

// NewLimiter builds a limiter. now is injectable so quota behaviour over time is
// testable without sleeping.
func NewLimiter(now func() time.Time) *Limiter {
	if now == nil {
		now = time.Now
	}
	return &Limiter{
		buckets: map[string]*bucket{},
		day:     map[string]*dayCounter{},
		conc:    map[string]int{},
		quotas:  map[string]Quota{},
		now:     now,
	}
}

// SetQuota installs or replaces a team's limits.
func (l *Limiter) SetQuota(team string, q Quota) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.quotas[team] = q
}

// Quota returns a team's limits.
func (l *Limiter) Quota(team string) (Quota, bool) {
	l.mu.Lock()
	defer l.mu.Unlock()
	q, ok := l.quotas[team]
	return q, ok
}

// SeedDailyTokens loads a team's tokens already spent today, so a gateway restart
// does not hand everybody a fresh daily budget.
func (l *Limiter) SeedDailyTokens(team string, spent int64, windowOpen time.Time) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.day[team] = &dayCounter{tokens: spent, windowOpen: windowOpen}
}

// Allow checks every limit and, if all pass, reserves one request slot. The caller
// must call Release when the request finishes.
//
// Checking and reserving together is not an optimisation; it is a correctness
// requirement. With a separate check and reserve, N concurrent requests all observe
// the same "below the limit" state and all proceed, which is precisely the overload
// the concurrency limit exists to prevent.
func (l *Limiter) Allow(team string) Decision {
	l.mu.Lock()
	defer l.mu.Unlock()

	q, ok := l.quotas[team]
	if !ok {
		// No quota configured means unlimited. Teams are created with a quota row, so
		// this only happens for an internal caller.
		l.conc[team]++
		return Decision{Allowed: true}
	}
	now := l.now()

	if q.MaxConcurrency > 0 && l.conc[team] >= q.MaxConcurrency {
		return Decision{Reason: "max_concurrency",
			// Concurrency frees up when a request finishes, which is soon and
			// unpredictable; a short retry is the honest answer.
			RetryAfter: time.Second}
	}

	if q.TokensPerDay > 0 {
		d := l.day[team]
		if d == nil || now.Sub(d.windowOpen) >= 24*time.Hour {
			d = &dayCounter{windowOpen: now}
			l.day[team] = d
		}
		if d.tokens >= q.TokensPerDay {
			return Decision{Reason: "tokens_per_day",
				RetryAfter: d.windowOpen.Add(24 * time.Hour).Sub(now)}
		}
	}

	if q.RPM > 0 {
		b := l.buckets[team]
		if b == nil {
			b = &bucket{tokens: float64(q.RPM), last: now}
			l.buckets[team] = b
		}
		// Refill continuously rather than in steps, so a client that paces itself
		// evenly is never throttled by an artefact of the window.
		perSec := float64(q.RPM) / 60.0
		b.tokens += now.Sub(b.last).Seconds() * perSec
		if b.tokens > float64(q.RPM) {
			b.tokens = float64(q.RPM)
		}
		b.last = now
		if b.tokens < 1 {
			need := (1 - b.tokens) / perSec
			return Decision{Reason: "rpm", RetryAfter: time.Duration(need * float64(time.Second))}
		}
		b.tokens--
	}

	l.conc[team]++
	return Decision{Allowed: true}
}

// Release frees a concurrency slot.
func (l *Limiter) Release(team string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.conc[team] > 0 {
		l.conc[team]--
	}
}

// RecordTokens adds to a team's daily spend. Called after a request completes, since
// token counts are not known in advance.
//
// This is why the daily budget can be exceeded slightly: a team at 99% of its budget
// can start a request that pushes it over. Pre-reserving a worst-case token count
// instead would reject requests that would have fit, which is the worse error.
func (l *Limiter) RecordTokens(team string, tokens int64) {
	if tokens <= 0 {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	d := l.day[team]
	now := l.now()
	if d == nil || now.Sub(d.windowOpen) >= 24*time.Hour {
		d = &dayCounter{windowOpen: now}
		l.day[team] = d
	}
	d.tokens += tokens
}

// DailyTokens reports a team's spend in the current window.
func (l *Limiter) DailyTokens(team string) int64 {
	l.mu.Lock()
	defer l.mu.Unlock()
	if d := l.day[team]; d != nil {
		return d.tokens
	}
	return 0
}

// Concurrency reports a team's in-flight requests.
func (l *Limiter) Concurrency(team string) int {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.conc[team]
}

// RetryAfterSeconds renders a duration for the Retry-After header, which is
// integer seconds and must be at least 1 — a "0" tells the client to retry
// immediately and turns a throttle into a hot loop.
func RetryAfterSeconds(d time.Duration) string {
	s := int(d.Seconds())
	if s < 1 {
		s = 1
	}
	return fmt.Sprintf("%d", s)
}

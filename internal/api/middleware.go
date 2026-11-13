package api

import (
	"context"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"github.com/FloareDor/kiln/internal/auth"
	"github.com/FloareDor/kiln/internal/store"
)

// Authenticator resolves API keys and enforces quotas on the request path.
//
// Key lookups are cached, which is not premature optimisation: without a cache every
// single request becomes a database round trip, so the gateway's p50 latency inherits
// Postgres's, and a database blip takes the whole data path down rather than just the
// control path.
//
// The cache is why revocation needs thinking about. A revoked key stays usable for at
// most TTL seconds, which is the price of not consulting the database per request. The
// TTL is short, and Invalidate exists for the urgent case.
type Authenticator struct {
	Store   *store.Store
	Limiter *auth.Limiter
	Log     *slog.Logger
	TTL     time.Duration

	mu    sync.RWMutex
	cache map[string]cachedKey
}

type cachedKey struct {
	team  string
	keyID string
	at    time.Time
}

// NewAuthenticator builds the middleware.
func NewAuthenticator(s *store.Store, l *auth.Limiter, log *slog.Logger) *Authenticator {
	return &Authenticator{Store: s, Limiter: l, Log: log, TTL: 30 * time.Second,
		cache: map[string]cachedKey{}}
}

// LoadQuotas refreshes every team's limits into the limiter and seeds the day's token
// spend, so a gateway restart does not hand every tenant a fresh daily budget.
func (a *Authenticator) LoadQuotas(ctx context.Context) error {
	quotas, err := a.Store.Quotas(ctx)
	if err != nil {
		return err
	}
	dayStart := time.Now().Add(-24 * time.Hour)
	for team, q := range quotas {
		a.Limiter.SetQuota(team, q)
		spent, err := a.Store.TokensSince(ctx, team, dayStart)
		if err != nil {
			a.Log.Warn("could not seed daily tokens", "team", team, "err", err)
			continue
		}
		a.Limiter.SeedDailyTokens(team, spent, dayStart)
	}
	return nil
}

// RunQuotaRefresh keeps quotas current so an admin change takes effect without a
// restart.
func (a *Authenticator) RunQuotaRefresh(ctx context.Context, every time.Duration) {
	t := time.NewTicker(every)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			if err := a.LoadQuotas(ctx); err != nil {
				a.Log.Warn("quota refresh failed", "err", err)
			}
		}
	}
}

// Invalidate drops a key from the cache so a revocation takes effect immediately.
func (a *Authenticator) Invalidate(plaintext string) {
	a.mu.Lock()
	delete(a.cache, plaintext)
	a.mu.Unlock()
}

func (a *Authenticator) lookup(ctx context.Context, plaintext string) (team, keyID string, err error) {
	a.mu.RLock()
	c, ok := a.cache[plaintext]
	a.mu.RUnlock()
	if ok && time.Since(c.at) < a.TTL {
		return c.team, c.keyID, nil
	}

	k, err := a.Store.LookupKey(ctx, plaintext)
	if err != nil {
		return "", "", err
	}
	a.mu.Lock()
	a.cache[plaintext] = cachedKey{team: k.TeamID, keyID: k.ID, at: time.Now()}
	a.mu.Unlock()
	return k.TeamID, k.ID, nil
}

// Middleware authenticates and applies quotas.
func (a *Authenticator) Middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Health and metrics must stay reachable without a key, or a failing
		// database makes the gateway look dead to Kubernetes and it gets restarted
		// in a loop for a problem a restart cannot fix.
		if r.URL.Path == "/health" || r.URL.Path == "/metrics" {
			next.ServeHTTP(w, r)
			return
		}

		token, err := auth.BearerToken(r.Header.Get("Authorization"))
		if err != nil {
			WriteError(w, http.StatusUnauthorized, "invalid_request_error", "missing_api_key",
				"provide an API key as: Authorization: Bearer <key>")
			return
		}
		team, keyID, err := a.lookup(r.Context(), token)
		if err != nil {
			// Deliberately the same message for an unknown key and a revoked one:
			// distinguishing them tells an attacker which guesses were once valid.
			WriteError(w, http.StatusUnauthorized, "invalid_request_error", "invalid_api_key",
				"invalid API key")
			return
		}

		d := a.Limiter.Allow(team)
		if !d.Allowed {
			w.Header().Set("Retry-After", auth.RetryAfterSeconds(d.RetryAfter))
			w.Header().Set("X-Kiln-Quota-Exceeded", d.Reason)
			WriteError(w, http.StatusTooManyRequests, "rate_limit_error", d.Reason,
				"quota exceeded: "+d.Reason)
			return
		}
		defer a.Limiter.Release(team)

		// Last-used is an operator convenience; making every request wait on a write
		// would be a poor trade, so it is fire-and-forget on a detached context.
		go func() {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			_ = a.Store.TouchKey(ctx, keyID)
		}()

		next.ServeHTTP(w, r.WithContext(WithTeam(r.Context(), team)))
	})
}

// AdminAuth guards the admin API with a shared token.
//
// A single static token rather than per-user accounts: the admin API is meant to be
// reachable only from inside the cluster, and building an identity system for it
// would be scope that earns nothing. The token is compared in constant time anyway,
// and an empty configured token disables the admin API outright rather than leaving
// it open, which is the safer failure.
func AdminAuth(token string, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if token == "" {
			adminErr(w, http.StatusForbidden,
				"admin API is disabled; set KILN_ADMIN_TOKEN to enable it")
			return
		}
		got, err := auth.BearerToken(r.Header.Get("Authorization"))
		if err != nil || subtleCompare(got, token) != 1 {
			adminErr(w, http.StatusUnauthorized, "invalid admin token")
			return
		}
		next.ServeHTTP(w, r)
	})
}

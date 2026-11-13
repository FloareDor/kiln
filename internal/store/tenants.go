package store

import (
	"context"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/sairavig/kiln/internal/auth"
)

// Team is a tenant.
type Team struct {
	ID        string
	Name      string
	CreatedAt time.Time
}

// CreateTeam inserts a team and its quota in one transaction.
//
// Together, not separately: a team with no quota row is treated as unlimited by the
// limiter, so a crash between the two inserts would silently create an unthrottled
// tenant. The transaction makes that state unreachable.
func (s *Store) CreateTeam(ctx context.Context, id, name string, q auth.Quota) (*Team, error) {
	tx, err := s.Pool.Begin(ctx)
	if err != nil {
		return nil, err
	}
	defer tx.Rollback(ctx)

	var t Team
	err = tx.QueryRow(ctx,
		`INSERT INTO teams (id, name) VALUES ($1, $2) RETURNING id, name, created_at`,
		id, name).Scan(&t.ID, &t.Name, &t.CreatedAt)
	if err != nil {
		return nil, wrap(err)
	}
	_, err = tx.Exec(ctx,
		`INSERT INTO quotas (team_id, rpm, tokens_per_day, max_concurrency)
		 VALUES ($1, $2, $3, $4)`,
		id, q.RPM, q.TokensPerDay, q.MaxConcurrency)
	if err != nil {
		return nil, wrap(err)
	}
	return &t, wrap(tx.Commit(ctx))
}

// Teams lists all tenants.
func (s *Store) Teams(ctx context.Context) ([]Team, error) {
	rows, err := s.Pool.Query(ctx, `SELECT id, name, created_at FROM teams ORDER BY id`)
	if err != nil {
		return nil, wrap(err)
	}
	defer rows.Close()
	out := []Team{}
	for rows.Next() {
		var t Team
		if err := rows.Scan(&t.ID, &t.Name, &t.CreatedAt); err != nil {
			return nil, err
		}
		out = append(out, t)
	}
	return out, rows.Err()
}

// SetQuota upserts a team's limits.
func (s *Store) SetQuota(ctx context.Context, team string, q auth.Quota) error {
	_, err := s.Pool.Exec(ctx,
		`INSERT INTO quotas (team_id, rpm, tokens_per_day, max_concurrency)
		 VALUES ($1, $2, $3, $4)
		 ON CONFLICT (team_id) DO UPDATE
		   SET rpm = EXCLUDED.rpm,
		       tokens_per_day = EXCLUDED.tokens_per_day,
		       max_concurrency = EXCLUDED.max_concurrency`,
		team, q.RPM, q.TokensPerDay, q.MaxConcurrency)
	return wrap(err)
}

// Quotas loads every team's limits, which the gateway caches in its limiter.
func (s *Store) Quotas(ctx context.Context) (map[string]auth.Quota, error) {
	rows, err := s.Pool.Query(ctx,
		`SELECT team_id, rpm, tokens_per_day, max_concurrency FROM quotas`)
	if err != nil {
		return nil, wrap(err)
	}
	defer rows.Close()
	out := map[string]auth.Quota{}
	for rows.Next() {
		var team string
		var q auth.Quota
		if err := rows.Scan(&team, &q.RPM, &q.TokensPerDay, &q.MaxConcurrency); err != nil {
			return nil, err
		}
		out[team] = q
	}
	return out, rows.Err()
}

// CreateAPIKey mints a key and returns the plaintext exactly once. Only the hash and
// a short prefix are persisted, so a database dump contains no usable credentials.
func (s *Store) CreateAPIKey(ctx context.Context, id, team string) (plaintext string, err error) {
	pt, prefix, hash, err := auth.GenerateKey()
	if err != nil {
		return "", err
	}
	_, err = s.Pool.Exec(ctx,
		`INSERT INTO api_keys (id, team_id, key_prefix, key_hash) VALUES ($1, $2, $3, $4)`,
		id, team, prefix, hash)
	if err != nil {
		return "", wrap(err)
	}
	return pt, nil
}

// LookupKey resolves a plaintext key to its team.
//
// The stored prefix narrows the search to the handful of rows sharing it — in
// practice one — so this is an indexed lookup rather than a scan over every key, and
// the hash comparison that follows is constant-time. Revoked keys are excluded by the
// partial index rather than filtered in Go, so a revocation takes effect for every
// gateway immediately.
func (s *Store) LookupKey(ctx context.Context, plaintext string) (*auth.Key, error) {
	rows, err := s.Pool.Query(ctx,
		`SELECT id, team_id, key_prefix, key_hash FROM api_keys
		 WHERE key_prefix = $1 AND revoked_at IS NULL`,
		auth.PrefixOf(plaintext))
	if err != nil {
		return nil, wrap(err)
	}
	defer rows.Close()

	for rows.Next() {
		var k auth.Key
		if err := rows.Scan(&k.ID, &k.TeamID, &k.Prefix, &k.Hash); err != nil {
			return nil, err
		}
		if k.Matches(plaintext) {
			return &k, nil
		}
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return nil, ErrNotFound
}

// RevokeKey marks a key unusable. Idempotent: revoking twice is not an error, since
// the caller is usually a human reacting to a leak and should not be made to care.
func (s *Store) RevokeKey(ctx context.Context, id string) error {
	tag, err := s.Pool.Exec(ctx,
		`UPDATE api_keys SET revoked_at = now() WHERE id = $1 AND revoked_at IS NULL`, id)
	if err != nil {
		return wrap(err)
	}
	if tag.RowsAffected() == 0 {
		// Either unknown or already revoked. Distinguish so a typo'd id is caught.
		var exists bool
		if err := s.Pool.QueryRow(ctx,
			`SELECT EXISTS (SELECT 1 FROM api_keys WHERE id = $1)`, id).Scan(&exists); err != nil {
			return wrap(err)
		}
		if !exists {
			return ErrNotFound
		}
	}
	return nil
}

// TouchKey records that a key was used. Best-effort and deliberately not awaited on
// the request path: it is an operator convenience, and making every request wait for
// a write would be a poor trade.
func (s *Store) TouchKey(ctx context.Context, id string) error {
	_, err := s.Pool.Exec(ctx, `UPDATE api_keys SET last_used_at = now() WHERE id = $1`, id)
	return wrap(err)
}

// UsageRecord is one request's token accounting.
type UsageRecord struct {
	TeamID           string
	ModelID          string
	PromptTokens     int
	CompletionTokens int
	StatusCode       int
}

// RecordUsage appends usage rows.
//
// Batched through pgx's implicit pipelining rather than one round trip per request:
// usage is written on every completion, and a per-request round trip would put the
// database on the latency path of the whole gateway.
func (s *Store) RecordUsage(ctx context.Context, recs []UsageRecord) error {
	if len(recs) == 0 {
		return nil
	}
	batch := &pgx.Batch{}
	for _, r := range recs {
		batch.Queue(
			`INSERT INTO usage (team_id, model_id, prompt_tokens, completion_tokens, status_code)
			 VALUES ($1, $2, $3, $4, $5)`,
			r.TeamID, r.ModelID, r.PromptTokens, r.CompletionTokens, r.StatusCode)
	}
	br := s.Pool.SendBatch(ctx, batch)
	defer br.Close()
	for range recs {
		if _, err := br.Exec(); err != nil {
			return wrap(err)
		}
	}
	return nil
}

// TokensSince totals a team's tokens in a window, used to seed the limiter after a
// gateway restart so a redeploy does not hand everyone a fresh daily budget.
func (s *Store) TokensSince(ctx context.Context, team string, since time.Time) (int64, error) {
	var total *int64
	err := s.Pool.QueryRow(ctx,
		`SELECT SUM(prompt_tokens + completion_tokens) FROM usage
		 WHERE team_id = $1 AND ts >= $2`, team, since).Scan(&total)
	if err != nil {
		return 0, wrap(err)
	}
	if total == nil {
		return 0, nil
	}
	return *total, nil
}

// UsageSummary is per-team, per-model aggregate usage.
type UsageSummary struct {
	TeamID           string `json:"team_id"`
	ModelID          string `json:"model_id"`
	Requests         int64  `json:"requests"`
	PromptTokens     int64  `json:"prompt_tokens"`
	CompletionTokens int64  `json:"completion_tokens"`
}

// UsageBreakdown aggregates usage since a time.
func (s *Store) UsageBreakdown(ctx context.Context, since time.Time) ([]UsageSummary, error) {
	rows, err := s.Pool.Query(ctx,
		`SELECT team_id, model_id, COUNT(*),
		        COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(completion_tokens), 0)
		 FROM usage WHERE ts >= $1
		 GROUP BY team_id, model_id ORDER BY team_id, model_id`, since)
	if err != nil {
		return nil, wrap(err)
	}
	defer rows.Close()
	out := []UsageSummary{}
	for rows.Next() {
		var u UsageSummary
		if err := rows.Scan(&u.TeamID, &u.ModelID, &u.Requests, &u.PromptTokens, &u.CompletionTokens); err != nil {
			return nil, err
		}
		out = append(out, u)
	}
	return out, rows.Err()
}

// Package store is Kiln's PostgreSQL layer: teams, API keys, quotas, usage, the
// model registry and the deploy job queue.
package store

import (
	"context"
	"embed"
	"errors"
	"fmt"
	"sort"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

//go:embed migrations/*.sql
var migrationFS embed.FS

// ErrNotFound is returned when a row does not exist.
var ErrNotFound = errors.New("store: not found")

// ErrConflict is returned when a unique constraint rejects a write — most often the
// one live deploy job per model.
var ErrConflict = errors.New("store: conflicts with an existing row")

// Store wraps a pgx pool.
type Store struct {
	Pool *pgxpool.Pool
}

// Open connects and verifies the connection.
func Open(ctx context.Context, dsn string) (*Store, error) {
	cfg, err := pgxpool.ParseConfig(dsn)
	if err != nil {
		return nil, fmt.Errorf("store: bad DSN: %w", err)
	}
	// The job workers hold a connection for the duration of a claim transaction, and
	// the gateway's usage writes are frequent and short. A small pool starves the
	// former; an unbounded one exhausts Postgres's own connection limit.
	cfg.MaxConns = 16
	cfg.MinConns = 2
	cfg.MaxConnLifetime = time.Hour

	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("store: connect: %w", err)
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("store: ping: %w", err)
	}
	return &Store{Pool: pool}, nil
}

// Close releases the pool.
func (s *Store) Close() { s.Pool.Close() }

// Migrate applies the embedded migrations in filename order.
//
// Embedded rather than read from disk so a container image carries its own schema and
// cannot be deployed against the wrong migration set. Every migration is written to be
// idempotent (CREATE TABLE IF NOT EXISTS and friends) and they are applied inside one
// transaction, so a partial failure leaves the schema untouched rather than
// half-applied.
func (s *Store) Migrate(ctx context.Context) error {
	entries, err := migrationFS.ReadDir("migrations")
	if err != nil {
		return err
	}
	names := make([]string, 0, len(entries))
	for _, e := range entries {
		names = append(names, e.Name())
	}
	sort.Strings(names)

	tx, err := s.Pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)

	// If the DSN asked for a non-default search_path, create that schema first.
	// Test packages use this to get their own isolated copy of the schema: `go test
	// ./...` runs packages in parallel, and two suites truncating one shared set of
	// tables produce failures that look like product bugs and are not.
	if schema := s.Pool.Config().ConnConfig.RuntimeParams["search_path"]; schema != "" && schema != "public" {
		if _, err := tx.Exec(ctx, `CREATE SCHEMA IF NOT EXISTS `+pgx.Identifier{schema}.Sanitize()); err != nil {
			return fmt.Errorf("store: create schema %s: %w", schema, err)
		}
		if _, err := tx.Exec(ctx, `SET LOCAL search_path TO `+pgx.Identifier{schema}.Sanitize()); err != nil {
			return err
		}
	}

	for _, n := range names {
		sqlBytes, err := migrationFS.ReadFile("migrations/" + n)
		if err != nil {
			return err
		}
		if _, err := tx.Exec(ctx, string(sqlBytes)); err != nil {
			return fmt.Errorf("store: migration %s: %w", n, err)
		}
	}
	return tx.Commit(ctx)
}

func wrap(err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, pgx.ErrNoRows) {
		return ErrNotFound
	}
	var pgErr *pgconn.PgError
	if errors.As(err, &pgErr) && pgErr.Code == "23505" { // unique_violation
		return fmt.Errorf("%w: %s", ErrConflict, pgErr.ConstraintName)
	}
	return err
}

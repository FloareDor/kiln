package store

import (
	"context"
	"encoding/json"
	"errors"
	"time"

	"github.com/jackc/pgx/v5"
)

// Job kinds and statuses.
const (
	JobDeploy = "deploy"
	JobScale  = "scale"
	JobDelete = "delete"

	JobPending = "PENDING"
	JobRunning = "RUNNING"
	JobDone    = "DONE"
	JobFailed  = "FAILED"
)

// ErrLeaseLost is returned when a worker tries to finish or renew a job that another
// worker has since reclaimed. It is not a bug — it is the expected outcome when a
// worker stalls past its lease and then wakes up — and the right response is to drop
// the work, not to overwrite the new owner's progress.
var ErrLeaseLost = errors.New("store: lease lost; the job was reclaimed by another worker")

// Job is a unit of deployment work.
type Job struct {
	ID          string          `json:"id"`
	ModelID     string          `json:"model_id"`
	Kind        string          `json:"kind"`
	Status      string          `json:"status"`
	Attempts    int             `json:"attempts"`
	MaxAttempts int             `json:"max_attempts"`
	Payload     json.RawMessage `json:"payload"`
	Error       string          `json:"error,omitempty"`
	LockedBy    *string         `json:"locked_by,omitempty"`
	LeaseUntil  *time.Time      `json:"lease_until,omitempty"`
	RunAfter    time.Time       `json:"run_after"`
	CreatedAt   time.Time       `json:"created_at"`
	UpdatedAt   time.Time       `json:"updated_at"`
}

const jobCols = `id, model_id, kind, status, attempts, max_attempts, payload, error,
	locked_by, lease_until, run_after, created_at, updated_at`

func scanJob(row scanner) (*Job, error) {
	var j Job
	err := row.Scan(&j.ID, &j.ModelID, &j.Kind, &j.Status, &j.Attempts, &j.MaxAttempts,
		&j.Payload, &j.Error, &j.LockedBy, &j.LeaseUntil, &j.RunAfter, &j.CreatedAt, &j.UpdatedAt)
	if err != nil {
		return nil, wrap(err)
	}
	return &j, nil
}

// EnqueueJob adds a job.
//
// A partial unique index allows at most one PENDING-or-RUNNING job per model, so two
// concurrent deploys of the same model cannot race to start replicas and
// double-allocate the GPU. The constraint lives in the database rather than in a
// check-then-insert in Go, because a check-then-insert has a window between the two
// statements that is exactly wide enough for the race it is trying to prevent.
func (s *Store) EnqueueJob(ctx context.Context, id, modelID, kind string, payload any) (*Job, error) {
	raw := json.RawMessage("{}")
	if payload != nil {
		b, err := json.Marshal(payload)
		if err != nil {
			return nil, err
		}
		raw = b
	}
	row := s.Pool.QueryRow(ctx,
		`INSERT INTO deploy_jobs (id, model_id, kind, payload) VALUES ($1,$2,$3,$4)
		 RETURNING `+jobCols,
		id, modelID, kind, raw)
	return scanJob(row)
}

// ClaimJob atomically takes ownership of one claimable job, or returns ErrNotFound.
//
// This is the concurrency-safety core of the control plane, and both halves matter.
//
// `FOR UPDATE SKIP LOCKED` is what lets several control-plane replicas run at once:
// each claim takes a row lock, and SKIP LOCKED makes competing workers step over
// locked rows rather than block on them. Without SKIP LOCKED, N workers serialise
// behind one row and the queue stops being a queue.
//
// The row lock alone is not enough, because it dies with the transaction, and the
// claim has to outlive it — a deploy takes minutes while this transaction commits in
// milliseconds. So the claim is also a *lease*: locked_by plus lease_until. A worker
// that crashes leaves a lease that simply expires and the job becomes claimable
// again; a worker that is merely slow renews it. That is why the WHERE clause accepts
// both genuinely PENDING rows and RUNNING rows whose lease has lapsed.
//
// attempts is incremented at claim time, not at failure time. A worker that is killed
// mid-job never gets to report a failure, and counting only reported failures would
// let a job that reliably crashes its worker retry forever.
func (s *Store) ClaimJob(ctx context.Context, workerID string, lease time.Duration) (*Job, error) {
	tx, err := s.Pool.Begin(ctx)
	if err != nil {
		return nil, err
	}
	defer tx.Rollback(ctx)

	var id string
	err = tx.QueryRow(ctx,
		`SELECT id FROM deploy_jobs
		 WHERE run_after <= now()
		   AND attempts < max_attempts
		   AND (status = 'PENDING'
		        OR (status = 'RUNNING' AND (lease_until IS NULL OR lease_until < now())))
		 ORDER BY created_at
		 FOR UPDATE SKIP LOCKED
		 LIMIT 1`).Scan(&id)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, ErrNotFound
		}
		return nil, wrap(err)
	}

	row := tx.QueryRow(ctx,
		`UPDATE deploy_jobs
		 SET status = 'RUNNING', locked_by = $2, lease_until = now() + $3::interval,
		     attempts = attempts + 1, updated_at = now()
		 WHERE id = $1
		 RETURNING `+jobCols,
		id, workerID, lease.String())
	j, err := scanJob(row)
	if err != nil {
		return nil, err
	}
	return j, wrap(tx.Commit(ctx))
}

// RenewLease extends a running job's lease.
//
// Guarded on locked_by: if the lease already lapsed and another worker reclaimed the
// job, this worker must find out rather than silently extend someone else's claim and
// have two workers deploying the same model at once.
func (s *Store) RenewLease(ctx context.Context, jobID, workerID string, lease time.Duration) error {
	tag, err := s.Pool.Exec(ctx,
		`UPDATE deploy_jobs SET lease_until = now() + $3::interval, updated_at = now()
		 WHERE id = $1 AND locked_by = $2 AND status = 'RUNNING'`,
		jobID, workerID, lease.String())
	if err != nil {
		return wrap(err)
	}
	if tag.RowsAffected() == 0 {
		return ErrLeaseLost
	}
	return nil
}

// CompleteJob marks a job DONE, guarded on ownership.
func (s *Store) CompleteJob(ctx context.Context, jobID, workerID string) error {
	tag, err := s.Pool.Exec(ctx,
		`UPDATE deploy_jobs
		 SET status = 'DONE', locked_by = NULL, lease_until = NULL, error = '', updated_at = now()
		 WHERE id = $1 AND locked_by = $2`,
		jobID, workerID)
	if err != nil {
		return wrap(err)
	}
	if tag.RowsAffected() == 0 {
		return ErrLeaseLost
	}
	return nil
}

// FailJob records a failure and either reschedules the job with backoff or gives up.
//
// backoff is applied via run_after rather than by sleeping in the worker: a job whose
// weights 404 would otherwise spin a worker at full speed, and a worker that sleeps
// is a worker not draining the rest of the queue.
func (s *Store) FailJob(ctx context.Context, jobID, workerID, errMsg string, backoff time.Duration) error {
	tag, err := s.Pool.Exec(ctx,
		`UPDATE deploy_jobs
		 SET status = CASE WHEN attempts >= max_attempts THEN 'FAILED' ELSE 'PENDING' END,
		     error = $3,
		     locked_by = NULL,
		     lease_until = NULL,
		     run_after = now() + $4::interval,
		     updated_at = now()
		 WHERE id = $1 AND locked_by = $2`,
		jobID, workerID, errMsg, backoff.String())
	if err != nil {
		return wrap(err)
	}
	if tag.RowsAffected() == 0 {
		return ErrLeaseLost
	}
	return nil
}

// Job loads one job.
func (s *Store) Job(ctx context.Context, id string) (*Job, error) {
	return scanJob(s.Pool.QueryRow(ctx, `SELECT `+jobCols+` FROM deploy_jobs WHERE id = $1`, id))
}

// Jobs lists jobs, newest first.
func (s *Store) Jobs(ctx context.Context, limit int) ([]*Job, error) {
	if limit <= 0 {
		limit = 100
	}
	rows, err := s.Pool.Query(ctx,
		`SELECT `+jobCols+` FROM deploy_jobs ORDER BY created_at DESC LIMIT $1`, limit)
	if err != nil {
		return nil, wrap(err)
	}
	defer rows.Close()
	out := []*Job{}
	for rows.Next() {
		j, err := scanJob(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, j)
	}
	return out, rows.Err()
}

// JobForModel returns the live job for a model, if any.
func (s *Store) JobForModel(ctx context.Context, modelID string) (*Job, error) {
	return scanJob(s.Pool.QueryRow(ctx,
		`SELECT `+jobCols+` FROM deploy_jobs
		 WHERE model_id = $1 AND status IN ('PENDING','RUNNING')`, modelID))
}

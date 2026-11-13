// Package jobs runs deploy work: claiming jobs from Postgres, driving node agents,
// and recording what was measured.
package jobs

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/FloareDor/kiln/internal/store"
)

// Executor performs the side effects of one job. It is an interface so the worker's
// claim/lease/retry behaviour can be tested against a scripted executor with no node
// agent, no GPU and no HTTP.
type Executor interface {
	Deploy(ctx context.Context, m *store.Model, payload json.RawMessage) error
	Scale(ctx context.Context, m *store.Model, payload json.RawMessage) error
	Delete(ctx context.Context, m *store.Model, payload json.RawMessage) error
}

// Config tunes a worker.
type Config struct {
	// WorkerID identifies this worker in locked_by. Must be unique per process, or
	// two workers can renew and complete each other's claims.
	WorkerID string
	// Lease is how long a claim is valid without renewal. It trades recovery time
	// after a crash against the risk of stealing a job from a worker that is merely
	// slow: too short and healthy jobs get duplicated, too long and a crash blocks
	// the model for that duration.
	Lease time.Duration
	// RenewEvery should be comfortably under Lease. A third is the usual choice: it
	// tolerates two consecutive missed renewals (a GC pause, a database blip) before
	// the lease actually lapses.
	RenewEvery time.Duration
	// PollEvery is how often an idle worker checks for work.
	PollEvery time.Duration
	// BaseBackoff is the first retry delay; it doubles per attempt.
	BaseBackoff time.Duration
	// MaxBackoff caps it.
	MaxBackoff time.Duration
}

// DefaultConfig returns workable defaults. The lease is generous because a cold start
// of a 7B model legitimately takes minutes.
func DefaultConfig(workerID string) Config {
	return Config{
		WorkerID: workerID, Lease: 90 * time.Second, RenewEvery: 30 * time.Second,
		PollEvery: time.Second, BaseBackoff: 5 * time.Second, MaxBackoff: 5 * time.Minute,
	}
}

// Worker claims and runs deploy jobs.
type Worker struct {
	Store *store.Store
	Exec  Executor
	Cfg   Config
	Log   *slog.Logger
}

// NewWorker builds a worker.
func NewWorker(s *store.Store, ex Executor, cfg Config, log *slog.Logger) *Worker {
	if cfg.Lease == 0 {
		cfg = DefaultConfig(cfg.WorkerID)
	}
	return &Worker{Store: s, Exec: ex, Cfg: cfg, Log: log}
}

// Run claims and executes jobs until the context is cancelled.
func (w *Worker) Run(ctx context.Context) {
	t := time.NewTimer(0)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
		worked, err := w.RunOnce(ctx)
		if err != nil && !errors.Is(err, context.Canceled) {
			w.Log.Error("job worker iteration failed", "err", err)
		}
		// Drain the queue greedily when there is work, then back off to the poll
		// interval. Sleeping a full interval between every job would make a batch of
		// ten deploys take ten extra seconds for no reason.
		if worked {
			t.Reset(0)
		} else {
			t.Reset(w.Cfg.PollEvery)
		}
	}
}

// RunOnce claims at most one job and runs it to completion. Returns whether a job was
// found.
func (w *Worker) RunOnce(ctx context.Context) (bool, error) {
	job, err := w.Store.ClaimJob(ctx, w.Cfg.WorkerID, w.Cfg.Lease)
	if errors.Is(err, store.ErrNotFound) {
		return false, nil
	}
	if err != nil {
		return false, err
	}

	log := w.Log.With("job", job.ID, "model", job.ModelID, "kind", job.Kind, "attempt", job.Attempts)
	log.Info("job claimed")

	// Renew in the background for as long as the job runs. Without this, any deploy
	// slower than the lease would be reclaimed by a second worker while the first is
	// still working, and two workers would drive the same GPU.
	jobCtx, stopRenew := context.WithCancel(ctx)
	var renewWG sync.WaitGroup
	lost := make(chan struct{})
	var lostOnce sync.Once
	renewWG.Add(1)
	go func() {
		defer renewWG.Done()
		ticker := time.NewTicker(w.Cfg.RenewEvery)
		defer ticker.Stop()
		for {
			select {
			case <-jobCtx.Done():
				return
			case <-ticker.C:
				if err := w.Store.RenewLease(context.WithoutCancel(jobCtx), job.ID, w.Cfg.WorkerID, w.Cfg.Lease); err != nil {
					if errors.Is(err, store.ErrLeaseLost) {
						// Someone else owns this job now. Stop immediately rather
						// than finishing and clobbering their work.
						log.Warn("lease lost; abandoning the job to its new owner")
						lostOnce.Do(func() { close(lost) })
						return
					}
					log.Warn("lease renewal failed; will retry", "err", err)
				}
			}
		}
	}()

	err = w.execute(jobCtx, job)
	stopRenew()
	renewWG.Wait()

	select {
	case <-lost:
		// The new owner is responsible now; writing anything would corrupt its state.
		return true, nil
	default:
	}

	// Detached context for the bookkeeping: if the process is shutting down, a job
	// that finished must still be recorded as finished, or it is retried needlessly.
	finishCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 30*time.Second)
	defer cancel()

	if err != nil {
		backoff := w.backoff(job.Attempts)
		log.Error("job failed", "err", err, "retry_in", backoff)
		_ = w.Store.SetModelStatus(finishCtx, job.ModelID, store.ModelFailed, err.Error())
		if ferr := w.Store.FailJob(finishCtx, job.ID, w.Cfg.WorkerID, err.Error(), backoff); ferr != nil &&
			!errors.Is(ferr, store.ErrLeaseLost) {
			return true, ferr
		}
		return true, nil
	}

	if cerr := w.Store.CompleteJob(finishCtx, job.ID, w.Cfg.WorkerID); cerr != nil &&
		!errors.Is(cerr, store.ErrLeaseLost) {
		return true, cerr
	}
	log.Info("job done")
	return true, nil
}

func (w *Worker) execute(ctx context.Context, job *store.Job) error {
	m, err := w.Store.Model(ctx, job.ModelID)
	if err != nil {
		return fmt.Errorf("load model %s: %w", job.ModelID, err)
	}
	switch job.Kind {
	case store.JobDeploy:
		if err := w.Store.SetModelStatus(ctx, m.ID, store.ModelDeploying, ""); err != nil {
			return err
		}
		if err := w.Exec.Deploy(ctx, m, job.Payload); err != nil {
			return err
		}
		return w.Store.SetModelStatus(ctx, m.ID, store.ModelReady, "")
	case store.JobScale:
		if err := w.Exec.Scale(ctx, m, job.Payload); err != nil {
			return err
		}
		return w.Store.SetModelStatus(ctx, m.ID, store.ModelReady, "")
	case store.JobDelete:
		if err := w.Store.SetModelStatus(ctx, m.ID, store.ModelDeleting, ""); err != nil {
			return err
		}
		if err := w.Exec.Delete(ctx, m, job.Payload); err != nil {
			return err
		}
		return w.Store.DeleteModel(ctx, m.ID)
	}
	return fmt.Errorf("jobs: unknown kind %q", job.Kind)
}

// backoff returns an exponential delay with jitter.
//
// The jitter is not decoration. Several workers that fail at the same moment — a
// shared dependency going down, which is the common case — would otherwise retry in
// perfect lockstep and hammer it again at exactly the same instant on every attempt.
func (w *Worker) backoff(attempt int) time.Duration {
	if attempt < 1 {
		attempt = 1
	}
	d := time.Duration(float64(w.Cfg.BaseBackoff) * math.Pow(2, float64(attempt-1)))
	if d > w.Cfg.MaxBackoff || d <= 0 {
		d = w.Cfg.MaxBackoff
	}
	jitter := time.Duration(rand.Int63n(int64(d/2) + 1))
	return d/2 + jitter
}

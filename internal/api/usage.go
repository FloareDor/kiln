package api

import (
	"context"
	"log/slog"
	"sync"
	"time"

	"github.com/sairavig/kiln/internal/auth"
	"github.com/sairavig/kiln/internal/routing"
	"github.com/sairavig/kiln/internal/store"
)

// UsageRecorder buffers per-request usage and flushes it to Postgres in batches.
//
// It is the gateway's Observer, so it sits on the completion path of every request.
// That is why it buffers: a synchronous insert per request would put the database on
// the gateway's latency path and make a database hiccup look like a model slowdown.
//
// The cost of buffering is that a hard crash loses at most one flush interval of
// usage rows. That is an acceptable trade for billing-grade-but-not-financial data,
// and it is bounded — the in-memory limiter is updated immediately, so quota
// enforcement never lags even when the write does.
type UsageRecorder struct {
	store   *store.Store
	limiter *auth.Limiter
	log     *slog.Logger

	// FlushEvery bounds how much usage a crash can lose.
	FlushEvery time.Duration
	// MaxBuffer forces an early flush under load, so a burst cannot grow the buffer
	// without limit.
	MaxBuffer int

	mu   sync.Mutex
	buf  []store.UsageRecord
	// QueueWaits is exposed for the metrics layer in M4.
	queueWaits map[string]time.Duration
}

// NewUsageRecorder builds a recorder.
func NewUsageRecorder(s *store.Store, l *auth.Limiter, log *slog.Logger) *UsageRecorder {
	return &UsageRecorder{
		store: s, limiter: l, log: log,
		FlushEvery: 2 * time.Second, MaxBuffer: 512,
		queueWaits: map[string]time.Duration{},
	}
}

// RequestDone records one completed request.
func (u *UsageRecorder) RequestDone(model, team string, _ routing.ReplicaID, res *Result, _ error) {
	if team == "" || res == nil {
		return
	}
	total := int64(res.Usage.PromptTokens + res.Usage.CompletionTokens)
	// Update the limiter synchronously even though the database write is deferred.
	// Quota enforcement must not lag behind actual consumption, or a team can
	// outrun its budget during any flush interval.
	if total > 0 {
		u.limiter.RecordTokens(team, total)
	}

	u.mu.Lock()
	u.buf = append(u.buf, store.UsageRecord{
		TeamID: team, ModelID: model,
		PromptTokens: res.Usage.PromptTokens, CompletionTokens: res.Usage.CompletionTokens,
		StatusCode: res.Status,
	})
	full := len(u.buf) >= u.MaxBuffer
	u.mu.Unlock()

	if full {
		go u.Flush(context.Background())
	}
}

// QueueWait records how long a request waited for its model to become servable.
func (u *UsageRecorder) QueueWait(model string, d time.Duration) {
	u.mu.Lock()
	u.queueWaits[model] = d
	u.mu.Unlock()
}

// Run flushes on a timer until the context is cancelled.
func (u *UsageRecorder) Run(ctx context.Context) {
	t := time.NewTicker(u.FlushEvery)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			u.Flush(ctx)
		}
	}
}

// Flush writes buffered rows.
func (u *UsageRecorder) Flush(ctx context.Context) {
	u.mu.Lock()
	if len(u.buf) == 0 {
		u.mu.Unlock()
		return
	}
	batch := u.buf
	u.buf = nil
	u.mu.Unlock()

	ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 10*time.Second)
	defer cancel()
	if err := u.store.RecordUsage(ctx, batch); err != nil {
		// Do not put the rows back. A failing database with sustained traffic would
		// grow the buffer without bound and take the gateway down with an OOM, which
		// is a far worse outcome than losing a usage row. The limiter already has
		// the tokens, so quotas stay correct either way.
		u.log.Error("dropping usage rows after a failed flush", "rows", len(batch), "err", err)
	}
}

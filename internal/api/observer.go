package api

import (
	"time"

	"github.com/sairavig/kiln/internal/metrics"
	"github.com/sairavig/kiln/internal/routing"
)

// MetricsObserver reports request telemetry to Prometheus.
type MetricsObserver struct{}

// RequestDone records a completed request.
func (MetricsObserver) RequestDone(model, team string, rep routing.ReplicaID, res *Result, err error) {
	if res == nil {
		return
	}
	if team == "" {
		// An unauthenticated gateway (benchmark mode) still needs usable metrics; an
		// empty label value is legal but reads as a bug on a dashboard.
		team = "anonymous"
	}
	metrics.ObserveRequest(model, team, rep, res.Status, res.TTFT, res.Total,
		res.Usage.PromptTokens, res.Usage.CompletionTokens)
	if res.Status == 429 {
		metrics.QuotaRejections.WithLabelValues(team, "gateway_queue").Inc()
	}
}

// QueueWait records how long a request waited for its model to become servable. Once
// tiering exists this is where a wake becomes visible to the client.
func (MetricsObserver) QueueWait(model string, d time.Duration) {
	metrics.QueueWait.WithLabelValues(model).Observe(d.Seconds())
}

// MultiObserver fans telemetry out to several observers.
//
// It exists so metrics and usage accounting stay separate concerns: Prometheus wants
// every request including unauthenticated ones, while usage accounting only wants
// billable ones, and merging them into one type would tangle the two rules.
type MultiObserver []Observer

// RequestDone forwards to each observer.
func (m MultiObserver) RequestDone(model, team string, rep routing.ReplicaID, res *Result, err error) {
	for _, o := range m {
		o.RequestDone(model, team, rep, res, err)
	}
}

// QueueWait forwards to each observer.
func (m MultiObserver) QueueWait(model string, d time.Duration) {
	for _, o := range m {
		o.QueueWait(model, d)
	}
}

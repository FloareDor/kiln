// Package metrics defines Kiln's Prometheus instrumentation.
//
// Every metric here exists to answer a question the project actually asks in its
// experiments. Metrics that merely look thorough are a liability: they cost
// cardinality, they clutter the dashboard, and nobody notices when they go wrong.
package metrics

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/FloareDor/kiln/internal/routing"
)

// Latency buckets.
//
// Hand-chosen rather than prometheus.DefBuckets, which tops out at 10s. LLM serving
// spans four orders of magnitude: a prefix-cache hit answers in ~50ms, a cold start
// takes 40s, and a 7B cold start on a busy GPU can take minutes. DefBuckets would put
// every cold start in +Inf and make the p99 — the number the whole swapping argument
// turns on — unmeasurable.
var latencyBuckets = []float64{
	0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3, 5, 10, 20, 40, 80, 160, 320,
}

// swapBuckets cover sleep and wake transitions, which run from ~50ms (a level-2
// sleep) to a few minutes (a large cold start).
var swapBuckets = []float64{
	0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30, 60, 120, 300,
}

var (
	// TTFT is the headline latency metric. Labelled by model, team and replica:
	// model to compare deployments, team to spot a tenant hurting others, replica to
	// catch one bad GPU. Deliberately *not* labelled by request id or prompt hash,
	// which would be unbounded cardinality.
	TTFT = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "kiln_ttft_seconds",
		Help:    "Time to first token, in seconds.",
		Buckets: latencyBuckets,
	}, []string{"model", "team", "replica"})

	// RequestDuration is end-to-end latency, which for a stream includes the whole
	// generation and so says more about output length than about the platform.
	RequestDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "kiln_request_duration_seconds",
		Help:    "End-to-end request latency, in seconds.",
		Buckets: latencyBuckets,
	}, []string{"model", "team"})

	// RequestsTotal is labelled by status code so error rate is a query away.
	RequestsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "kiln_requests_total",
		Help: "Requests served, by model, team and HTTP status code.",
	}, []string{"model", "team", "status"})

	// TokensTotal feeds both the throughput panel and the billing sanity check.
	TokensTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "kiln_tokens_total",
		Help: "Tokens processed, by model, team and kind (prompt or completion).",
	}, []string{"model", "team", "kind"})

	// TokensPerSecond is output throughput per request. A histogram rather than a
	// gauge: the distribution is what matters, since one request decoding slowly
	// while others are fast is invisible in an average.
	TokensPerSecond = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "kiln_tokens_per_second",
		Help:    "Completion tokens per second, per request.",
		Buckets: []float64{1, 5, 10, 25, 50, 100, 200, 400, 800, 1600},
	}, []string{"model"})

	// QueueDepth is the primary saturation signal and the one KEDA would autoscale
	// on.
	QueueDepth = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "kiln_queue_depth",
		Help: "Requests waiting for a model to become servable.",
	}, []string{"model"})

	// QueueWait is where a wake shows up from the client's point of view: with
	// tiering, a request for a sleeping model waits here rather than failing.
	QueueWait = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "kiln_queue_wait_seconds",
		Help:    "Time spent waiting for a model to become servable, in seconds.",
		Buckets: latencyBuckets,
	}, []string{"model"})

	// QuotaRejections is labelled by which limit fired, so a 429 spike can be
	// attributed rather than guessed at.
	QuotaRejections = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "kiln_quota_rejections_total",
		Help: "Requests rejected by a quota, by team and which limit was hit.",
	}, []string{"team", "reason"})

	InFlight = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "kiln_inflight_requests",
		Help: "Requests currently being served, per replica.",
	}, []string{"model", "replica"})

	// ---------------------------------------------------------------- tiering

	// SwapDuration is labelled from/to so the results table can be read straight off
	// Prometheus: gpu->host, host->gpu, disk->gpu, none->gpu.
	SwapDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "kiln_swap_duration_seconds",
		Help:    "Duration of a tier transition, in seconds.",
		Buckets: swapBuckets,
	}, []string{"model", "from", "to"})

	Evictions = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "kiln_evictions_total",
		Help: "Models evicted from VRAM, by destination tier and eviction policy.",
	}, []string{"model", "to", "policy"})

	Wakes = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "kiln_wakes_total",
		Help: "Models woken into VRAM, by source tier.",
	}, []string{"model", "from"})

	ColdStarts = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "kiln_cold_starts_total",
		Help: "Full cold starts: a process spawned and weights loaded from scratch.",
	}, []string{"model"})

	// ResidencyHits and ResidencyMisses give the ratio that is the clearest single
	// measure of whether a policy is working. Exported as two counters rather than a
	// precomputed ratio gauge so Prometheus can compute it over any window.
	ResidencyHits = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "kiln_residency_hits_total",
		Help: "Requests that found their model already resident in VRAM.",
	}, []string{"model"})

	ResidencyMisses = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "kiln_residency_misses_total",
		Help: "Requests that had to wait for a wake or a cold start.",
	}, []string{"model", "from"})

	// Thrash counts models evicted again within their minimum residency window,
	// which should be zero by construction. A non-zero value means the anti-thrash
	// guard has a hole, and it is the metric that would catch a policy change that
	// looks good on average while oscillating underneath.
	Thrash = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "kiln_thrash_total",
		Help: "Evictions of a model that was resident for less than the minimum residency.",
	}, []string{"model", "policy"})

	SwapFailures = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "kiln_swap_failures_total",
		Help: "Tier transitions that failed.",
	}, []string{"model", "op"})

	DrainTimeouts = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "kiln_drain_timeouts_total",
		Help: "Drains that hit their deadline with requests still in flight.",
	}, []string{"model"})

	// ---------------------------------------------------------------- routing

	PrefixHits = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "kiln_prefix_hits_total",
		Help: "Requests routed to a replica already holding a matching prompt prefix.",
	}, []string{"model", "router"})

	PrefixMisses = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "kiln_prefix_misses_total",
		Help: "Requests routed with no prefix affinity available.",
	}, []string{"model", "router"})

	PrefixBlocksMatched = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "kiln_prefix_blocks_matched",
		Help:    "Leading prompt blocks the chosen replica was known to hold.",
		Buckets: []float64{0, 1, 2, 4, 8, 16, 32, 64},
	}, []string{"model", "router"})

	// ----------------------------------------------------------------- memory

	GPUMemoryBytes = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "kiln_gpu_memory_bytes",
		Help: "GPU memory, by node, device and kind (total, used, free).",
	}, []string{"node", "device", "kind"})

	HostMemoryBytes = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "kiln_host_memory_bytes",
		Help: "Host memory, by node and kind. Bounds how many level-1 sleepers fit.",
	}, []string{"node", "kind"})

	ModelsByState = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "kiln_models_by_state",
		Help: "Models in each lifecycle state; the tiering picture at a glance.",
	}, []string{"state"})

	// ------------------------------------------------------------------- jobs

	JobsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "kiln_deploy_jobs_total",
		Help: "Deploy jobs finished, by kind and outcome.",
	}, []string{"kind", "outcome"})

	JobDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "kiln_deploy_job_duration_seconds",
		Help:    "Deploy job duration, in seconds.",
		Buckets: swapBuckets,
	}, []string{"kind"})

	// JobsReclaimed counts jobs taken over from an expired lease, which is the
	// observable signature of a worker crash.
	JobsReclaimed = promauto.NewCounter(prometheus.CounterOpts{
		Name: "kiln_deploy_jobs_reclaimed_total",
		Help: "Jobs reclaimed from a worker whose lease expired.",
	})
)

// ObserveRequest records one completed request.
func ObserveRequest(model, team string, replica routing.ReplicaID, status int,
	ttft, total time.Duration, promptTokens, completionTokens int) {

	RequestsTotal.WithLabelValues(model, team, statusLabel(status)).Inc()
	RequestDuration.WithLabelValues(model, team).Observe(total.Seconds())

	// Only record TTFT when it was genuinely measured. A non-streaming request has
	// no meaningful time-to-*first*-token, and folding its total latency into the
	// TTFT histogram would silently inflate the project's headline metric.
	if ttft > 0 {
		TTFT.WithLabelValues(model, team, string(replica)).Observe(ttft.Seconds())
	}
	if promptTokens > 0 {
		TokensTotal.WithLabelValues(model, team, "prompt").Add(float64(promptTokens))
	}
	if completionTokens > 0 {
		TokensTotal.WithLabelValues(model, team, "completion").Add(float64(completionTokens))
		// Measure decode rate over the generation phase only. Including prefill
		// would conflate two very different things and make the number depend on
		// prompt length rather than on decode speed.
		if decode := total - ttft; decode > 0 && ttft > 0 {
			TokensPerSecond.WithLabelValues(model).Observe(float64(completionTokens) / decode.Seconds())
		}
	}
}

// statusLabel buckets status codes into classes.
//
// Raw codes would be fine today but are a cardinality trap: an upstream that starts
// returning a long tail of distinct codes would multiply every series. Classes answer
// the questions anyone actually asks.
func statusLabel(code int) string {
	switch {
	case code == 0:
		return "none"
	case code >= 500:
		return "5xx"
	case code == 429:
		return "429"
	case code >= 400:
		return "4xx"
	case code >= 200 && code < 300:
		return "2xx"
	}
	return "other"
}

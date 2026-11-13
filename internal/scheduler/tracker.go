package scheduler

import (
	"math"
	"sync"
	"time"
)

// Tracker maintains the per-model demand statistics the policies score on: an
// exponentially decayed arrival rate and its trend.
//
// Why not a plain counter: a serving system's useful question is "how busy is this
// model *now*", and a counter answers "how busy has it ever been". A sliding window
// would answer it but needs per-request storage proportional to the window. An EWMA
// answers it in two floats per model, which matters when the scheduler ticks several
// times a second across hundreds of models.
//
// Tracker is the one stateful piece in this package. It is concurrency-safe because
// the gateway calls Observe from request handlers while the control plane reads a
// snapshot from its scheduling loop.
type Tracker struct {
	mu       sync.Mutex
	halfLife time.Duration
	// trendHalfLife is deliberately longer than halfLife: the trend is a comparison
	// between a fast and a slow average, and if both move at the same speed their
	// difference is always ~0 and the predictive policy never fires.
	trendHalfLife time.Duration
	models        map[ModelID]*rateState
}

type rateState struct {
	fast     float64   // decayed rate, req/s
	slow     float64   // slower-decaying rate, used only to derive the trend
	lastSeen time.Time // last Observe
	lastTick time.Time // last decay application
	seeded   bool
}

// NewTracker builds a Tracker. halfLife is how long it takes an idle model's measured
// rate to halve: 30s makes the scheduler responsive to bursts, several minutes makes
// it stable across them. 60s is the default used in the experiments.
func NewTracker(halfLife time.Duration) *Tracker {
	if halfLife <= 0 {
		halfLife = time.Minute
	}
	return &Tracker{
		halfLife:      halfLife,
		trendHalfLife: 4 * halfLife,
		models:        make(map[ModelID]*rateState),
	}
}

// decayFactor is the multiplier applied to a rate after dt has elapsed with no
// arrivals: 0.5 per half-life.
func decayFactor(dt, halfLife time.Duration) float64 {
	if dt <= 0 {
		return 1
	}
	return math.Exp(-math.Ln2 * dt.Seconds() / halfLife.Seconds())
}

func (t *Tracker) stateFor(id ModelID, now time.Time) *rateState {
	st, ok := t.models[id]
	if !ok {
		st = &rateState{lastTick: now, lastSeen: now}
		t.models[id] = st
	}
	return st
}

// advance applies decay up to now. Called before every read and every write so the
// rate is correct at an arbitrary instant rather than only at tick boundaries.
func (st *rateState) advance(now time.Time, halfLife, trendHalfLife time.Duration) {
	dt := now.Sub(st.lastTick)
	if dt <= 0 {
		return
	}
	st.fast *= decayFactor(dt, halfLife)
	st.slow *= decayFactor(dt, trendHalfLife)
	st.lastTick = now
}

// Observe records n arrivals for a model at time now.
//
// The increment is 1/halfLife rather than 1, so the EWMA converges to the true
// arrival rate in requests per second under steady load: a model receiving r req/s
// settles at approximately r. Incrementing by 1 would make the units arbitrary and
// the HostTierMinRate / PrewarmMinRate thresholds impossible to reason about.
func (t *Tracker) Observe(id ModelID, now time.Time, n int) {
	if n <= 0 {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	st := t.stateFor(id, now)
	st.advance(now, t.halfLife, t.trendHalfLife)
	st.fast += float64(n) * math.Ln2 / t.halfLife.Seconds()
	st.slow += float64(n) * math.Ln2 / t.trendHalfLife.Seconds()
	st.lastSeen = now
	st.seeded = true
}

// Rate returns the decayed arrival rate in req/s.
func (t *Tracker) Rate(id ModelID, now time.Time) float64 {
	t.mu.Lock()
	defer t.mu.Unlock()
	st, ok := t.models[id]
	if !ok {
		return 0
	}
	st.advance(now, t.halfLife, t.trendHalfLife)
	return st.fast
}

// Trend is the rate's derivative in req/s per second, estimated as the gap between
// the fast and slow averages normalised by the fast half-life. Positive means demand
// is climbing, which is what the predictive policy prewarms on.
func (t *Tracker) Trend(id ModelID, now time.Time) float64 {
	t.mu.Lock()
	defer t.mu.Unlock()
	st, ok := t.models[id]
	if !ok {
		return 0
	}
	st.advance(now, t.halfLife, t.trendHalfLife)
	return (st.fast - st.slow) / t.halfLife.Seconds()
}

// LastSeen is when a request for the model last arrived, zero if never.
func (t *Tracker) LastSeen(id ModelID) time.Time {
	t.mu.Lock()
	defer t.mu.Unlock()
	if st, ok := t.models[id]; ok {
		return st.lastSeen
	}
	return time.Time{}
}

// Apply stamps the tracker's statistics onto a ClusterState before it is handed to a
// policy. Keeping this as an explicit step means a policy can be unit-tested with
// hand-written Rate and Trend values and no Tracker at all.
func (t *Tracker) Apply(c *ClusterState) {
	t.mu.Lock()
	defer t.mu.Unlock()
	for id, m := range c.Models {
		st, ok := t.models[id]
		if !ok {
			continue
		}
		st.advance(c.Now, t.halfLife, t.trendHalfLife)
		m.Rate = st.fast
		m.Trend = (st.fast - st.slow) / t.halfLife.Seconds()
		if st.seeded && st.lastSeen.After(m.LastRequest) {
			m.LastRequest = st.lastSeen
		}
	}
}

// Forget drops a model's statistics, called when a deployment is deleted so a
// redeployed model under the same name does not inherit stale demand.
func (t *Tracker) Forget(id ModelID) {
	t.mu.Lock()
	defer t.mu.Unlock()
	delete(t.models, id)
}

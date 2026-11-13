package nodeagent

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/sairavig/kiln/internal/metrics"
)

// NodeStatus is what GET /status returns: everything the control plane needs to make
// a placement decision for this node in one round trip.
type NodeStatus struct {
	Node     string    `json:"node"`
	GPUs     []GPUInfo `json:"gpus"`
	Host     HostInfo  `json:"host"`
	Replicas []Status  `json:"replicas"`
}

// Server is the node agent's HTTP API.
type Server struct {
	Node string
	Sup  *Supervisor
	GPU  *GPUProbe
	Log  *slog.Logger
}

// NewServer builds the API.
func NewServer(node string, sup *Supervisor, log *slog.Logger) *Server {
	return &Server{Node: node, Sup: sup, GPU: NewGPUProbe(time.Second), Log: log}
}

// Routes returns the mux. The endpoints mirror the scheduler's vocabulary — start,
// sleep, wake, stop — so a swap plan maps onto HTTP calls one for one.
func (s *Server) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", s.health)
	mux.HandleFunc("GET /status", s.status)
	mux.HandleFunc("POST /replicas", s.start)
	mux.HandleFunc("GET /replicas/{id}", s.replica)
	mux.HandleFunc("DELETE /replicas/{id}", s.stop)
	mux.HandleFunc("POST /replicas/{id}/sleep", s.sleep)
	mux.HandleFunc("POST /replicas/{id}/wake", s.wake)
	mux.Handle("GET /metrics", promhttp.Handler())
	return mux
}

// PublishMemory exports this node's GPU and host memory to Prometheus.
//
// Host memory is published alongside VRAM because it is the real ceiling on the host
// tier: VRAM decides how many models can be *active*, host RAM decides how many can
// be kept warm, and running out of the latter gets the node OOM-killed rather than
// merely slowed down.
func (s *Server) PublishMemory(ctx context.Context, every time.Duration) {
	t := time.NewTicker(every)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			if gpus, err := s.GPU.Query(ctx); err == nil {
				for _, g := range gpus {
					dev := strconv.Itoa(g.Index)
					metrics.GPUMemoryBytes.WithLabelValues(s.Node, dev, "total").Set(float64(g.TotalBytes))
					metrics.GPUMemoryBytes.WithLabelValues(s.Node, dev, "used").Set(float64(g.UsedBytes))
					metrics.GPUMemoryBytes.WithLabelValues(s.Node, dev, "free").Set(float64(g.FreeBytes))
				}
			}
			if h, err := ReadHostInfo(); err == nil {
				metrics.HostMemoryBytes.WithLabelValues(s.Node, "total").Set(float64(h.TotalBytes))
				metrics.HostMemoryBytes.WithLabelValues(s.Node, "available").Set(float64(h.AvailableBytes))
			}
			byState := map[string]int{}
			for _, r := range s.Sup.List() {
				byState[r.State]++
			}
			for _, st := range []string{"COLD", "STARTING", "ACTIVE", "DRAINING",
				"SLEEPING_L1", "SLEEPING_L2", "WAKING", "FAILED"} {
				// Publish every state, including the zeros. A gauge that simply
				// disappears when a state empties leaves the last value on the
				// dashboard forever, which reads as a model stuck in that state.
				metrics.ModelsByState.WithLabelValues(st).Set(float64(byState[st]))
			}
		}
	}
}

func (s *Server) health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok", "node": s.Node})
}

func (s *Server) status(w http.ResponseWriter, r *http.Request) {
	out := NodeStatus{Node: s.Node, Replicas: s.Sup.List()}
	// A GPU read failing must not blank the replica list: the control plane can still
	// route to healthy replicas using its own last-known budget, and a 500 here would
	// take the whole node out of rotation over a transient nvidia-smi hiccup.
	if gpus, err := s.GPU.Query(r.Context()); err != nil {
		s.Log.Warn("gpu probe failed", "err", err)
	} else {
		out.GPUs = gpus
	}
	if host, err := ReadHostInfo(); err == nil {
		out.Host = host
	}
	writeJSON(w, http.StatusOK, out)
}

func (s *Server) start(w http.ResponseWriter, r *http.Request) {
	var spec Spec
	if err := json.NewDecoder(r.Body).Decode(&spec); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	if spec.GPUMemoryUtilization <= 0 {
		spec.GPUMemoryUtilization = 0.35
	}
	// A cold start can take minutes and must not be cancelled by the caller's
	// timeout: an abandoned half-loaded model holds GPU memory nobody is tracking.
	// The request is detached from the client's context on purpose.
	ctx, cancel := context.WithTimeout(context.WithoutCancel(r.Context()), s.Sup.cfg.StartTimeout)
	defer cancel()

	st, err := s.Sup.Start(ctx, spec)
	if err != nil {
		s.Log.Error("start failed", "replica", spec.Replica, "err", err)
		writeJSON(w, http.StatusInternalServerError, map[string]any{"error": err.Error(), "replica": st})
		return
	}
	metrics.ColdStarts.WithLabelValues(spec.Model).Inc()
	metrics.SwapDuration.WithLabelValues(spec.Model, "none", "gpu").Observe(float64(st.ColdStartMS) / 1000)
	s.Log.Info("replica started", "replica", spec.Replica, "port", st.Port, "cold_start_ms", st.ColdStartMS)
	writeJSON(w, http.StatusOK, st)
}

func (s *Server) replica(w http.ResponseWriter, r *http.Request) {
	rep, ok := s.Sup.Get(r.PathValue("id"))
	if !ok {
		writeErr(w, http.StatusNotFound, errNotFound)
		return
	}
	writeJSON(w, http.StatusOK, rep.Status())
}

func (s *Server) stop(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if err := s.Sup.Stop(context.WithoutCancel(r.Context()), id); err != nil {
		writeErr(w, http.StatusNotFound, err)
		return
	}
	s.Log.Info("replica stopped", "replica", id)
	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) sleep(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Level int    `json:"level"`
		Mode  string `json:"mode"`
	}
	_ = json.NewDecoder(r.Body).Decode(&body)
	if body.Level == 0 {
		body.Level = 1
	}
	ctx, cancel := context.WithTimeout(context.WithoutCancel(r.Context()), 10*time.Minute)
	defer cancel()

	start := time.Now()
	st, err := s.Sup.Sleep(ctx, r.PathValue("id"), body.Level, body.Mode)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"error": err.Error(), "replica": st})
		return
	}
	to := "host"
	if body.Level == 2 {
		to = "disk"
	}
	metrics.SwapDuration.WithLabelValues(st.Spec.Model, "gpu", to).Observe(time.Since(start).Seconds())
	metrics.Evictions.WithLabelValues(st.Spec.Model, to, "agent").Inc()
	s.Log.Info("replica slept", "replica", r.PathValue("id"), "level", body.Level, "state", st.State)
	writeJSON(w, http.StatusOK, st)
}

func (s *Server) wake(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(context.WithoutCancel(r.Context()), 15*time.Minute)
	defer cancel()

	start := time.Now()
	st, err := s.Sup.Wake(ctx, r.PathValue("id"))
	if err != nil {
		metrics.SwapFailures.WithLabelValues(st.Spec.Model, "wake").Inc()
		writeJSON(w, http.StatusInternalServerError, map[string]any{"error": err.Error(), "replica": st})
		return
	}
	from := "host"
	if st.WakeDiskMS > st.WakeHostMS {
		from = "disk"
	}
	metrics.SwapDuration.WithLabelValues(st.Spec.Model, from, "gpu").Observe(time.Since(start).Seconds())
	metrics.Wakes.WithLabelValues(st.Spec.Model, from).Inc()
	s.Log.Info("replica woke", "replica", r.PathValue("id"),
		"wake_host_ms", st.WakeHostMS, "wake_disk_ms", st.WakeDiskMS)
	writeJSON(w, http.StatusOK, st)
}

type apiError struct{ msg string }

func (e apiError) Error() string { return e.msg }

var errNotFound = apiError{"replica not found"}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, code int, err error) {
	writeJSON(w, code, map[string]string{"error": err.Error()})
}

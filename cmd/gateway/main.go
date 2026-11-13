// Command gateway serves the OpenAI-compatible API and owns GPU residency.
//
// Residency lives here rather than in the control plane because draining does: a
// model may only be slept once its in-flight requests have finished, and the gateway
// is the only component that knows that count exactly and instantly. The control
// plane owns deployments (what exists, what it costs); the gateway owns which of
// those are in VRAM right now.
package main

import (
	"context"
	"flag"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/sairavig/kiln/internal/api"
	"github.com/sairavig/kiln/internal/auth"
	"github.com/sairavig/kiln/internal/routing"
	"github.com/sairavig/kiln/internal/scheduler"
	"github.com/sairavig/kiln/internal/store"
	"github.com/sairavig/kiln/internal/swap"
)

func main() {
	var (
		addr       = flag.String("addr", envOr("KILN_GATEWAY_ADDR", ":8080"), "listen address")
		agentsCSV  = flag.String("agents", envOr("KILN_AGENTS", "http://127.0.0.1:8081"), "comma-separated node agent base URLs")
		dsn        = flag.String("database-url", os.Getenv("KILN_DATABASE_URL"), "PostgreSQL DSN; empty disables auth, quotas and tiering")
		routerName = flag.String("router", envOr("KILN_ROUTER", "round-robin"), "router: "+strings.Join(routing.RouterNames(), ", "))
		policyName = flag.String("policy", envOr("KILN_POLICY", ""), "eviction policy: "+strings.Join(scheduler.PolicyNames(), ", ")+"; empty disables tiering")
		pollEvery  = flag.Duration("poll", 2*time.Second, "how often to poll node agents for the fleet view")
		queueDepth = flag.Int("max-queue", 256, "per-model queue depth before shedding load")
		queueWait  = flag.Duration("queue-timeout", 180*time.Second, "how long a request waits for a model to become servable")

		gpuTotal    = flag.Int64("gpu-bytes", envInt64("KILN_GPU_BYTES", 8*1024*1024*1024), "total VRAM on the GPU the scheduler budgets against")
		gpuReserve  = flag.Int64("gpu-reserve-bytes", envInt64("KILN_GPU_RESERVE_BYTES", 1024*1024*1024), "VRAM held back for the CUDA context, fragmentation and the desktop compositor")
		hostBudget  = flag.Int64("host-ram-bytes", envInt64("KILN_HOST_RAM_BYTES", 16*1024*1024*1024), "pinned host memory budget for level-1 sleepers")
		minResidency = flag.Duration("min-residency", 15*time.Second, "how long a model must serve before it may be evicted (anti-thrash)")
		drainDeadline = flag.Duration("drain-deadline", 10*time.Second, "how long to wait for in-flight requests before sleeping a victim")
		tickEvery    = flag.Duration("tick", 5*time.Second, "how often the policy does background prewarm/demote work")
	)
	flag.Parse()

	log := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))

	router, err := routing.NewRouter(*routerName)
	if err != nil {
		log.Error("bad router", "err", err)
		os.Exit(2)
	}

	var agents []string
	for _, u := range strings.Split(*agentsCSV, ",") {
		if u = strings.TrimSpace(u); u != "" {
			agents = append(agents, u)
		}
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	// The fleet view comes from polling node agents even when a database is present.
	// The database knows what *should* be running; the agents know what *is*, and
	// routing to a replica the database believes in but that died two minutes ago is
	// a worse failure than briefly missing one that just came up.
	agentReg := api.NewAgentRegistry(agents, log)
	go agentReg.Run(ctx, *pollEvery)

	reg := &api.HookedRegistry{Registry: agentReg}

	cfg := api.DefaultConfig()
	cfg.MaxQueueDepth = *queueDepth
	cfg.QueueTimeout = *queueWait

	obs := api.MultiObserver{api.MetricsObserver{}}
	srv := api.NewServer(reg, router, cfg, log, obs)

	var (
		usage   *api.UsageRecorder
		handler http.Handler
		st      *store.Store
	)

	if *dsn != "" {
		st, err = store.Open(ctx, *dsn)
		if err != nil {
			log.Error("database", "err", err)
			os.Exit(1)
		}
		defer st.Close()

		limiter := auth.NewLimiter(time.Now)
		authn := api.NewAuthenticator(st, limiter, log)
		if err := authn.LoadQuotas(ctx); err != nil {
			log.Error("load quotas", "err", err)
			os.Exit(1)
		}
		go authn.RunQuotaRefresh(ctx, 30*time.Second)

		usage = api.NewUsageRecorder(st, limiter, log)
		go usage.Run(ctx)

		srv = api.NewServer(reg, router, cfg, log,
			api.MultiObserver{api.MetricsObserver{}, usage})
		handler = authn.Middleware(withMetrics(srv.Routes()))
		log.Info("auth and quotas enabled")
	} else {
		// No database: an open gateway. Useful for benchmarking, where auth would
		// only add noise to the latency being measured, and explicitly logged so it
		// is never mistaken for a production configuration.
		handler = withMetrics(srv.Routes())
		log.Warn("no database configured: running WITHOUT authentication, quotas or tiering")
	}

	// ------------------------------------------------------------ GPU tiering
	if *policyName != "" && st != nil {
		policy, err := scheduler.NewPolicy(*policyName)
		if err != nil {
			log.Error("bad eviction policy", "err", err)
			os.Exit(2)
		}

		pool := swap.NewAgentPool()
		fleet := swap.NewStoreFleet(st, pool, srv.InFlightForModel, srv.QueueDepth)
		go fleet.Run(ctx)

		scfg := scheduler.DefaultConfig()
		scfg.MinResidency = *minResidency
		scfg.DrainDeadline = *drainDeadline
		scfg.HostRAMBudget = *hostBudget

		ctrl := swap.New(fleet, pool, policy, scfg,
			scheduler.GPU{Node: "local", TotalBytes: *gpuTotal, ReservedBytes: *gpuReserve},
			log.With("component", "swap"))

		// Install the hooks now that both halves exist.
		reg.Ensurer = ctrl.Ensure
		reg.StateFn = func(model string) (string, bool) {
			s, ok := ctrl.States()[model]
			return s, ok
		}

		go ctrl.RunTick(ctx, *tickEvery)
		// Reconcile against what the agents actually report, so the controller's view
		// is eventually corrected by observation rather than trusted indefinitely.
		go func() {
			t := time.NewTicker(*pollEvery)
			defer t.Stop()
			for {
				select {
				case <-ctx.Done():
					return
				case <-t.C:
					// Report a model as resident if ANY replica is: the
					// controller budgets per model, and one live replica means
					// the weights are on the device.
					observed := map[string]string{}
					for _, m := range agentReg.Models() {
						for _, rp := range agentReg.Replicas(m) {
							if cur, ok := observed[m]; !ok || cur != "ACTIVE" {
								observed[m] = rp.State
							}
						}
					}
					ctrl.Sync(ctx, observed)
				}
			}
		}()

		log.Info("GPU memory tiering enabled",
			"policy", policy.Name(),
			"gpu_budget_bytes", *gpuTotal-*gpuReserve,
			"host_budget_bytes", *hostBudget,
			"min_residency", *minResidency,
			"drain_deadline", *drainDeadline)
	}

	httpSrv := &http.Server{
		Addr:    *addr,
		Handler: handler,
		// No write timeout: a streaming generation can legitimately run for minutes,
		// and a deadline here truncates it mid-stream with no useful error.
		ReadHeaderTimeout: 10 * time.Second,
	}

	go func() {
		log.Info("gateway listening", "addr", *addr, "router", router.Name(), "agents", agents)
		if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Error("listen failed", "err", err)
			os.Exit(1)
		}
	}()

	<-ctx.Done()
	log.Info("gateway shutting down; draining in-flight requests")
	// Generous drain: cutting a streaming response at shutdown gives the client a
	// truncated answer with no error, which is worse than a slightly slow deploy.
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	_ = httpSrv.Shutdown(shutdownCtx)
	if usage != nil {
		// Flush buffered usage rows. Dropping them on shutdown would let a tenant
		// reset their daily spend by waiting for a deploy.
		usage.Flush(context.Background())
	}
	log.Info("gateway stopped")
}

// withMetrics mounts /metrics alongside the API.
//
// Served from the gateway process itself rather than a sidecar: the in-process
// counters are the authoritative ones, and anything scraping a sidecar would be
// reporting a copy with its own staleness.
func withMetrics(next http.Handler) http.Handler {
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.Handler())
	mux.Handle("/", next)
	return mux
}

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func envInt64(k string, def int64) int64 {
	v := os.Getenv(k)
	if v == "" {
		return def
	}
	out, err := strconv.ParseInt(v, 10, 64)
	if err != nil {
		return def
	}
	return out
}

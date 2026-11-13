// Command gateway serves the OpenAI-compatible API.
package main

import (
	"context"
	"flag"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/sairavig/kiln/internal/api"
	"github.com/sairavig/kiln/internal/auth"
	"github.com/sairavig/kiln/internal/routing"
	"github.com/sairavig/kiln/internal/store"
)

func main() {
	var (
		addr       = flag.String("addr", envOr("KILN_GATEWAY_ADDR", ":8080"), "listen address")
		agentsCSV  = flag.String("agents", envOr("KILN_AGENTS", "http://127.0.0.1:8081"), "comma-separated node agent base URLs")
		dsn        = flag.String("database-url", os.Getenv("KILN_DATABASE_URL"), "PostgreSQL DSN; empty disables auth and quotas")
		routerName = flag.String("router", envOr("KILN_ROUTER", "round-robin"), "router: "+strings.Join(routing.RouterNames(), ", "))
		pollEvery  = flag.Duration("poll", 2*time.Second, "how often to poll node agents for the fleet view")
		queueDepth = flag.Int("max-queue", 256, "per-model queue depth before shedding load")
		queueWait  = flag.Duration("queue-timeout", 120*time.Second, "how long a request waits for a model to become servable")
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
	reg := api.NewAgentRegistry(agents, log)
	go reg.Run(ctx, *pollEvery)

	cfg := api.DefaultConfig()
	cfg.MaxQueueDepth = *queueDepth
	cfg.QueueTimeout = *queueWait

	var usage *api.UsageRecorder
	var handler http.Handler

	// Prometheus always observes; usage accounting is added only when there is a
	// database to write to. Keeping them separate means a benchmark run without auth
	// still produces the full metric set the experiments read.
	obs := api.MultiObserver{api.MetricsObserver{}}
	srv := api.NewServer(reg, router, cfg, log, obs)

	if *dsn != "" {
		st, err := store.Open(ctx, *dsn)
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
		// Refresh so an admin quota change takes effect without a restart.
		go authn.RunQuotaRefresh(ctx, 30*time.Second)

		usage = api.NewUsageRecorder(st, limiter, log)
		go usage.Run(ctx)

		srv = api.NewServer(reg, router, cfg, log, api.MultiObserver{api.MetricsObserver{}, usage})
		handler = authn.Middleware(withMetrics(srv.Routes()))
		log.Info("auth and quotas enabled")
	} else {
		// No database: an open gateway. Useful for benchmarking, where auth would
		// only add noise to the latency being measured, and explicitly logged so it
		// is never mistaken for a production configuration.
		handler = withMetrics(srv.Routes())
		log.Warn("no database configured: running WITHOUT authentication or quotas")
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
// counters are the authoritative ones, and anything that scraped a sidecar would be
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

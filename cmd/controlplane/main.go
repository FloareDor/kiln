// Command controlplane owns the model registry, the admin API and the deploy job
// workers.
//
// It is deliberately separate from the gateway. The gateway must keep serving when
// the control plane is down — a platform whose data path depends on its control path
// has outages its users cannot explain — and separating them makes that structural
// rather than a promise.
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/google/uuid"
	"github.com/sairavig/kiln/internal/api"
	"github.com/sairavig/kiln/internal/jobs"
	"github.com/sairavig/kiln/internal/store"
)

func main() {
	var (
		addr       = flag.String("addr", envOr("KILN_CP_ADDR", ":8082"), "listen address")
		dsn        = flag.String("database-url", os.Getenv("KILN_DATABASE_URL"), "PostgreSQL DSN")
		agentsCSV  = flag.String("agents", envOr("KILN_AGENTS", "http://127.0.0.1:8081"), "comma-separated node agent base URLs")
		adminToken = flag.String("admin-token", os.Getenv("KILN_ADMIN_TOKEN"), "shared token for the admin API; empty disables it")
		workers    = flag.Int("workers", 2, "deploy job workers in this process")
		measure    = flag.Bool("measure-wakes", true, "measure sleep/wake costs at deploy time; needed by the cost-aware eviction policy")
		migrateOnly = flag.Bool("migrate-only", false, "apply migrations and exit")
	)
	flag.Parse()

	log := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	if *dsn == "" {
		log.Error("KILN_DATABASE_URL or -database-url is required")
		os.Exit(2)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	st, err := store.Open(ctx, *dsn)
	if err != nil {
		log.Error("database", "err", err)
		os.Exit(1)
	}
	defer st.Close()

	// Migrate on start. Every migration is idempotent and they run in one
	// transaction, so a control plane in a crash loop cannot half-apply a schema.
	if err := st.Migrate(ctx); err != nil {
		log.Error("migrate", "err", err)
		os.Exit(1)
	}
	log.Info("migrations applied")
	if *migrateOnly {
		return
	}

	var agents []string
	for _, u := range strings.Split(*agentsCSV, ",") {
		if u = strings.TrimSpace(u); u != "" {
			agents = append(agents, u)
		}
	}
	deployer := jobs.NewDeployer(st, agents, log, *measure)

	var wg sync.WaitGroup
	for i := 0; i < *workers; i++ {
		// A unique worker id per goroutine. Sharing one id would let two workers
		// renew and complete each other's claims, defeating the lease entirely.
		id := fmt.Sprintf("%s-%d-%s", hostname(), i, uuid.NewString()[:8])
		w := jobs.NewWorker(st, deployer, jobs.DefaultConfig(id), log.With("worker", id))
		wg.Add(1)
		go func() {
			defer wg.Done()
			w.Run(ctx)
		}()
	}
	log.Info("job workers started", "count", *workers, "agents", agents)

	admin := api.NewAdmin(st, log)
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		if err := st.Pool.Ping(r.Context()); err != nil {
			http.Error(w, `{"status":"degraded"}`, http.StatusServiceUnavailable)
			return
		}
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	})
	mux.Handle("/admin/", api.AdminAuth(*adminToken, admin.Routes()))

	httpSrv := &http.Server{Addr: *addr, Handler: mux, ReadHeaderTimeout: 10 * time.Second}
	go func() {
		log.Info("control plane listening", "addr", *addr, "admin_enabled", *adminToken != "")
		if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Error("listen failed", "err", err)
			os.Exit(1)
		}
	}()

	<-ctx.Done()
	log.Info("control plane shutting down")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	_ = httpSrv.Shutdown(shutdownCtx)
	// Workers exit on ctx, but a job already running finishes its bookkeeping on a
	// detached context so it is not retried needlessly after the restart.
	wg.Wait()
	log.Info("control plane stopped")
}

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func hostname() string {
	h, err := os.Hostname()
	if err != nil {
		return "cp"
	}
	return h
}

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

	"github.com/sairavig/kiln/internal/api"
	"github.com/sairavig/kiln/internal/routing"
)

func main() {
	var (
		addr       = flag.String("addr", envOr("KILN_GATEWAY_ADDR", ":8080"), "listen address")
		agentsCSV  = flag.String("agents", envOr("KILN_AGENTS", "http://127.0.0.1:8081"), "comma-separated node agent base URLs")
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
	reg := api.NewAgentRegistry(agents, log)

	cfg := api.DefaultConfig()
	cfg.MaxQueueDepth = *queueDepth
	cfg.QueueTimeout = *queueWait
	srv := api.NewServer(reg, router, cfg, log, nil)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	go reg.Run(ctx, *pollEvery)

	httpSrv := &http.Server{
		Addr:    *addr,
		Handler: srv.Routes(),
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
	log.Info("gateway stopped")
}

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

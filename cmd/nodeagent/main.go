// Command nodeagent owns one GPU and runs vLLM processes on it.
//
// It runs as a DaemonSet pod under Kubernetes and as a plain container elsewhere —
// notably on Modal, which has no Kubernetes — so it takes all of its configuration
// from flags and environment rather than from the cluster API.
package main

import (
	"context"
	"flag"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/FloareDor/kiln/internal/nodeagent"
)

func main() {
	var (
		addr      = flag.String("addr", envOr("KILN_AGENT_ADDR", ":8081"), "listen address")
		node      = flag.String("node", envOr("KILN_NODE", hostname()), "node name reported to the control plane")
		python    = flag.String("python", envOr("KILN_PYTHON", "python"), "python interpreter that has vLLM installed")
		portBase  = flag.Int("port-base", 19000, "first port assigned to a vLLM replica")
		logDir    = flag.String("log-dir", envOr("KILN_LOG_DIR", "/tmp/kiln-logs"), "where per-replica vLLM logs are written")
		hfHome    = flag.String("hf-home", os.Getenv("HF_HOME"), "shared Hugging Face cache; sharing it is what makes a level-2 wake cheap")
		startWait = flag.Duration("start-timeout", 15*time.Minute, "how long a cold start may take")
		advertise = flag.String("advertise-host", envOr("KILN_ADVERTISE_HOST", "127.0.0.1"),
			"host other components use to reach this node's replicas; set to the pod IP under Kubernetes")
	)
	flag.Parse()

	log := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))

	sup := nodeagent.NewSupervisor(nodeagent.Config{
		PythonBin: *python, PortBase: *portBase, LogDir: *logDir,
		HFHome: *hfHome, StartTimeout: *startWait, AdvertiseHost: *advertise,
	})
	srv := nodeagent.NewServer(*node, sup, log)

	httpSrv := &http.Server{
		Addr:    *addr,
		Handler: srv.Routes(),
		// No write timeout: a start call legitimately blocks for minutes while a
		// model loads, and a timeout here would abandon a half-loaded model that
		// still holds GPU memory nobody is tracking.
		ReadHeaderTimeout: 10 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	// Export memory and state gauges on a timer so Prometheus sees a live picture
	// even when no requests are arriving.
	go srv.PublishMemory(ctx, 2*time.Second)

	go func() {
		log.Info("node agent listening", "addr", *addr, "node", *node,
			"python", *python, "advertise_host", *advertise)
		if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Error("listen failed", "err", err)
			os.Exit(1)
		}
	}()

	<-ctx.Done()
	// Stop every vLLM child before exiting. A restarted agent that finds orphaned
	// processes still holding the GPU cannot place anything, and the orphans are
	// invisible to Kubernetes because they are not pods.
	log.Info("shutting down; stopping replicas")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	sup.Shutdown(shutdownCtx)
	_ = httpSrv.Shutdown(shutdownCtx)
	log.Info("node agent stopped")
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
		return "node"
	}
	return h
}

// Package nodeagent manages vLLM server processes on one GPU.
//
// Kubernetes hands a pod exclusive use of a GPU for the pod's entire lifetime, so
// one-pod-per-model would pin exactly one model to each GPU forever. Kiln instead runs
// a single node-agent pod per GPU which spawns vLLM as child processes, which is what
// makes it possible to share a device between several models and to page them in and
// out. The same binary runs standalone under plain Docker so it works on Modal, where
// there is no Kubernetes at all.
package nodeagent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/sairavig/kiln/internal/scheduler"
)

// Spec describes one vLLM replica to run.
type Spec struct {
	// Replica is the unique id for this process, e.g. "qwen-0.5b#0".
	Replica string `json:"replica"`
	// Model is the id clients use in the OpenAI API; it is also what is passed to
	// vLLM as --served-model-name, so a replica answers to the same name whatever
	// Hugging Face repo backs it.
	Model string `json:"model"`
	// HFRepo and Revision identify the weights.
	HFRepo   string `json:"hf_repo"`
	Revision string `json:"revision,omitempty"`

	GPUMemoryUtilization float64 `json:"gpu_memory_utilization"`
	MaxModelLen          int     `json:"max_model_len"`
	// EnforceEager skips CUDA graph capture. It costs throughput but removes ~20s
	// from a cold start, and graph memory from the sleep/wake path.
	EnforceEager bool `json:"enforce_eager"`
	// EnablePrefixCaching is on by default in vLLM; the prefix-routing experiment
	// needs it on and the control arm needs to be able to turn it off.
	DisablePrefixCaching bool `json:"disable_prefix_caching"`
	// ExtraArgs is passed through verbatim for benchmark-specific flags.
	ExtraArgs []string `json:"extra_args,omitempty"`
}

// Status is the externally visible state of one replica.
type Status struct {
	Spec     Spec            `json:"spec"`
	State    string          `json:"state"`
	Port     int             `json:"port"`
	PID      int             `json:"pid,omitempty"`
	BaseURL  string          `json:"base_url"`
	Err      string          `json:"error,omitempty"`
	Restarts int             `json:"restarts"`
	// Timings measured on this node, fed back to the scheduler so cost-aware
	// eviction scores against this GPU's real numbers rather than an estimate.
	ColdStartMS    int64 `json:"cold_start_ms,omitempty"`
	WakeHostMS     int64 `json:"wake_host_ms,omitempty"`
	WakeDiskMS     int64 `json:"wake_disk_ms,omitempty"`
	VRAMBytes      int64 `json:"vram_bytes,omitempty"`
	LastTransition time.Time `json:"last_transition"`
}

// Config configures the supervisor.
type Config struct {
	// PythonBin is the interpreter that has vLLM installed.
	PythonBin string
	// PortBase is the first port handed to a replica; each gets the next free one.
	PortBase int
	// LogDir receives one log file per replica. vLLM's startup errors are the only
	// diagnostic when a load fails, so they are kept rather than discarded.
	LogDir string
	// StartTimeout bounds a cold start.
	StartTimeout time.Duration
	// HFHome is the shared weight cache. Pointing every replica at one cache is what
	// makes a level-2 wake cheap: the weights are re-read from the host page cache
	// rather than from the network.
	HFHome string
	// Env is extra environment for every child process.
	Env []string

	// AdvertiseHost is the host other components should use to reach this node's
	// replicas. It defaults to 127.0.0.1, which is right for a single-host dev setup
	// and wrong everywhere else: under Kubernetes the gateway is a different pod, so
	// the agent must advertise its pod IP. Set from the downward API in the manifest.
	AdvertiseHost string
}

// DefaultConfig returns workable defaults for local development.
func DefaultConfig() Config {
	return Config{
		PythonBin:     "python",
		PortBase:      19000,
		LogDir:        "/tmp/kiln-logs",
		StartTimeout:  15 * time.Minute,
		HFHome:        os.Getenv("HF_HOME"),
		AdvertiseHost: "127.0.0.1",
	}
}

// Replica is one supervised vLLM process.
type Replica struct {
	mu   sync.Mutex
	spec Spec
	cfg  Config

	cmd     *exec.Cmd
	// done is closed once the child has been reaped. Liveness cannot be tested with
	// signal 0: an unreaped child becomes a zombie, and a zombie still accepts
	// signals, so signal 0 reports a dead process as alive forever.
	done    chan struct{}
	port    int
	state   scheduler.State
	err     error
	restarts int
	logFile *os.File

	coldStart time.Duration
	wakeHost  time.Duration
	wakeDisk  time.Duration
	vramBytes int64
	lastTrans time.Time

	client *http.Client
}

func newReplica(spec Spec, cfg Config, port int) *Replica {
	return &Replica{
		spec: spec, cfg: cfg, port: port,
		state:     scheduler.StateCold,
		lastTrans: time.Now(),
		// No global timeout: a streaming completion legitimately runs for minutes.
		// Per-request timeouts are applied at the call site instead.
		client: &http.Client{},
	}
}

// BaseURL is where other components should send requests for this replica.
//
// Built from AdvertiseHost rather than hard-coded to loopback: the gateway may be on
// another host entirely, and an advertised 127.0.0.1 silently points it at itself.
func (r *Replica) BaseURL() string {
	host := r.cfg.AdvertiseHost
	if host == "" {
		host = "127.0.0.1"
	}
	return fmt.Sprintf("http://%s:%d", host, r.port)
}

// localURL is the address the agent itself uses to drive vLLM. Always loopback: the
// process is this agent's own child, so routing that traffic through the advertised
// address would leave the node and come back for no reason.
func (r *Replica) localURL() string { return fmt.Sprintf("http://127.0.0.1:%d", r.port) }

// State returns the current lifecycle state.
func (r *Replica) State() scheduler.State {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.state
}

func (r *Replica) setState(s scheduler.State) {
	r.state = s
	r.lastTrans = time.Now()
}

// Status snapshots the replica.
func (r *Replica) Status() Status {
	r.mu.Lock()
	defer r.mu.Unlock()
	st := Status{
		Spec: r.spec, State: r.state.String(), Port: r.port,
		BaseURL: r.BaseURL(), Restarts: r.restarts,
		ColdStartMS: r.coldStart.Milliseconds(),
		WakeHostMS:  r.wakeHost.Milliseconds(),
		WakeDiskMS:  r.wakeDisk.Milliseconds(),
		VRAMBytes:   r.vramBytes, LastTransition: r.lastTrans,
	}
	if r.cmd != nil && r.cmd.Process != nil {
		st.PID = r.cmd.Process.Pid
	}
	if r.err != nil {
		st.Err = r.err.Error()
	}
	return st
}

// args builds the vLLM command line.
func (r *Replica) args() []string {
	model := r.spec.HFRepo
	if model == "" {
		model = r.spec.Model
	}
	a := []string{
		"-m", "vllm.entrypoints.openai.api_server",
		"--model", model,
		// Clients address the model by its Kiln name, not by the HF repo path, so a
		// deployment can be repointed at a different repo without breaking callers.
		"--served-model-name", r.spec.Model,
		"--port", strconv.Itoa(r.port),
		// Sleep mode has to be requested at startup; it cannot be enabled later. Every
		// replica gets it, because a model that cannot be slept cannot be tiered and
		// would have to be killed instead.
		"--enable-sleep-mode",
		"--gpu-memory-utilization", strconv.FormatFloat(r.spec.GPUMemoryUtilization, 'f', 3, 64),
		// 0.32.0 removed --disable-log-requests in favour of this negated pair.
		"--no-enable-log-requests",
	}
	if r.spec.Revision != "" {
		a = append(a, "--revision", r.spec.Revision)
	}
	if r.spec.MaxModelLen > 0 {
		a = append(a, "--max-model-len", strconv.Itoa(r.spec.MaxModelLen))
	}
	if r.spec.EnforceEager {
		a = append(a, "--enforce-eager")
	}
	if r.spec.DisablePrefixCaching {
		a = append(a, "--no-enable-prefix-caching")
	}
	return append(a, r.spec.ExtraArgs...)
}

func (r *Replica) env() []string {
	env := append(os.Environ(),
		// Gates /sleep, /wake_up, /is_sleeping and /collective_rpc. Without it the
		// endpoints 404 and tiering is impossible over HTTP.
		"VLLM_SERVER_DEV_MODE=1",
		"VLLM_LOGGING_LEVEL=INFO",
	)
	if r.cfg.HFHome != "" {
		env = append(env, "HF_HOME="+r.cfg.HFHome)
	}
	return append(env, r.cfg.Env...)
}

// Start cold-starts the replica and blocks until it serves a real completion.
func (r *Replica) Start(ctx context.Context) error {
	r.mu.Lock()
	if r.state != scheduler.StateCold && r.state != scheduler.StateFailed {
		st := r.state
		r.mu.Unlock()
		return fmt.Errorf("nodeagent: cannot start replica in state %s", st)
	}
	r.setState(scheduler.StateStarting)
	r.mu.Unlock()

	start := time.Now()
	if err := r.spawn(); err != nil {
		r.fail(err)
		return err
	}
	if err := r.awaitReady(ctx, r.cfg.StartTimeout); err != nil {
		r.kill()
		r.fail(err)
		return err
	}

	r.mu.Lock()
	r.coldStart = time.Since(start)
	r.setState(scheduler.StateActive)
	r.err = nil
	r.mu.Unlock()
	return nil
}

func (r *Replica) spawn() error {
	if err := os.MkdirAll(r.cfg.LogDir, 0o755); err != nil {
		return fmt.Errorf("nodeagent: log dir: %w", err)
	}
	logPath := filepath.Join(r.cfg.LogDir, sanitize(r.spec.Replica)+".log")
	f, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return fmt.Errorf("nodeagent: open log: %w", err)
	}

	cmd := exec.Command(r.cfg.PythonBin, r.args()...)
	cmd.Env = r.env()
	cmd.Stdout, cmd.Stderr = f, f
	// Own process group: vLLM spawns an engine-core child, and killing only the
	// parent leaves that child holding the GPU, which then refuses the next model.
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	if err := cmd.Start(); err != nil {
		f.Close()
		return fmt.Errorf("nodeagent: spawn vllm: %w", err)
	}

	done := make(chan struct{})
	r.mu.Lock()
	r.cmd, r.logFile, r.done = cmd, f, done
	r.mu.Unlock()

	// Reap the child as soon as it exits and signal that through `done`. This is what
	// lets a failed start -- a gated Hugging Face repo, a CUDA mismatch, bad flags --
	// surface in seconds instead of after the full start timeout.
	go func() {
		_ = cmd.Wait()
		close(done)
	}()
	return nil
}

// awaitReady polls until the server answers a real one-token completion. Health alone
// is not enough: vLLM reports healthy before the model is loaded on some paths, and a
// replica the gateway routes to must actually be able to generate.
func (r *Replica) awaitReady(ctx context.Context, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	var last error
	for time.Now().Before(deadline) {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		if r.exited() {
			return fmt.Errorf("nodeagent: vllm exited during startup (%s); see %s",
				r.exitReason(), filepath.Join(r.cfg.LogDir, sanitize(r.spec.Replica)+".log"))
		}
		if err := r.probe(ctx); err == nil {
			return nil
		} else {
			last = err
		}
		time.Sleep(500 * time.Millisecond)
	}
	return fmt.Errorf("nodeagent: replica %s not ready within %s (last: %v)", r.spec.Replica, timeout, last)
}

// probe issues a single greedy token, which is the cheapest end-to-end proof the
// model is loaded and generating.
func (r *Replica) probe(ctx context.Context) error {
	body, _ := json.Marshal(map[string]any{
		"model": r.spec.Model, "prompt": "ok", "max_tokens": 1, "temperature": 0.0,
	})
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, r.localURL()+"/v1/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp, err := r.client.Do(req)
	if err != nil {
		return err
	}
	defer func() { io.Copy(io.Discard, resp.Body); resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("probe returned %s", resp.Status)
	}
	return nil
}

// Sleep offloads the replica. level 1 parks the weights in pinned host memory; level
// 2 discards them.
//
// mode is passed to vLLM: "wait" lets in-flight requests finish, "abort" cancels
// them. The scheduler drains before calling this, so "wait" here is a second line of
// defence against a request that arrived in the gap between the drain completing and
// the sleep being issued.
func (r *Replica) Sleep(ctx context.Context, level int, mode string) error {
	if level != 1 && level != 2 {
		return fmt.Errorf("nodeagent: sleep level must be 1 or 2, got %d", level)
	}
	if mode == "" {
		mode = "wait"
	}
	r.mu.Lock()
	if r.state != scheduler.StateActive && r.state != scheduler.StateDraining {
		st := r.state
		r.mu.Unlock()
		return fmt.Errorf("nodeagent: cannot sleep replica in state %s", st)
	}
	r.mu.Unlock()

	u := fmt.Sprintf("%s/sleep?level=%d&mode=%s", r.localURL(), level, mode)
	if err := r.post(ctx, u, nil, 10*time.Minute); err != nil {
		r.fail(err)
		return err
	}

	r.mu.Lock()
	if level == 1 {
		r.setState(scheduler.StateSleepingL1)
	} else {
		r.setState(scheduler.StateSleepingL2)
	}
	r.mu.Unlock()
	return nil
}

// Wake restores a sleeping replica to a serving state.
//
// The level-2 path is the subtle one. vLLM's level-2 sleep *discards* the weights,
// and /wake_up restores only the allocator and the model's saved buffers — see
// GPUWorker.wake_up, which replays _sleep_saved_buffers and nothing else. Waking from
// level 2 without reloading therefore returns 200 in milliseconds and then generates
// garbage. M0 measured exactly that. The weights must be re-read explicitly via
// collective_rpc, and the probe afterwards is what stops a silently-wrong replica
// from being routed to.
func (r *Replica) Wake(ctx context.Context) error {
	r.mu.Lock()
	from := r.state
	if from != scheduler.StateSleepingL1 && from != scheduler.StateSleepingL2 {
		r.mu.Unlock()
		return fmt.Errorf("nodeagent: cannot wake replica in state %s", from)
	}
	r.setState(scheduler.StateWaking)
	r.mu.Unlock()

	start := time.Now()
	if err := r.post(ctx, r.localURL()+"/wake_up", nil, 10*time.Minute); err != nil {
		r.fail(err)
		return err
	}

	if from == scheduler.StateSleepingL2 {
		body, _ := json.Marshal(map[string]any{"method": "reload_weights"})
		if err := r.post(ctx, r.localURL()+"/collective_rpc", body, 15*time.Minute); err != nil {
			r.fail(fmt.Errorf("reload_weights after level-2 wake: %w", err))
			return err
		}
	}

	// Generate before declaring the replica servable. A wake that restored memory but
	// not weights looks healthy and produces nonsense.
	if err := r.probe(ctx); err != nil {
		r.fail(fmt.Errorf("replica woke but cannot generate: %w", err))
		return err
	}

	r.mu.Lock()
	if from == scheduler.StateSleepingL1 {
		r.wakeHost = time.Since(start)
	} else {
		r.wakeDisk = time.Since(start)
	}
	r.setState(scheduler.StateActive)
	r.err = nil
	r.mu.Unlock()
	return nil
}

// Stop terminates the process and releases its GPU memory entirely.
func (r *Replica) Stop(ctx context.Context) error {
	r.kill()
	r.mu.Lock()
	r.setState(scheduler.StateCold)
	r.mu.Unlock()
	return nil
}

func (r *Replica) kill() {
	r.mu.Lock()
	cmd, f := r.cmd, r.logFile
	r.cmd, r.logFile = nil, nil
	r.mu.Unlock()
	if cmd == nil || cmd.Process == nil {
		return
	}
	pgid, err := syscall.Getpgid(cmd.Process.Pid)
	if err != nil {
		pgid = cmd.Process.Pid
	}
	// Signal the whole group: the engine-core child holds the GPU allocation, and
	// killing only the parent leaves it resident and the device unusable.
	_ = syscall.Kill(-pgid, syscall.SIGTERM)

	done := make(chan struct{})
	go func() { _ = cmd.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(45 * time.Second):
		_ = syscall.Kill(-pgid, syscall.SIGKILL)
		<-done
	}
	if f != nil {
		_ = f.Close()
	}
}

func (r *Replica) exited() bool {
	r.mu.Lock()
	done := r.done
	cmd := r.cmd
	r.mu.Unlock()
	if cmd == nil || cmd.Process == nil {
		return true
	}
	if done == nil {
		return false
	}
	select {
	case <-done:
		return true
	default:
		return false
	}
}

// exitReason summarises why the child stopped, and pulls the last error-looking line
// out of its log so the deploy job's error field says something an operator can act
// on rather than just "exited".
func (r *Replica) exitReason() string {
	r.mu.Lock()
	cmd := r.cmd
	r.mu.Unlock()

	reason := "unknown"
	if cmd != nil && cmd.ProcessState != nil {
		reason = fmt.Sprintf("exit status %d", cmd.ProcessState.ExitCode())
	}
	if hint := lastErrorLine(filepath.Join(r.cfg.LogDir, sanitize(r.spec.Replica)+".log")); hint != "" {
		reason += ": " + hint
	}
	return reason
}

// lastErrorLine returns the final line of a log that looks like a Python exception.
func lastErrorLine(path string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	// Only the tail matters, and a vLLM startup log can be megabytes.
	const window = 32 << 10
	if len(data) > window {
		data = data[len(data)-window:]
	}
	lines := strings.Split(string(data), "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		ln := strings.TrimSpace(lines[i])
		// A Python traceback's final line is "SomeError: message", which is the one
		// sentence worth surfacing.
		if m := errLineRe.FindString(ln); m != "" && len(ln) < 400 {
			return ln
		}
	}
	return ""
}

var errLineRe = regexp.MustCompile(`^[A-Za-z_.]*(Error|Exception)[: ]`)

func (r *Replica) fail(err error) {
	r.mu.Lock()
	r.err = err
	r.setState(scheduler.StateFailed)
	r.mu.Unlock()
}

func (r *Replica) post(ctx context.Context, url string, body []byte, timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	var rdr io.Reader
	if body != nil {
		rdr = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, rdr)
	if err != nil {
		return err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := r.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return fmt.Errorf("%s -> %s: %s", url, resp.Status, strings.TrimSpace(string(msg)))
	}
	io.Copy(io.Discard, resp.Body)
	return nil
}

func sanitize(s string) string {
	return strings.NewReplacer("/", "_", "#", "-", ":", "-", " ", "_").Replace(s)
}

// freePort asks the kernel for an unused port. Racy in principle, but the alternative
// — a fixed range — collides much more often in practice when replicas churn.
func freePort(base int) (int, error) {
	for p := base; p < base+2000; p++ {
		l, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", p))
		if err == nil {
			l.Close()
			return p, nil
		}
	}
	return 0, fmt.Errorf("nodeagent: no free port in [%d, %d)", base, base+2000)
}

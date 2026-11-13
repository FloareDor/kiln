package jobs

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"sort"
	"strings"
	"time"

	"github.com/sairavig/kiln/internal/agentclient"
	"github.com/sairavig/kiln/internal/nodeagent"
	"github.com/sairavig/kiln/internal/store"
)

// Agent is one node agent the deployer can place replicas on.
type Agent struct {
	Node   string
	URL    string
	Client *agentclient.Client
}

// Deployer executes deploy, scale and delete jobs against node agents.
type Deployer struct {
	Store  *store.Store
	Agents []Agent
	Log    *slog.Logger
	// MeasureWakes controls whether a deploy also measures sleep/wake costs.
	//
	// It is on for the swapping experiments and off for plain serving, because the
	// measurement costs a sleep and two wakes per replica — a few seconds that are
	// pure overhead if the model will never be tiered. The cost-aware policy needs
	// these numbers, so the alternative is to guess them, which would make its
	// scoring confidently wrong.
	MeasureWakes bool
}

// NewDeployer builds a deployer over agent base URLs.
func NewDeployer(s *store.Store, agentURLs []string, log *slog.Logger, measureWakes bool) *Deployer {
	d := &Deployer{Store: s, Log: log, MeasureWakes: measureWakes}
	for _, u := range agentURLs {
		u = strings.TrimSpace(u)
		if u == "" {
			continue
		}
		d.Agents = append(d.Agents, Agent{URL: u, Client: agentclient.New(u)})
	}
	return d
}

// ReplicaID is the stable name for the n-th replica of a model.
//
// `--` rather than `#` or `/`: replica ids are interpolated into node-agent URL paths,
// and `#` begins a fragment, so `POST /replicas/model#0/sleep` arrives at the server
// as `POST /replicas/model`. That cost an afternoon during M1.
func ReplicaID(modelID string, n int) string {
	return fmt.Sprintf("%s--%d", modelID, n)
}

// Deploy brings a model's replicas up and records what they cost.
func (d *Deployer) Deploy(ctx context.Context, m *store.Model, _ json.RawMessage) error {
	if len(d.Agents) == 0 {
		return fmt.Errorf("jobs: no node agents configured")
	}

	placements, err := d.place(ctx, m, m.Replicas)
	if err != nil {
		return err
	}

	var meas store.Measurements
	for i, a := range placements {
		id := ReplicaID(m.ID, i)
		spec := nodeagent.Spec{
			Replica: id, Model: m.ID, HFRepo: m.HFRepo, Revision: m.Revision,
			GPUMemoryUtilization: m.GPUMemoryUtilization,
			MaxModelLen:          m.MaxModelLen,
			EnforceEager:         m.EnforceEager,
		}

		// Record the replica as STARTING before the call, not after. A deploy that
		// crashes mid-start would otherwise leave a vLLM process holding GPU memory
		// that no database row accounts for, and the scheduler would place against
		// memory it believes is free.
		if err := d.Store.UpsertReplica(ctx, &store.Replica{
			ID: id, ModelID: m.ID, Node: a.Node, AgentURL: a.URL, State: "STARTING",
		}); err != nil {
			return err
		}

		st, err := a.Client.Start(ctx, spec)
		if err != nil {
			_ = d.Store.UpsertReplica(ctx, &store.Replica{
				ID: id, ModelID: m.ID, Node: a.Node, AgentURL: a.URL, State: "FAILED",
			})
			return fmt.Errorf("start %s on %s: %w", id, a.URL, err)
		}

		if err := d.Store.UpsertReplica(ctx, &store.Replica{
			ID: id, ModelID: m.ID, Node: st.Spec.Replica, AgentURL: a.URL,
			Port: st.Port, State: st.State,
		}); err != nil {
			return err
		}
		d.Log.Info("replica started", "replica", id, "agent", a.URL,
			"cold_start_ms", st.ColdStartMS, "port", st.Port)

		// Measure on the first replica only. The others are identical by
		// construction, and each measurement costs real GPU time.
		if i == 0 {
			if st.ColdStartMS > 0 {
				meas.ColdStartMS = st.ColdStartMS
			}
			if vram := d.measureFootprint(ctx, a, m); vram > 0 {
				meas.FootprintVRAMBytes = vram
			}
			if d.MeasureWakes {
				host, disk, hostBytes := d.measureWakes(ctx, a, id)
				meas.WakeHostMS, meas.WakeDiskMS = host, disk
				meas.FootprintHostBytes = hostBytes
			}
		}
	}

	if err := d.Store.RecordMeasurements(ctx, m.ID, meas); err != nil {
		return err
	}
	d.Log.Info("deploy complete", "model", m.ID, "replicas", len(placements),
		"vram_bytes", meas.FootprintVRAMBytes, "cold_start_ms", meas.ColdStartMS,
		"wake_host_ms", meas.WakeHostMS, "wake_disk_ms", meas.WakeDiskMS)
	return nil
}

// measureFootprint reads how much VRAM the model actually took.
//
// Measured as the device delta across the start rather than estimated from parameter
// count, because the KV cache arena vLLM reserves from --gpu-memory-utilization
// usually dominates the weights and depends entirely on the flags we chose. The
// scheduler budgets against this number, so an estimate here produces
// over-subscription and an OOM at the worst moment.
func (d *Deployer) measureFootprint(ctx context.Context, a Agent, m *store.Model) int64 {
	st, err := a.Client.Status(ctx)
	if err != nil || len(st.GPUs) == 0 {
		d.Log.Warn("could not measure footprint", "model", m.ID, "err", err)
		return 0
	}
	// With one model on the device, used-minus-idle is the model. With several, this
	// over-attributes, which is why the deploy job runs against a quiet GPU and why
	// gpu_memory_utilization is the authoritative upper bound.
	used := st.GPUs[0].UsedBytes
	byUtil := int64(float64(st.GPUs[0].TotalBytes) * m.GPUMemoryUtilization)
	if used > byUtil {
		return byUtil
	}
	return used
}

// measureWakes times a sleep/wake round trip at both tiers.
//
// The level-2 wake is measured through the node agent, which performs the
// reload_weights step vLLM's own /wake_up omits. Timing a bare /wake_up would record
// a wake four times faster than reality that produces garbage — see M0.
func (d *Deployer) measureWakes(ctx context.Context, a Agent, replicaID string) (hostMS, diskMS, hostBytes int64) {
	before, _ := a.Client.Status(ctx)

	if _, err := a.Client.Sleep(ctx, replicaID, 1, "wait"); err != nil {
		d.Log.Warn("sleep L1 measurement failed", "replica", replicaID, "err", err)
		return 0, 0, 0
	}
	if after, err := a.Client.Status(ctx); err == nil && len(before.GPUs) > 0 && len(after.GPUs) > 0 {
		// Host-tier cost is approximated by the VRAM the offload released; vLLM
		// copies roughly the weights across, excluding the KV arena it drops.
		if freed := before.GPUs[0].UsedBytes - after.GPUs[0].UsedBytes; freed > 0 {
			hostBytes = freed
		}
	}
	t0 := time.Now()
	if _, err := a.Client.Wake(ctx, replicaID); err != nil {
		d.Log.Warn("wake from L1 failed", "replica", replicaID, "err", err)
		return 0, 0, hostBytes
	}
	hostMS = time.Since(t0).Milliseconds()

	if _, err := a.Client.Sleep(ctx, replicaID, 2, "wait"); err != nil {
		d.Log.Warn("sleep L2 measurement failed", "replica", replicaID, "err", err)
		return hostMS, 0, hostBytes
	}
	t0 = time.Now()
	if _, err := a.Client.Wake(ctx, replicaID); err != nil {
		d.Log.Warn("wake from L2 failed", "replica", replicaID, "err", err)
		return hostMS, 0, hostBytes
	}
	diskMS = time.Since(t0).Milliseconds()
	return hostMS, diskMS, hostBytes
}

// place picks which agents host a model's replicas, most free VRAM first.
//
// Deliberately simple: with one GPU in the development setup, placement is not where
// the interesting problems are, and a sophisticated bin-packer here would be untested
// complexity. Spreading by free VRAM is the behaviour that matters — it keeps two
// replicas of one model off the same device when another is available, so a single
// GPU failure does not take the model down entirely.
func (d *Deployer) place(ctx context.Context, m *store.Model, n int) ([]Agent, error) {
	type scored struct {
		a    Agent
		free int64
	}
	var avail []scored
	for _, a := range d.Agents {
		st, err := a.Client.Status(ctx)
		if err != nil {
			d.Log.Warn("agent unreachable during placement", "agent", a.URL, "err", err)
			continue
		}
		a.Node = st.Node
		var free int64
		for _, g := range st.GPUs {
			free += g.FreeBytes
		}
		avail = append(avail, scored{a, free})
	}
	if len(avail) == 0 {
		return nil, fmt.Errorf("jobs: no reachable node agent for model %s", m.ID)
	}
	sort.SliceStable(avail, func(i, j int) bool {
		if avail[i].free != avail[j].free {
			return avail[i].free > avail[j].free
		}
		return avail[i].a.URL < avail[j].a.URL
	})

	out := make([]Agent, 0, n)
	for i := 0; i < n; i++ {
		out = append(out, avail[i%len(avail)].a)
	}
	return out, nil
}

// Scale changes a model's replica count.
func (d *Deployer) Scale(ctx context.Context, m *store.Model, payload json.RawMessage) error {
	var p struct {
		Replicas int `json:"replicas"`
	}
	if len(payload) > 0 {
		if err := json.Unmarshal(payload, &p); err != nil {
			return fmt.Errorf("jobs: bad scale payload: %w", err)
		}
	}
	if p.Replicas <= 0 {
		p.Replicas = m.Replicas
	}

	existing, err := d.Store.Replicas(ctx, m.ID)
	if err != nil {
		return err
	}

	// Scaling down: remove the highest-numbered replicas, so the surviving set stays
	// a stable prefix. Removing arbitrary ones would churn the prefix router's
	// affinity table on every scale event and throw away cache locality for no gain.
	if p.Replicas < len(existing) {
		for i := len(existing) - 1; i >= p.Replicas; i-- {
			id := ReplicaID(m.ID, i)
			if a, ok := d.agentFor(existing, id); ok {
				if err := a.Stop(ctx, id); err != nil {
					d.Log.Warn("stop during scale-down failed", "replica", id, "err", err)
				}
			}
			if err := d.Store.DeleteReplica(ctx, id); err != nil {
				return err
			}
		}
		m.Replicas = p.Replicas
		_, err = d.Store.UpsertModel(ctx, m)
		return err
	}

	m.Replicas = p.Replicas
	if _, err := d.Store.UpsertModel(ctx, m); err != nil {
		return err
	}
	// Deploy is idempotent on the agent side: starting an already-ACTIVE replica
	// returns its status without touching the GPU, so scaling up is just a redeploy.
	return d.Deploy(ctx, m, nil)
}

// Delete stops every replica of a model.
func (d *Deployer) Delete(ctx context.Context, m *store.Model, _ json.RawMessage) error {
	existing, err := d.Store.Replicas(ctx, m.ID)
	if err != nil {
		return err
	}
	for _, r := range existing {
		if a, ok := d.agentFor(existing, r.ID); ok {
			// Carry on past a failed stop. A node that is already gone cannot be
			// asked to stop anything, and refusing to delete the model would leave
			// an undeletable row pointing at a dead node forever.
			if err := a.Stop(ctx, r.ID); err != nil {
				d.Log.Warn("stop during delete failed", "replica", r.ID, "err", err)
			}
		}
		if err := d.Store.DeleteReplica(ctx, r.ID); err != nil {
			return err
		}
	}
	return nil
}

func (d *Deployer) agentFor(reps []*store.Replica, id string) (*agentclient.Client, bool) {
	for _, r := range reps {
		if r.ID == id {
			for _, a := range d.Agents {
				if a.URL == r.AgentURL {
					return a.Client, true
				}
			}
			// The replica names an agent that is no longer configured. Fall back to a
			// fresh client rather than silently skipping the stop and orphaning a
			// vLLM process that still holds the GPU.
			return agentclient.New(r.AgentURL), true
		}
	}
	return nil, false
}

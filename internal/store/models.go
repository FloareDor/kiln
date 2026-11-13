package store

import (
	"context"
	"time"
)

// Model is a deployment's registry row.
type Model struct {
	ID                   string  `json:"id"`
	HFRepo               string  `json:"hf_repo"`
	Revision             string  `json:"revision"`
	Replicas             int     `json:"replicas"`
	GPUMemoryUtilization float64 `json:"gpu_memory_utilization"`
	MaxModelLen          int     `json:"max_model_len"`
	EnforceEager         bool    `json:"enforce_eager"`

	// Measured during the deploy job rather than estimated from parameter count.
	// The cost-aware eviction policy scores against these, so they must be this
	// GPU's real numbers; a guess here produces confidently wrong evictions.
	FootprintVRAMBytes int64 `json:"footprint_vram_bytes"`
	FootprintHostBytes int64 `json:"footprint_host_bytes"`
	ColdStartMS        int64 `json:"cold_start_ms"`
	WakeHostMS         int64 `json:"wake_host_ms"`
	WakeDiskMS         int64 `json:"wake_disk_ms"`

	Pinned    bool      `json:"pinned"`
	Status    string    `json:"status"`
	Error     string    `json:"error,omitempty"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

// Model status values.
const (
	ModelPending  = "PENDING"
	ModelDeploying = "DEPLOYING"
	ModelReady    = "READY"
	ModelFailed   = "FAILED"
	ModelDeleting = "DELETING"
)

const modelCols = `id, hf_repo, revision, replicas, gpu_memory_utilization, max_model_len,
	enforce_eager, footprint_vram_bytes, footprint_host_bytes, cold_start_ms,
	wake_host_ms, wake_disk_ms, pinned, status, error, created_at, updated_at`

type scanner interface{ Scan(dest ...any) error }

func scanModel(row scanner) (*Model, error) {
	var m Model
	err := row.Scan(&m.ID, &m.HFRepo, &m.Revision, &m.Replicas, &m.GPUMemoryUtilization,
		&m.MaxModelLen, &m.EnforceEager, &m.FootprintVRAMBytes, &m.FootprintHostBytes,
		&m.ColdStartMS, &m.WakeHostMS, &m.WakeDiskMS, &m.Pinned, &m.Status, &m.Error,
		&m.CreatedAt, &m.UpdatedAt)
	if err != nil {
		return nil, wrap(err)
	}
	return &m, nil
}

// UpsertModel creates or updates a deployment row.
//
// Upsert rather than insert: POST /admin/models with the same id is how a caller
// changes a deployment's shape (more replicas, a different revision), and making that
// a conflict would force every client to implement create-or-update itself.
func (s *Store) UpsertModel(ctx context.Context, m *Model) (*Model, error) {
	row := s.Pool.QueryRow(ctx,
		`INSERT INTO models (id, hf_repo, revision, replicas, gpu_memory_utilization,
		                     max_model_len, enforce_eager, pinned, status)
		 VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
		 ON CONFLICT (id) DO UPDATE SET
		   hf_repo = EXCLUDED.hf_repo,
		   revision = EXCLUDED.revision,
		   replicas = EXCLUDED.replicas,
		   gpu_memory_utilization = EXCLUDED.gpu_memory_utilization,
		   max_model_len = EXCLUDED.max_model_len,
		   enforce_eager = EXCLUDED.enforce_eager,
		   pinned = EXCLUDED.pinned,
		   status = EXCLUDED.status,
		   updated_at = now()
		 RETURNING `+modelCols,
		m.ID, m.HFRepo, m.Revision, m.Replicas, m.GPUMemoryUtilization,
		m.MaxModelLen, m.EnforceEager, m.Pinned, orDefault(m.Status, ModelPending))
	return scanModel(row)
}

// Model loads one deployment.
func (s *Store) Model(ctx context.Context, id string) (*Model, error) {
	return scanModel(s.Pool.QueryRow(ctx, `SELECT `+modelCols+` FROM models WHERE id = $1`, id))
}

// Models lists deployments.
func (s *Store) Models(ctx context.Context) ([]*Model, error) {
	rows, err := s.Pool.Query(ctx, `SELECT `+modelCols+` FROM models ORDER BY id`)
	if err != nil {
		return nil, wrap(err)
	}
	defer rows.Close()
	out := []*Model{}
	for rows.Next() {
		m, err := scanModel(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, m)
	}
	return out, rows.Err()
}

// SetModelStatus updates status and error text.
func (s *Store) SetModelStatus(ctx context.Context, id, status, errMsg string) error {
	_, err := s.Pool.Exec(ctx,
		`UPDATE models SET status = $2, error = $3, updated_at = now() WHERE id = $1`,
		id, status, errMsg)
	return wrap(err)
}

// Measurements are the per-model costs a deploy job measures and the scheduler scores
// against.
type Measurements struct {
	FootprintVRAMBytes int64
	FootprintHostBytes int64
	ColdStartMS        int64
	WakeHostMS         int64
	WakeDiskMS         int64
}

// RecordMeasurements stores what a deploy job measured.
//
// COALESCE-style guards: a zero means "not measured this time", not "measured as
// zero". A redeploy that only cold-starts should not wipe the wake costs an earlier
// run established, because the cost-aware policy would then score every model's wake
// as free and stop distinguishing them at all.
func (s *Store) RecordMeasurements(ctx context.Context, id string, m Measurements) error {
	_, err := s.Pool.Exec(ctx,
		// The ::bigint casts are load-bearing. Inside a CASE expression pgx has
		// nothing to infer the parameter type from and falls back to int4, so any
		// footprint above 2 GiB -- which is every model worth serving -- fails to
		// encode at runtime.
		`UPDATE models SET
		   footprint_vram_bytes = CASE WHEN $2::bigint > 0 THEN $2::bigint ELSE footprint_vram_bytes END,
		   footprint_host_bytes = CASE WHEN $3::bigint > 0 THEN $3::bigint ELSE footprint_host_bytes END,
		   cold_start_ms        = CASE WHEN $4::bigint > 0 THEN $4::bigint ELSE cold_start_ms END,
		   wake_host_ms         = CASE WHEN $5::bigint > 0 THEN $5::bigint ELSE wake_host_ms END,
		   wake_disk_ms         = CASE WHEN $6::bigint > 0 THEN $6::bigint ELSE wake_disk_ms END,
		   updated_at = now()
		 WHERE id = $1`,
		id, m.FootprintVRAMBytes, m.FootprintHostBytes, m.ColdStartMS, m.WakeHostMS, m.WakeDiskMS)
	return wrap(err)
}

// SetPinned marks a model never-evictable.
func (s *Store) SetPinned(ctx context.Context, id string, pinned bool) error {
	_, err := s.Pool.Exec(ctx,
		`UPDATE models SET pinned = $2, updated_at = now() WHERE id = $1`, id, pinned)
	return wrap(err)
}

// DeleteModel removes a deployment and, by cascade, its replicas and jobs.
func (s *Store) DeleteModel(ctx context.Context, id string) error {
	tag, err := s.Pool.Exec(ctx, `DELETE FROM models WHERE id = $1`, id)
	if err != nil {
		return wrap(err)
	}
	if tag.RowsAffected() == 0 {
		return ErrNotFound
	}
	return nil
}

// Replica is one running copy of a model.
type Replica struct {
	ID        string    `json:"id"`
	ModelID   string    `json:"model_id"`
	Node      string    `json:"node"`
	AgentURL  string    `json:"agent_url"`
	Port      int       `json:"port"`
	State     string    `json:"state"`
	UpdatedAt time.Time `json:"updated_at"`
}

// UpsertReplica records a replica's placement and state.
func (s *Store) UpsertReplica(ctx context.Context, r *Replica) error {
	_, err := s.Pool.Exec(ctx,
		`INSERT INTO replicas (id, model_id, node, agent_url, port, state)
		 VALUES ($1,$2,$3,$4,$5,$6)
		 ON CONFLICT (id) DO UPDATE SET
		   node = EXCLUDED.node, agent_url = EXCLUDED.agent_url,
		   port = EXCLUDED.port, state = EXCLUDED.state, updated_at = now()`,
		r.ID, r.ModelID, r.Node, r.AgentURL, r.Port, r.State)
	return wrap(err)
}

// Replicas lists a model's replicas.
func (s *Store) Replicas(ctx context.Context, modelID string) ([]*Replica, error) {
	rows, err := s.Pool.Query(ctx,
		`SELECT id, model_id, node, agent_url, port, state, updated_at
		 FROM replicas WHERE model_id = $1 ORDER BY id`, modelID)
	if err != nil {
		return nil, wrap(err)
	}
	defer rows.Close()
	out := []*Replica{}
	for rows.Next() {
		var r Replica
		if err := rows.Scan(&r.ID, &r.ModelID, &r.Node, &r.AgentURL, &r.Port, &r.State, &r.UpdatedAt); err != nil {
			return nil, err
		}
		out = append(out, &r)
	}
	return out, rows.Err()
}

// AllReplicas lists every replica, used to rebuild the scheduler's cluster view.
func (s *Store) AllReplicas(ctx context.Context) ([]*Replica, error) {
	rows, err := s.Pool.Query(ctx,
		`SELECT id, model_id, node, agent_url, port, state, updated_at FROM replicas ORDER BY id`)
	if err != nil {
		return nil, wrap(err)
	}
	defer rows.Close()
	out := []*Replica{}
	for rows.Next() {
		var r Replica
		if err := rows.Scan(&r.ID, &r.ModelID, &r.Node, &r.AgentURL, &r.Port, &r.State, &r.UpdatedAt); err != nil {
			return nil, err
		}
		out = append(out, &r)
	}
	return out, rows.Err()
}

// DeleteReplica removes a replica row.
func (s *Store) DeleteReplica(ctx context.Context, id string) error {
	_, err := s.Pool.Exec(ctx, `DELETE FROM replicas WHERE id = $1`, id)
	return wrap(err)
}

func orDefault(s, def string) string {
	if s == "" {
		return def
	}
	return s
}

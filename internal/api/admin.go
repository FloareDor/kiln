package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"github.com/google/uuid"
	"github.com/sairavig/kiln/internal/auth"
	"github.com/sairavig/kiln/internal/store"
)

// Admin is the control plane's management API: deployments, teams, keys and usage.
type Admin struct {
	Store *store.Store
	Log   *slog.Logger
}

// NewAdmin builds the admin API.
func NewAdmin(s *store.Store, log *slog.Logger) *Admin { return &Admin{Store: s, Log: log} }

// Routes returns the admin surface, mounted under /admin.
func (a *Admin) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /admin/models", a.createModel)
	mux.HandleFunc("GET /admin/models", a.listModels)
	mux.HandleFunc("GET /admin/models/{id}", a.getModel)
	mux.HandleFunc("DELETE /admin/models/{id}", a.deleteModel)
	mux.HandleFunc("POST /admin/models/{id}/scale", a.scaleModel)
	mux.HandleFunc("POST /admin/models/{id}/pin", a.pinModel)

	mux.HandleFunc("POST /admin/teams", a.createTeam)
	mux.HandleFunc("GET /admin/teams", a.listTeams)
	mux.HandleFunc("POST /admin/teams/{id}/keys", a.createKey)
	mux.HandleFunc("DELETE /admin/keys/{id}", a.revokeKey)

	mux.HandleFunc("GET /admin/jobs", a.listJobs)
	mux.HandleFunc("GET /admin/usage", a.usage)
	return mux
}

// DeployRequest is the body of POST /admin/models — the "deploy a Hugging Face model
// with one API call" surface.
type DeployRequest struct {
	// ID is what clients will pass as `model`. Defaults to the repo's last segment.
	ID       string `json:"id"`
	HFRepo   string `json:"hf_repo"`
	Revision string `json:"revision,omitempty"`
	Replicas int    `json:"replicas,omitempty"`
	// GPUMemoryUtilization is the fraction of the device vLLM may claim. It is the
	// authoritative upper bound on the model's footprint and therefore on how many
	// models the scheduler believes fit.
	GPUMemoryUtilization float64 `json:"gpu_memory_utilization,omitempty"`
	MaxModelLen          int     `json:"max_model_len,omitempty"`
	EnforceEager         bool    `json:"enforce_eager,omitempty"`
	Pinned               bool    `json:"pinned,omitempty"`
}

// DeployResponse returns the model and the job that will bring it up.
type DeployResponse struct {
	Model *store.Model `json:"model"`
	Job   *store.Job   `json:"job"`
}

func (a *Admin) createModel(w http.ResponseWriter, r *http.Request) {
	var req DeployRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		adminErr(w, http.StatusBadRequest, "malformed JSON: "+err.Error())
		return
	}
	if req.HFRepo == "" {
		adminErr(w, http.StatusBadRequest, "hf_repo is required")
		return
	}
	if req.ID == "" {
		req.ID = lastSegment(req.HFRepo)
	}
	if req.Replicas <= 0 {
		req.Replicas = 1
	}
	if req.GPUMemoryUtilization <= 0 {
		req.GPUMemoryUtilization = 0.35
	}
	if req.MaxModelLen <= 0 {
		req.MaxModelLen = 4096
	}

	ctx := r.Context()

	// Check for an in-flight deploy BEFORE touching the model row.
	//
	// The obvious client behaviour after a timeout is to POST again, often with a
	// shorter body, so this endpoint has to be idempotent. The subtle part is that
	// idempotency must cover the *model row* too, not just the job: upserting first
	// and deduplicating the job afterwards means a bare retry silently rewrites the
	// deployment with this handler's defaults. That is not theoretical -- it cost a
	// debugging session when a retry reset enforce_eager to false and the running
	// deploy then tried to invoke a CUDA compiler that was not installed.
	if existing, err := a.Store.JobForModel(ctx, req.ID); err == nil {
		current, cerr := a.Store.Model(ctx, req.ID)
		if cerr == nil {
			a.Log.Info("deploy already in flight; returning the existing job",
				"model", req.ID, "job", existing.ID)
			writeJSON(w, http.StatusAccepted, DeployResponse{Model: current, Job: existing})
			return
		}
	}

	m, err := a.Store.UpsertModel(ctx, &store.Model{
		ID: req.ID, HFRepo: req.HFRepo, Revision: req.Revision, Replicas: req.Replicas,
		GPUMemoryUtilization: req.GPUMemoryUtilization, MaxModelLen: req.MaxModelLen,
		EnforceEager: req.EnforceEager, Pinned: req.Pinned, Status: store.ModelPending,
	})
	if err != nil {
		adminErr(w, http.StatusInternalServerError, err.Error())
		return
	}

	job, err := a.Store.EnqueueJob(ctx, uuid.NewString(), m.ID, store.JobDeploy, req)
	if errors.Is(err, store.ErrConflict) {
		// Lost the race with a concurrent POST between the check above and here.
		// The database's partial unique index is the real guard; this branch just
		// turns it into the same idempotent answer.
		if existing, lookupErr := a.Store.JobForModel(ctx, m.ID); lookupErr == nil {
			writeJSON(w, http.StatusAccepted, DeployResponse{Model: m, Job: existing})
			return
		}
	}
	if err != nil {
		adminErr(w, http.StatusConflict, err.Error())
		return
	}
	a.Log.Info("deploy requested", "model", m.ID, "repo", m.HFRepo, "replicas", m.Replicas, "job", job.ID)
	writeJSON(w, http.StatusAccepted, DeployResponse{Model: m, Job: job})
}

func (a *Admin) listModels(w http.ResponseWriter, r *http.Request) {
	models, err := a.Store.Models(r.Context())
	if err != nil {
		adminErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"models": models})
}

func (a *Admin) getModel(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	m, err := a.Store.Model(ctx, r.PathValue("id"))
	if errors.Is(err, store.ErrNotFound) {
		adminErr(w, http.StatusNotFound, "no such model")
		return
	}
	if err != nil {
		adminErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	reps, _ := a.Store.Replicas(ctx, m.ID)
	job, jobErr := a.Store.JobForModel(ctx, m.ID)
	out := map[string]any{"model": m, "replicas": reps}
	if jobErr == nil {
		out["job"] = job
	}
	writeJSON(w, http.StatusOK, out)
}

func (a *Admin) deleteModel(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	id := r.PathValue("id")
	if _, err := a.Store.Model(ctx, id); errors.Is(err, store.ErrNotFound) {
		adminErr(w, http.StatusNotFound, "no such model")
		return
	}
	// Delete through a job, not inline. The handler cannot stop vLLM processes on
	// remote nodes within one HTTP request, and a delete that removed the row without
	// stopping the processes would orphan them holding GPU memory nothing accounts
	// for.
	job, err := a.Store.EnqueueJob(ctx, uuid.NewString(), id, store.JobDelete, nil)
	if err != nil {
		adminErr(w, http.StatusConflict, err.Error())
		return
	}
	writeJSON(w, http.StatusAccepted, map[string]any{"job": job})
}

func (a *Admin) scaleModel(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Replicas int `json:"replicas"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		adminErr(w, http.StatusBadRequest, err.Error())
		return
	}
	if req.Replicas < 0 {
		adminErr(w, http.StatusBadRequest, "replicas must be >= 0")
		return
	}
	job, err := a.Store.EnqueueJob(r.Context(), uuid.NewString(), r.PathValue("id"), store.JobScale, req)
	if err != nil {
		adminErr(w, http.StatusConflict, err.Error())
		return
	}
	writeJSON(w, http.StatusAccepted, map[string]any{"job": job})
}

func (a *Admin) pinModel(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Pinned bool `json:"pinned"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		adminErr(w, http.StatusBadRequest, err.Error())
		return
	}
	if err := a.Store.SetPinned(r.Context(), r.PathValue("id"), req.Pinned); err != nil {
		adminErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"id": r.PathValue("id"), "pinned": req.Pinned})
}

func (a *Admin) createTeam(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ID             string `json:"id"`
		Name           string `json:"name"`
		RPM            int    `json:"rpm"`
		TokensPerDay   int64  `json:"tokens_per_day"`
		MaxConcurrency int    `json:"max_concurrency"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		adminErr(w, http.StatusBadRequest, err.Error())
		return
	}
	if req.ID == "" {
		adminErr(w, http.StatusBadRequest, "id is required")
		return
	}
	// Defaults rather than unlimited. A team created with a zero quota would be
	// treated as unthrottled by the limiter, which is the opposite of what someone
	// omitting the field expects.
	if req.RPM <= 0 {
		req.RPM = 60
	}
	if req.TokensPerDay <= 0 {
		req.TokensPerDay = 1_000_000
	}
	if req.MaxConcurrency <= 0 {
		req.MaxConcurrency = 8
	}
	if req.Name == "" {
		req.Name = req.ID
	}

	team, err := a.Store.CreateTeam(r.Context(), req.ID, req.Name, auth.Quota{
		RPM: req.RPM, TokensPerDay: req.TokensPerDay, MaxConcurrency: req.MaxConcurrency,
	})
	if errors.Is(err, store.ErrConflict) {
		adminErr(w, http.StatusConflict, "team already exists")
		return
	}
	if err != nil {
		adminErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusCreated, map[string]any{"team": team})
}

func (a *Admin) listTeams(w http.ResponseWriter, r *http.Request) {
	teams, err := a.Store.Teams(r.Context())
	if err != nil {
		adminErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	quotas, _ := a.Store.Quotas(r.Context())
	writeJSON(w, http.StatusOK, map[string]any{"teams": teams, "quotas": quotas})
}

func (a *Admin) createKey(w http.ResponseWriter, r *http.Request) {
	id := uuid.NewString()
	pt, err := a.Store.CreateAPIKey(r.Context(), id, r.PathValue("id"))
	if err != nil {
		adminErr(w, http.StatusBadRequest, err.Error())
		return
	}
	// The only time the plaintext exists outside the client's hands. It is not
	// stored and cannot be shown again.
	writeJSON(w, http.StatusCreated, map[string]any{
		"id": id, "team_id": r.PathValue("id"), "api_key": pt,
		"warning": "this key is shown once and is not recoverable",
	})
}

func (a *Admin) revokeKey(w http.ResponseWriter, r *http.Request) {
	if err := a.Store.RevokeKey(r.Context(), r.PathValue("id")); errors.Is(err, store.ErrNotFound) {
		adminErr(w, http.StatusNotFound, "no such key")
		return
	} else if err != nil {
		adminErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (a *Admin) listJobs(w http.ResponseWriter, r *http.Request) {
	jobs, err := a.Store.Jobs(r.Context(), 100)
	if err != nil {
		adminErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"jobs": jobs})
}

func (a *Admin) usage(w http.ResponseWriter, r *http.Request) {
	since := time.Now().Add(-24 * time.Hour)
	if v := r.URL.Query().Get("since"); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			since = time.Now().Add(-d)
		}
	}
	rows, err := a.Store.UsageBreakdown(r.Context(), since)
	if err != nil {
		adminErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"since": since, "usage": rows})
}

func adminErr(w http.ResponseWriter, code int, msg string) {
	writeJSON(w, code, map[string]string{"error": msg})
}

func lastSegment(repo string) string {
	for i := len(repo) - 1; i >= 0; i-- {
		if repo[i] == '/' {
			return repo[i+1:]
		}
	}
	return repo
}

// StoreRegistry adapts the database to the gateway's Registry interface, replacing
// the M1 agent-polling registry once a control plane exists.
type StoreRegistry struct {
	Store *store.Store
	// Ensurer is installed by the swap scheduler in M5. Until then a request for a
	// model with no ready replica fails rather than waiting.
	Ensurer func(ctx context.Context, model string) error
}

// Replicas returns a model's replicas from the registry.
func (sr *StoreRegistry) Replicas(model string) []Replica {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	reps, err := sr.Store.Replicas(ctx, model)
	if err != nil {
		return nil
	}
	out := make([]Replica, 0, len(reps))
	for _, r := range reps {
		out = append(out, Replica{
			ID: replicaID(r.ID), Model: model,
			BaseURL: fmt.Sprintf("http://127.0.0.1:%d", r.Port),
			State:   r.State, Ready: r.State == "ACTIVE",
		})
	}
	return out
}

// Models lists deployed model ids.
func (sr *StoreRegistry) Models() []string {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	ms, err := sr.Store.Models(ctx)
	if err != nil {
		return nil
	}
	out := make([]string, 0, len(ms))
	for _, m := range ms {
		out = append(out, m.ID)
	}
	return out
}

// Ensure delegates to the installed ensurer.
func (sr *StoreRegistry) Ensure(ctx context.Context, model string) error {
	if sr.Ensurer != nil {
		return sr.Ensurer(ctx, model)
	}
	for _, r := range sr.Replicas(model) {
		if r.Ready {
			return nil
		}
	}
	return fmt.Errorf("model %q has no ready replica", model)
}

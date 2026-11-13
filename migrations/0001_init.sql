-- Kiln schema.
--
-- Three concerns live here: multi-tenancy (teams, keys, quotas, usage), the model
-- registry (models, replicas), and the deploy job queue. The job queue is the part
-- with non-obvious requirements, and its comments explain why.

CREATE TABLE IF NOT EXISTS teams (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- API keys are stored only as a hash. A leaked database dump must not hand the
-- attacker working credentials.
--
-- key_prefix is the first few characters of the plaintext, stored deliberately: it is
-- what lets a human identify which key to revoke from a dashboard or a log line
-- without ever seeing the secret, and it narrows the lookup to one row so verification
-- is a single hash comparison rather than a scan over every key.
CREATE TABLE IF NOT EXISTS api_keys (
    id          TEXT PRIMARY KEY,
    team_id     TEXT NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
    key_prefix  TEXT NOT NULL,
    key_hash    BYTEA NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_used_at TIMESTAMPTZ,
    revoked_at  TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS api_keys_prefix_idx ON api_keys (key_prefix) WHERE revoked_at IS NULL;
CREATE INDEX IF NOT EXISTS api_keys_team_idx ON api_keys (team_id);

CREATE TABLE IF NOT EXISTS quotas (
    team_id         TEXT PRIMARY KEY REFERENCES teams(id) ON DELETE CASCADE,
    rpm             INTEGER NOT NULL DEFAULT 60,
    tokens_per_day  BIGINT  NOT NULL DEFAULT 1000000,
    max_concurrency INTEGER NOT NULL DEFAULT 8
);

-- Usage is append-only, one row per request.
--
-- Not an UPDATE on a running total: concurrent requests from one team would contend
-- on a single row, and the write lock would serialise the whole gateway behind the
-- busiest tenant. Appending is contention-free and the daily total is a cheap indexed
-- aggregate.
CREATE TABLE IF NOT EXISTS usage (
    id                BIGSERIAL PRIMARY KEY,
    team_id           TEXT NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
    model_id          TEXT NOT NULL,
    ts                TIMESTAMPTZ NOT NULL DEFAULT now(),
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    status_code       INTEGER NOT NULL DEFAULT 200
);
CREATE INDEX IF NOT EXISTS usage_team_ts_idx ON usage (team_id, ts DESC);
CREATE INDEX IF NOT EXISTS usage_model_ts_idx ON usage (model_id, ts DESC);

CREATE TABLE IF NOT EXISTS models (
    id                     TEXT PRIMARY KEY,
    hf_repo                TEXT NOT NULL,
    revision               TEXT NOT NULL DEFAULT '',
    replicas               INTEGER NOT NULL DEFAULT 1,
    gpu_memory_utilization DOUBLE PRECISION NOT NULL DEFAULT 0.35,
    max_model_len          INTEGER NOT NULL DEFAULT 4096,
    enforce_eager          BOOLEAN NOT NULL DEFAULT false,
    -- Measured at deploy time rather than estimated from parameter count: the KV
    -- cache arena usually dominates the footprint and depends on the flags above.
    -- The cost-aware eviction policy scores against these, so they have to be this
    -- GPU's real numbers.
    footprint_vram_bytes   BIGINT NOT NULL DEFAULT 0,
    footprint_host_bytes   BIGINT NOT NULL DEFAULT 0,
    cold_start_ms          BIGINT NOT NULL DEFAULT 0,
    wake_host_ms           BIGINT NOT NULL DEFAULT 0,
    wake_disk_ms           BIGINT NOT NULL DEFAULT 0,
    -- Pinned models are never chosen as eviction victims.
    pinned                 BOOLEAN NOT NULL DEFAULT false,
    status                 TEXT NOT NULL DEFAULT 'PENDING',
    error                  TEXT NOT NULL DEFAULT '',
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at             TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS replicas (
    id         TEXT PRIMARY KEY,
    model_id   TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    node       TEXT NOT NULL,
    agent_url  TEXT NOT NULL,
    port       INTEGER NOT NULL DEFAULT 0,
    state      TEXT NOT NULL DEFAULT 'COLD',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS replicas_model_idx ON replicas (model_id);

-- Deploy jobs.
--
-- Workers claim jobs with SELECT ... FOR UPDATE SKIP LOCKED, which is what makes
-- several control-plane replicas safe to run at once: each transaction takes a row
-- lock, and SKIP LOCKED makes the others step over it instead of blocking. Without
-- SKIP LOCKED, N workers serialise behind one row and the queue stops being a queue.
--
-- The row lock alone is not enough, because it dies with the transaction, and the
-- claim must outlive it — a job takes minutes while the claiming transaction commits
-- in milliseconds. So the claim is also a lease: locked_by plus lease_until. A worker
-- that crashes mid-job leaves its lease to expire, and the job is reclaimed. A worker
-- that is merely slow renews it.
CREATE TABLE IF NOT EXISTS deploy_jobs (
    id          TEXT PRIMARY KEY,
    model_id    TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL,                       -- deploy | scale | delete
    status      TEXT NOT NULL DEFAULT 'PENDING',     -- PENDING | RUNNING | DONE | FAILED
    attempts    INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    payload     JSONB NOT NULL DEFAULT '{}'::jsonb,
    error       TEXT NOT NULL DEFAULT '',
    locked_by   TEXT,
    lease_until TIMESTAMPTZ,
    -- run_after implements retry backoff: a failed job is rescheduled rather than
    -- retried immediately, so a model whose weights 404 does not spin the worker.
    run_after   TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- The claim query's index. Partial, because only claimable rows are ever scanned and
-- the DONE rows are the ones that accumulate.
CREATE INDEX IF NOT EXISTS deploy_jobs_claimable_idx
    ON deploy_jobs (run_after, created_at)
    WHERE status IN ('PENDING', 'RUNNING');

-- At most one live job per model. Two concurrent deploys of one model would race to
-- start replicas and double-allocate the GPU; the database refuses rather than
-- relying on every caller to check first.
CREATE UNIQUE INDEX IF NOT EXISTS deploy_jobs_one_live_per_model
    ON deploy_jobs (model_id)
    WHERE status IN ('PENDING', 'RUNNING');

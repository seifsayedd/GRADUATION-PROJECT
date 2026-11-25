-- SentinelAI Phase 5 schema: PostgreSQL 16+ with the pgvector extension.
-- Embedding size 384 = sentence-transformers/all-MiniLM-L6-v2 (SENTINEL_EMBED_DIM).
-- Apply with: python scripts/genai_phase5.py init-db   (needs DATABASE_URL)

CREATE EXTENSION IF NOT EXISTS vector;

-- Project documents (findings, taxonomy, reports) split into chunks
CREATE TABLE IF NOT EXISTS chunks (
    id         text PRIMARY KEY,                 -- "<source>#<n>"
    source     text NOT NULL,                    -- repository path
    section    text NOT NULL,                    -- heading path inside the file
    kind       text NOT NULL CHECK (kind IN ('doc', 'report', 'explanation')),
    text       text NOT NULL,
    meta       jsonb NOT NULL DEFAULT '{}'::jsonb,
    embedding  vector(384) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_kind ON chunks (kind);

-- One Phase 4 explanation object per alert (docs/schema/explanation.schema.json)
CREATE TABLE IF NOT EXISTS explanations (
    alert_id   text PRIMARY KEY,
    stage      smallint NOT NULL CHECK (stage IN (1, 2, 3)),
    model      text NOT NULL,
    flagged    boolean NOT NULL,
    body       jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

-- Generated reports; only reports that passed the groundedness check are stored
CREATE TABLE IF NOT EXISTS reports (
    id         bigserial PRIMARY KEY,
    alert_id   text REFERENCES explanations (alert_id),
    body       jsonb NOT NULL,
    citations  text[] NOT NULL,                  -- chunk ids and "alert:<id>" the report cites
    provider   text NOT NULL,                    -- anthropic | template
    created_at timestamptz NOT NULL DEFAULT now()
);

-- Analyst Q&A history
CREATE TABLE IF NOT EXISTS chat_sessions (
    id         bigserial PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS chat_messages (
    id         bigserial PRIMARY KEY,
    session_id bigint NOT NULL REFERENCES chat_sessions (id) ON DELETE CASCADE,
    role       text NOT NULL CHECK (role IN ('analyst', 'assistant')),
    content    text NOT NULL,
    citations  text[] NOT NULL DEFAULT '{}',
    created_at timestamptz NOT NULL DEFAULT now()
);

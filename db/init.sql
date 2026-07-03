-- 导演工厂 — PostgreSQL 初始化脚本

CREATE EXTENSION IF NOT EXISTS vector;

-- 资产记录表 (script_repo / char_asset_db / prompt_repo / sfx_library)
CREATE TABLE IF NOT EXISTS asset_records (
    id SERIAL PRIMARY KEY,
    db_type VARCHAR(64) NOT NULL,
    key VARCHAR(512) NOT NULL,
    data JSONB NOT NULL DEFAULT '{}',
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at DOUBLE PRECISION NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    locked BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE(db_type, key)
);

CREATE INDEX IF NOT EXISTS idx_asset_db_type ON asset_records(db_type);
CREATE INDEX IF NOT EXISTS idx_asset_key ON asset_records(db_type, key);
CREATE INDEX IF NOT EXISTS idx_asset_data_gin ON asset_records USING GIN (data);
CREATE INDEX IF NOT EXISTS idx_asset_meta_gin ON asset_records USING GIN (metadata);

-- 角色向量嵌入表
CREATE TABLE IF NOT EXISTS char_embeddings (
    id SERIAL PRIMARY KEY,
    character_id VARCHAR(256) NOT NULL UNIQUE,
    embedding vector(512),
    created_at DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_char_embedding_ivfflat
    ON char_embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- 检查点表
CREATE TABLE IF NOT EXISTS checkpoints (
    id SERIAL PRIMARY KEY,
    project_id VARCHAR(256) NOT NULL,
    phase VARCHAR(64) NOT NULL,
    state JSONB NOT NULL DEFAULT '{}',
    saved_at DOUBLE PRECISION NOT NULL,
    UNIQUE(project_id, phase)
);

CREATE INDEX IF NOT EXISTS idx_checkpoint_project ON checkpoints(project_id);

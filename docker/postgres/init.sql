-- Multimodal DB. Script de inicialización de PostgreSQL.
-- Corre una sola vez al montarse en /docker-entrypoint-initdb.d.
-- El schema engine persiste el engine propio.
-- El schema compare guarda los baselines con GIN y pgvector.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Schema engine. Estado durable de nuestro engine multimodal.
CREATE SCHEMA IF NOT EXISTS engine;

-- Una fila por codebook construido con sus parámetros.
CREATE TABLE IF NOT EXISTS engine.codebook (
    codebook_id   SERIAL PRIMARY KEY,
    modality      TEXT    NOT NULL CHECK (modality IN ('TEXT', 'IMAGE', 'AUDIO')),
    feature_type  TEXT    NOT NULL,
    k             INTEGER NOT NULL,
    dim           INTEGER NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Una fila por codeword o cluster centroid de un codebook.
CREATE TABLE IF NOT EXISTS engine.codeword (
    codebook_id   INTEGER NOT NULL REFERENCES engine.codebook(codebook_id) ON DELETE CASCADE,
    word_id       INTEGER NOT NULL,
    centroid      vector,
    idf           REAL    NOT NULL DEFAULT 0,
    PRIMARY KEY (codebook_id, word_id)
);

-- Histograma por documento.
CREATE TABLE IF NOT EXISTS engine.histogram (
    codebook_id   INTEGER NOT NULL REFERENCES engine.codebook(codebook_id) ON DELETE CASCADE,
    doc_id        BIGINT  NOT NULL,
    histogram     vector,
    -- L2 norm cacheada para el cosine scoring
    norm          REAL    NOT NULL DEFAULT 0,
    PRIMARY KEY (codebook_id, doc_id)
);

-- Inverted index. Mapea cada codeword a su postings list.
CREATE TABLE IF NOT EXISTS engine.posting (
    codebook_id   INTEGER NOT NULL REFERENCES engine.codebook(codebook_id) ON DELETE CASCADE,
    word_id       INTEGER NOT NULL,
    doc_id        BIGINT  NOT NULL,
    weight        REAL    NOT NULL,
    PRIMARY KEY (codebook_id, word_id, doc_id)
);
CREATE INDEX IF NOT EXISTS idx_posting_lookup
    ON engine.posting (codebook_id, word_id);

-- Catalog metadata libre para tablas e índices del engine.
CREATE TABLE IF NOT EXISTS engine.metadata (
    key           TEXT PRIMARY KEY,
    value         JSONB NOT NULL,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Schema compare. Baselines nativos.
CREATE SCHEMA IF NOT EXISTS compare;

-- Baseline de texto. Full-text search con GIN index.
CREATE TABLE IF NOT EXISTS compare.documents (
    id        INTEGER PRIMARY KEY,
    body      TEXT NOT NULL,
    -- La columna tsvector generada mantiene el index sincronizado
    fts       tsvector GENERATED ALWAYS AS (to_tsvector('english', body)) STORED
);
CREATE INDEX IF NOT EXISTS idx_documents_fts
    ON compare.documents USING GIN (fts);

-- Baseline de imagen y audio. Histograma como columna pgvector para KNN.
-- La dimensión del vector se fija según el k del codebook en uso.
CREATE TABLE IF NOT EXISTS compare.media (
    id            INTEGER PRIMARY KEY,
    modality      TEXT NOT NULL CHECK (modality IN ('IMAGE', 'AUDIO')),
    path          TEXT NOT NULL,
    feature_vec   vector(256) NOT NULL
);

-- IVFFlat index cosine para KNN aproximado.
-- Construir después del bulk load y del ANALYZE.
-- El parámetro lists se afina a la raíz cuadrada del número de filas.
CREATE INDEX IF NOT EXISTS idx_media_ivfflat
    ON compare.media USING ivfflat (feature_vec vector_cosine_ops)
    WITH (lists = 100);

-- Alternativa HNSW opcional para comparar ANN graph contra IVF.
-- CREATE INDEX IF NOT EXISTS idx_media_hnsw
--     ON compare.media USING hnsw (feature_vec vector_cosine_ops);

-- Grants para el rol único de desarrollo del proyecto.
GRANT ALL ON ALL TABLES    IN SCHEMA engine, compare TO CURRENT_USER;
GRANT ALL ON ALL SEQUENCES IN SCHEMA engine, compare TO CURRENT_USER;

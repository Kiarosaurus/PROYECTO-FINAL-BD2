-- Multimodal DB. Script de inicialización de PostgreSQL.
-- Corre una sola vez al montarse en /docker-entrypoint-initdb.d.
-- El schema engine persiste el engine propio.
-- El schema compare guarda los baselines con GIN y pgvector.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Schema engine. Estado durable de nuestro engine multimodal.
CREATE SCHEMA IF NOT EXISTS engine;

-- Toda la persistencia del engine viaja como páginas del StorageEngine.
-- El codebook, los histogramas del KNN y los snapshots de índices llegan aquí.
-- Por eso engine.page es la única tabla del schema.
CREATE TABLE IF NOT EXISTS engine.page (
    file_id       TEXT    NOT NULL,
    page_no       INTEGER NOT NULL,
    data          BYTEA   NOT NULL,
    PRIMARY KEY (file_id, page_no)
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

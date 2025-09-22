
-- db/schema.sql
-- Enable pgvector (run as a DB superuser or have the privilege)
CREATE EXTENSION IF NOT EXISTS vector;

-- Table to store file metadata
CREATE TABLE IF NOT EXISTS netcdf_files (
    id SERIAL PRIMARY KEY,
    file_url TEXT NOT NULL UNIQUE,
    local_path TEXT,
    file_name TEXT,
    year INTEGER,
    ocean_region TEXT,
    lat_min REAL, lat_max REAL,
    lon_min REAL, lon_max REAL,
    time_start TIMESTAMP,
    time_end TIMESTAMP,
    variables TEXT[],
    summary TEXT,
    checksum TEXT,
    ingested_at TIMESTAMP DEFAULT now()
);

-- Table for embeddings (pgvector)
CREATE TABLE IF NOT EXISTS summaries_vector (
    file_id INTEGER PRIMARY KEY REFERENCES netcdf_files(id) ON DELETE CASCADE,
    embedding vector(384)
);

-- Optional: index for vector search (ivfflat). Note: populate table before creating ivfflat for best results.
-- Use vector_cosine_ops or vector_l2_ops depending on operator you plan to use.
CREATE INDEX IF NOT EXISTS idx_summaries_vector_embedding ON summaries_vector USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Index for quick year/region filtering
CREATE INDEX IF NOT EXISTS idx_netcdf_year_region ON netcdf_files(year, ocean_region);

# db/db_helpers.py
import os
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

# Provide DATABASE_URL via environment variable (do NOT commit credentials)
# Example: postgresql://user:password@host:5432/floatchat
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("Set the DATABASE_URL environment variable before using db_helpers.")

# SQLAlchemy engine (use future=True for 2.0 style)
engine: Engine = create_engine(DATABASE_URL, future=True)

def run_schema(sql_path: str):
    """
    Run SQL file (useful to create extensions / tables).
    """
    with engine.connect() as conn:
        with open(sql_path, "r", encoding="utf-8") as f:
            sql = f.read()
        conn.execute(text(sql))
        conn.commit()

def insert_file_metadata(conn, metadata: dict):
    """
    Insert a row into netcdf_files. Returns new id or None if already exists.
    Expects metadata keys matching columns (file_url, local_path, file_name, year, ocean_region,
    lat_min, lat_max, lon_min, lon_max, time_start, time_end, variables, summary, checksum)
    """
    sql = text("""
    INSERT INTO netcdf_files
      (file_url, local_path, file_name, year, ocean_region,
       lat_min, lat_max, lon_min, lon_max, time_start, time_end,
       variables, summary, checksum)
    VALUES
      (:file_url, :local_path, :file_name, :year, :ocean_region,
       :lat_min, :lat_max, :lon_min, :lon_max, :time_start, :time_end,
       :variables, :summary, :checksum)
    RETURNING id
    """)
    try:
        res = conn.execute(sql, metadata)
        conn.commit()
        return res.scalar_one()
    except IntegrityError:
        conn.rollback()
        return None

def upsert_embedding(conn, file_id: int, embedding: list):
    """
    Insert or update embedding for a file_id into summaries_vector.
    embedding should be a list of floats (length matches your vector dim, e.g. 384).
    """
    sql = text("""
    INSERT INTO summaries_vector (file_id, embedding)
    VALUES (:file_id, :embedding)
    ON CONFLICT (file_id) DO UPDATE SET embedding = EXCLUDED.embedding
    """)
    conn.execute(sql, {"file_id": file_id, "embedding": embedding})
    conn.commit()

def file_exists(conn, file_url: str) -> bool:
    """
    Check whether file_url already exists in netcdf_files.
    """
    res = conn.execute(text("SELECT id FROM netcdf_files WHERE file_url = :u"), {"u": file_url}).fetchone()
    return res is not None

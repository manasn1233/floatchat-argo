# downloader/ingest.py
"""
Ingest a downloaded netCDF file into Postgres:
 - extract metadata/summary using downloader.extractor.summarize_netcdf
 - insert metadata into netcdf_files (skips if file_url already exists)
 - create a text embedding for the summary and upsert into summaries_vector
"""

from db import db_helpers
from downloader.extractor import summarize_netcdf
from sentence_transformers import SentenceTransformer
from sqlalchemy import text
import os

# Embedding model (384 dims)
EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def ingest_file_entry(entry, ocean_region="unknown"):
    """
    entry: dict with keys:
      - url (file_url)
      - local_path
      - file_name
      - checksum
    ocean_region: 'atlantic' or 'indian' (or other label)
    """
    if not entry or "local_path" not in entry or not os.path.exists(entry["local_path"]):
        raise ValueError("Invalid entry or local file missing: %s" % (entry,))

    # extract metadata / summary
    meta = summarize_netcdf(entry["local_path"])

    payload = {
        "file_url": entry.get("url"),
        "local_path": entry.get("local_path"),
        "file_name": entry.get("file_name"),
        "year": meta.get("year"),
        "ocean_region": ocean_region,
        "lat_min": meta.get("lat_min"),
        "lat_max": meta.get("lat_max"),
        "lon_min": meta.get("lon_min"),
        "lon_max": meta.get("lon_max"),
        "time_start": meta.get("time_start"),
        "time_end": meta.get("time_end"),
        "variables": meta.get("variables"),
        "summary": meta.get("summary"),
        "checksum": entry.get("checksum")
    }

    with db_helpers.engine.begin() as conn:
        # try insert metadata (insert_file_metadata will return None if already exists)
        fid = db_helpers.insert_file_metadata(conn, payload)
        if not fid:
            print("File already present in DB, skipping ingestion:", entry.get("url"))
            return None

        # create embedding for the summary text
        summary = meta.get("summary") or ""
        emb = EMB_MODEL.encode(summary)
        emb_list = emb.tolist()
        db_helpers.upsert_embedding(conn, fid, emb_list)
        print("Ingested file id", fid, "url", entry.get("url"))
        return fid

# convenience helper to ingest multiple entries
def ingest_batch(entries, ocean_region="unknown"):
    results = []
    for e in entries:
        try:
            fid = ingest_file_entry(e, ocean_region=ocean_region)
            results.append({"entry": e, "id": fid})
        except Exception as exc:
            print("Failed to ingest", e.get("url"), exc)
    return results

if __name__ == "__main__":
    # quick test usage example (won't run in Actions without network/files)
    sample_entry = {
        "url": "https://data-argo.ifremer.fr/geo/atlantic_ocean/2023/sample.nc",
        "local_path": "./data/netcdf/sample.nc",
        "file_name": "sample.nc",
        "checksum": "dummy"
    }
    print("Ingest test (no file present will raise):")
    try:
        ingest_file_entry(sample_entry, ocean_region="atlantic")
    except Exception as e:
        print("Expected error for local test:", e)

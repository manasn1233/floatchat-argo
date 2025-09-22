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

    # Use a single connection and a single transaction for both metadata insert and embedding upsert.
    # This avoids nested/closed transaction errors that happen when helper functions
    # try to manage transactions on the same connection.
    conn = db_helpers.engine.connect()
    trans = conn.begin()
    try:
        # insert_file_metadata should accept a connection and perform only the INSERT/SELECT,
        # without committing/closing the connection itself.
        fid = db_helpers.insert_file_metadata(conn, payload)
        if not fid:
            # file already exists (insert_file_metadata returned None or False)
            trans.rollback()
            print("File already present in DB, skipping ingestion:", entry.get("url"))
            return None

        # create embedding for the summary text
        summary = meta.get("summary") or ""
        emb = EMB_MODEL.encode(summary)
        emb_list = emb.tolist()

        # upsert embedding using the same connection within the open transaction
        db_helpers.upsert_embedding(conn, fid, emb_list)

        # commit once after both operations succeed
        trans.commit()
        print("Ingested file id", fid, "url", entry.get("url"))
        return fid
    except Exception as exc:
        try:
            trans.rollback()
        except Exception:
            pass
        print("Failed to ingest", entry.get("url"), exc)
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass

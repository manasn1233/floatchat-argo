# apps/streamlit_rag.py
"""
Streamlit RAG chat for FloatChat (NetCDF metadata + summaries).
- Primary: query Postgres (Supabase) for metadata rows.
- Fallback: if DB returns no results for filters, search a local FAISS index (meta.json + index file).
- Re-ranks with SBERT and cosine similarity.
- Optional: Gemini generation if package & GEMINI_API_KEY available.
- Optional: gTTS playback when available.
"""

import os
import html
import time
import json
from typing import List, Dict, Optional

import streamlit as st
import numpy as np
import sqlalchemy
from sqlalchemy import text
from sentence_transformers import SentenceTransformer

# Optional Gemini package (try import but tolerate absence)
try:
    import google.generativeai as genai  # optional; may not exist in some deploy envs
    GEMINI_PACKAGE_AVAILABLE = True
except Exception:
    GEMINI_PACKAGE_AVAILABLE = False

# Optional TTS
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except Exception:
    GTTS_AVAILABLE = False

# Optional FAISS
FAISS_AVAILABLE = True
try:
    import faiss
except Exception:
    FAISS_AVAILABLE = False

# ---------------- Helpers to read secrets/env ----------------
def get_secret(key: str) -> Optional[str]:
    """Prefer streamlit secrets, fallback to environment variable."""
    try:
        if hasattr(st, "secrets") and st.secrets and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key)

DATABASE_URL = get_secret("DATABASE_URL")
GEMINI_API_KEY = get_secret("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY", "")

# Friendly message if no DB configured (we still allow FAISS fallback)
if not DATABASE_URL:
    st.warning("DATABASE_URL not set — DB queries will fail. FAISS local fallback will still work if available.")

def mask_db_url(url: str) -> str:
    try:
        parts = url.split("@")
        if len(parts) == 2:
            return f"...@{parts[1]}"
    except Exception:
        pass
    return "configured"

# ---------------- DB engine & model loading (cached) ----------------
@st.cache_resource(show_spinner=False)
def get_engine(db_url: str):
    if not db_url:
        return None
    return sqlalchemy.create_engine(db_url, future=True)

@st.cache_resource(show_spinner=False)
def load_sbert_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

engine = get_engine(DATABASE_URL)
sbert = load_sbert_model()

# ---------------- FAISS local index (fallback) ----------------
PERSIST_DIR = os.getenv("PERSIST_DIR", "./faiss_db_proto")
INDEX_PATH = os.path.join(PERSIST_DIR, "faiss_index.bin")
META_PATH = os.path.join(PERSIST_DIR, "meta.json")
os.makedirs(PERSIST_DIR, exist_ok=True)

def _ensure_sample_meta():
    """
    If no meta.json exists, create a tiny sample set so local fallback returns something.
    """
    if os.path.exists(META_PATH) and FAISS_AVAILABLE:
        return
    docs = [
        {"id":"R_sample_001","file_name":"sample_2023_01.nc","file_url":"local://sample_2023_01","summary":"Salinity profile near 7N,75E recorded on 2023-03-15."},
        {"id":"R_sample_002","file_name":"sample_2023_02.nc","file_url":"local://sample_2023_02","summary":"Temperature profile near 10N,70E on 2023-02-10."},
        {"id":"R_sample_003","file_name":"sample_2023_03.nc","file_url":"local://sample_2023_03","summary":"BGC float near 5N,72E: chlorophyll elevated at surface."},
    ]
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

def build_or_load_faiss():
    """
    Ensure an index exists and return (index, meta_list).
    meta_list is a list of dicts with at least 'summary' and 'file_url' keys.
    """
    if not FAISS_AVAILABLE:
        return None, []

    _ensure_sample_meta()

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    texts = [m.get("summary","") for m in meta]
    # encode with SBERT
    embs = sbert.encode(texts, show_progress_bar=False)
    arr = np.array(embs).astype("float32")
    # L2 normalize for cosine via inner product
    faiss.normalize_L2(arr)

    d = arr.shape[1] if arr.ndim == 2 and arr.shape[1] > 0 else 384
    index = faiss.IndexFlatIP(d)
    index.add(arr)
    # persist index for future runs
    try:
        faiss.write_index(index, INDEX_PATH)
    except Exception:
        pass
    return index, meta

# Cache index and meta (so not rebuilt on every run)
@st.cache_resource(show_spinner=False)
def get_faiss_resources():
    return build_or_load_faiss()

faiss_index, faiss_meta = get_faiss_resources()

def faiss_search(query: str, k: int = 5):
    """
    Return list of docs (dict) similar to query using local FAISS index.
    Each doc will have keys similar to DB rows: file_url, file_name, summary, year (maybe inferred).
    """
    if not FAISS_AVAILABLE or faiss_index is None or not faiss_meta:
        return []
    q_emb = sbert.encode([query], show_progress_bar=False)
    q_arr = np.array(q_emb, dtype=np.float32)
    faiss.normalize_L2(q_arr)
    D, I = faiss_index.search(q_arr, k)
    results = []
    for idx in I[0]:
        if idx < 0 or idx >= len(faiss_meta):
            continue
        m = dict(faiss_meta[idx])
        # try to infer year from file_name if present
        if not m.get("year") and m.get("file_name"):
            try:
                m["year"] = int(m["file_name"][:4])
            except Exception:
                pass
        results.append(m)
    return results

# ---------------- Gemini helper ----------------
def generate_with_gemini(question: str, top_docs: List[Dict]) -> str:
    """
    Uses google.generativeai if installed and GEMINI_API_KEY is set.
    Otherwise returns empty string (caller will use local synth).
    """
    if not GEMINI_PACKAGE_AVAILABLE or not GEMINI_API_KEY:
        return ""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-pro")
        context = "\n\n".join(f"[DOC {i+1}] {d.get('summary','')[:1000]}" for i, d in enumerate(top_docs))
        prompt = f"{context}\n\nUser question: {question}\nAnswer concisely using the above documents:"
        resp = model.generate_content(prompt, generation_config={"max_output_tokens":400, "temperature":0.2})
        # robust extraction
        if hasattr(resp, "text") and resp.text:
            return resp.text
        if hasattr(resp, "candidates") and resp.candidates:
            c = resp.candidates[0]
            if hasattr(c, "text") and c.text:
                return c.text
        return str(resp)
    except Exception as e:
        return f"Gemini generation failed: {e}"

def synthesize_answer_locally(question: str, top_docs: List[Dict]) -> str:
    """
    Local fallback answer: concatenate top summaries.
    """
    if not top_docs:
        return "No documents found for the selected filters."
    parts = [f"- {d.get('summary','')[:500]}..." for d in top_docs]
    answer = f"I found {len(top_docs)} relevant documents. Top summaries:\n\n" + "\n\n".join(parts)
    return answer

def speak_text_tts(text: str):
    if not GTTS_AVAILABLE or not text:
        return None
    try:
        tts = gTTS(text=text, lang="en")
        import tempfile, os
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            path = tmp.name
        tts.save(path)
        data = open(path, "rb").read()
        os.remove(path)
        return data
    except Exception:
        return None

# ---------------- DB fetch & ranking helpers (unchanged behaviour) ----------------
def fetch_candidates(year: Optional[int], region: Optional[str],
                     lat_min: Optional[float], lat_max: Optional[float],
                     lon_min: Optional[float], lon_max: Optional[float],
                     limit: int = 200) -> List[Dict]:
    """
    Fetch candidate rows from netcdf_files table.
    Fallback: if DB present but returns no rows, the calling code will use faiss_search().
    """
    clauses = []
    params = {}
    if year:
        clauses.append("year = :year")
        params["year"] = int(year)
    if region and region != "all":
        clauses.append("ocean_region = :region")
        params["region"] = region
    if lat_min is not None:
        clauses.append("lat_max >= :lat_min")
        params["lat_min"] = lat_min
    if lat_max is not None:
        clauses.append("lat_min <= :lat_max")
        params["lat_max"] = lat_max
    if lon_min is not None:
        clauses.append("lon_max >= :lon_min")
        params["lon_min"] = lon_min
    if lon_max is not None:
        clauses.append("lon_min <= :lon_max")
        params["lon_max"] = lon_max

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"""
    SELECT id, file_url, file_name, year, ocean_region, summary, lat_min, lat_max, lon_min, lon_max, time_start, time_end
    FROM netcdf_files
    {where}
    ORDER BY id DESC
    LIMIT :limit
    """
    params["limit"] = limit
    if not engine:
        return []
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(sql), params).fetchall()
        results = [dict(r._mapping) for r in rows]
        # infer year from file_name when missing to improve filtering
        for rec in results:
            if not rec.get("year") and rec.get("file_name"):
                try:
                    rec["year"] = int(rec["file_name"][:4])
                except Exception:
                    pass
        return results
    except Exception as e:
        # log and return empty so caller can fall back to FAISS
        st.error(f"DB query failed (falling back to local index if available): {e}")
        return []

def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, sbert.get_sentence_embedding_dimension()), dtype=np.float32)
    embs = sbert.encode(texts, show_progress_bar=False)
    arr = np.array(embs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms
    return arr

def top_k_by_cosine(query: str, docs: List[Dict], k: int = 5):
    summaries = [d.get("summary") or "" for d in docs]
    if len(summaries) == 0:
        return []
    doc_embs = embed_texts(summaries)
    q_emb = embed_texts([query])[0]
    sims = (doc_embs @ q_emb).astype(float)
    idx = np.argsort(-sims)[:k]
    results = []
    for i in idx:
        results.append({"score": float(sims[i]), "doc": docs[i]})
    return results

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="FloatChat — RAG", layout="centered")
st.title("FloatChat — RAG (NetCDF metadata)")

with st.sidebar:
    st.markdown("### Filters")
    year = st.number_input("Year", min_value=1900, max_value=2100, value=2025, step=1)
    region = st.selectbox("Region", options=["indian", "atlantic", "all"], index=0)
    use_bbox = st.checkbox("Filter by bounding box (lat/lon)?", value=False)
    lat_min = lat_max = lon_min = lon_max = None
    if use_bbox:
        lat_min = st.number_input("lat_min", value=-90.0, format="%.6f")
        lat_max = st.number_input("lat_max", value=90.0, format="%.6f")
        lon_min = st.number_input("lon_min", value=-180.0, format="%.6f")
        lon_max = st.number_input("lon_max", value=180.0, format="%.6f")
    top_k = st.slider("Documents to retrieve (k)", 1, 10, 5)

if DATABASE_URL:
    st.markdown(f"**DB:** {mask_db_url(DATABASE_URL)} — connection configured.")
else:
    st.markdown("**DB:** not configured (using local FAISS fallback if available).")

if GEMINI_PACKAGE_AVAILABLE:
    st.caption("Gemini package installed. Gemini will be used if GEMINI_API_KEY is set.")
else:
    st.caption("Gemini package not available; local synthesis will be used unless GEMINI configured.")

st.markdown("---")
question = st.text_area("Ask a question (about year, region, variables, summaries, etc.)", height=120)
run = st.button("Search & Answer")

if run:
    filter_region = None if region == "all" else region
    with st.spinner("Fetching candidates from DB..."):
        candidates = fetch_candidates(year if year else None, filter_region, lat_min, lat_max, lon_min, lon_max, limit=500)

    # If DB returned none, try FAISS fallback
    if not candidates:
        if FAISS_AVAILABLE and faiss_meta:
            st.info("No DB rows found for those filters — using local FAISS fallback.")
            candidates = faiss_search(question, k=500)
        else:
            st.info("No candidates found in DB, and no FAISS fallback available.")
            candidates = []

    st.success(f"Found {len(candidates)} candidate rows (pre-rank)")

    if not candidates:
        st.info("No candidates found for these filters.")
    else:
        with st.spinner("Ranking candidates with SBERT..."):
            top = top_k_by_cosine(question, candidates, k=top_k)

        st.markdown("### Top documents")
        for i, t in enumerate(top, start=1):
            doc = t["doc"]
            st.markdown(f"**{i}. score {t['score']:.4f} — {html.escape(doc.get('file_name') or doc.get('file_url'))}**")
            st.markdown(f"*{html.escape((doc.get('summary') or '')[:700])}*")
            st.caption(f"url: {doc.get('file_url')}")
            st.write("---")

        # Generate answer (Gemini if configured & available, else local synth)
        if GEMINI_PACKAGE_AVAILABLE and GEMINI_API_KEY:
            with st.spinner("Generating answer with Gemini..."):
                ans = generate_with_gemini(question, top)
            st.markdown("### Answer (Gemini)")
            st.write(ans)
            mp3 = speak_text_tts(ans)
            if mp3:
                st.audio(mp3, format="audio/mp3")
        else:
            ans = synthesize_answer_locally(question, top)
            st.markdown("### Answer (Local synth)")
            st.write(ans)
            mp3 = speak_text_tts(ans)
            if mp3:
                st.audio(mp3, format="audio/mp3")

st.markdown("---")
st.caption("FloatChat RAG — DB primary, FAISS local fallback, SBERT ranking.")

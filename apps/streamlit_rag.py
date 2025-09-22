# apps/streamlit_rag.py
"""
Streamlit RAG chat for FloatChat (NetCDF metadata + summaries).
- Queries Postgres for netcdf_files filtered by year/region/bbox.
- Re-ranks candidate summaries using SBERT embeddings (local) and cosine similarity.
- Optionally calls Gemini (if GEMINI_API_KEY is set in secrets or env).
"""

import os
import html
from typing import List, Dict, Optional

import streamlit as st
import numpy as np
import sqlalchemy
from sqlalchemy import text
from sentence_transformers import SentenceTransformer

# Optional Gemini package
try:
    import google.generativeai as genai
    GEMINI_PACKAGE_AVAILABLE = True
except Exception:
    GEMINI_PACKAGE_AVAILABLE = False

# ---------------- Helpers to read secrets ----------------
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

# Friendly message if no DB configured
if not DATABASE_URL:
    st.set_page_config(page_title="FloatChat — RAG", layout="centered")
    st.title("FloatChat — RAG (NetCDF metadata)")
    st.error(
        "DATABASE_URL is not configured. On Streamlit Cloud add it under Settings → Secrets as `DATABASE_URL`. "
        "Locally export DATABASE_URL before running the app."
    )
    st.stop()

# Mask DB URL for display
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
    return sqlalchemy.create_engine(db_url, future=True)

@st.cache_resource(show_spinner=False)
def load_sbert_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

engine = get_engine(DATABASE_URL)
sbert = load_sbert_model()

# ---------------- Gemini helper ----------------
def generate_with_gemini(question: str, top_docs: List[Dict]) -> str:
    if not GEMINI_PACKAGE_AVAILABLE or not GEMINI_API_KEY:
        return ""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-pro")
        context = "\n\n".join(f"[DOC {i+1}] {d['doc'].get('summary')[:1000]}" for i, d in enumerate(top_docs))
        prompt = f"{context}\n\nUser question: {question}\nAnswer concisely using the above documents:"
        resp = model.generate_content(prompt, generation_config={"max_output_tokens":400, "temperature":0.2})
        if hasattr(resp, "text") and resp.text:
            return resp.text
        if hasattr(resp, "candidates") and resp.candidates:
            c = resp.candidates[0]
            if hasattr(c, "text") and c.text:
                return c.text
        return str(resp)
    except Exception as e:
        return f"Gemini generation failed: {e}"

# ---------------- DB fetch & ranking helpers ----------------
def fetch_candidates(year: Optional[int], region: Optional[str],
                     lat_min: Optional[float], lat_max: Optional[float],
                     lon_min: Optional[float], lon_max: Optional[float],
                     limit: int = 200) -> List[Dict]:
    """
    Fetch candidate rows from netcdf_files table.
    Falls back to inferring year from file_name if DB.year is NULL.
    """
    clauses = []
    params = {}

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
    SELECT id, file_url, file_name, year, ocean_region, summary,
           lat_min, lat_max, lon_min, lon_max, time_start, time_end
    FROM netcdf_files
    {where}
    ORDER BY id DESC
    LIMIT :limit
    """
    params["limit"] = limit
    results = []
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
        for r in rows:
            rec = dict(r._mapping)
            # Infer year if missing
            if not rec.get("year") and rec.get("file_name"):
                try:
                    rec["year"] = int(rec["file_name"][:4])
                except Exception:
                    pass
            results.append(rec)

    # Apply year filter after fallback
    if year:
        results = [r for r in results if str(r.get("year")) == str(year)]
    return results

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

def synthesize_answer_locally(question: str, top_docs: List[Dict]) -> str:
    if not top_docs:
        return "No documents found for the selected filters."
    parts = [f"- {d['doc'].get('summary')[:500]}..." for d in top_docs]
    answer = f"I found {len(top_docs)} relevant documents. Top summaries:\n\n" + "\n\n".join(parts)
    return answer

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

st.markdown(f"**DB:** {mask_db_url(DATABASE_URL)} — connection looks configured.")
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
    st.success(f"Found {len(candidates)} candidate rows in DB (pre-filter)")

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

        if GEMINI_API_KEY and GEMINI_PACKAGE_AVAILABLE:
            with st.spinner("Generating answer with Gemini..."):
                ans = generate_with_gemini(question, top)
            st.markdown("### Answer (Gemini)")
            st.write(ans)
        else:
            ans = synthesize_answer_locally(question, top)
            st.markdown("### Answer (Local synth)")
            st.write(ans)

st.markdown("---")
st.caption("FloatChat RAG — retrieves summaries from Supabase/Postgres and ranks with SBERT.")

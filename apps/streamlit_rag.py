# apps/streamlit_rag.py
"""
Streamlit RAG chat for FloatChat (NetCDF metadata + summaries).
- Queries Postgres for netcdf_files filtered by year/region/bbox.
- Re-ranks candidate summaries using SBERT embeddings (local) and cosine similarity.
- Optionally calls Gemini (if GEMINI_API_KEY is set) to generate final responses.
"""

import os
import html
import time
from typing import List, Dict, Optional

import streamlit as st
import numpy as np
import sqlalchemy
from sqlalchemy import text
from sentence_transformers import SentenceTransformer

# Optional Gemini (same pattern as your other file)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

# ---------------- Config ----------------
DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    st.error("Please set DATABASE_URL as an environment variable (Supabase connection string).")
    st.stop()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # optional

# Initialize DB engine (SQLAlchemy)
engine = sqlalchemy.create_engine(DB_URL, future=True)

# Load SBERT once
@st.cache_resource(show_spinner=False)
def load_sbert_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

sbert = load_sbert_model()

# ---------------- Helpers ----------------
def fetch_candidates(year: Optional[int], region: Optional[str],
                     lat_min: Optional[float], lat_max: Optional[float],
                     lon_min: Optional[float], lon_max: Optional[float],
                     limit: int = 200) -> List[Dict]:
    """
    Fetch candidate rows from netcdf_files table.
    Assumes table netcdf_files has columns: id, file_url, year, ocean_region, summary, lat_min, lat_max, lon_min, lon_max
    """
    clauses = []
    params = {}
    if year:
        clauses.append("year = :year")
        params["year"] = int(year)
    if region:
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
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    results = []
    for r in rows:
        results.append(dict(r._mapping))
    return results

def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, sbert.get_sentence_embedding_dimension()), dtype=np.float32)
    embs = sbert.encode(texts, show_progress_bar=False)
    arr = np.array(embs, dtype=np.float32)
    # L2 normalize
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms
    return arr

def top_k_by_cosine(query: str, docs: List[Dict], k: int = 5):
    """
    Return top-k documents by cosine similarity between query and doc.summary.
    """
    summaries = [d.get("summary") or "" for d in docs]
    if len(summaries) == 0:
        return []
    doc_embs = embed_texts(summaries)
    q_emb = embed_texts([query])[0]
    sims = (doc_embs @ q_emb).astype(float)  # cosine after normalization
    idx = np.argsort(-sims)[:k]
    results = []
    for i in idx:
        results.append({"score": float(sims[i]), "doc": docs[i]})
    return results

def generate_answer_gemini(question: str, top_docs: List[Dict]) -> str:
    if not GEMINI_AVAILABLE or not GEMINI_API_KEY:
        return ""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-pro")
        context = "\n\n".join(f"[DOC {i+1}] {d['doc'].get('summary')[:1000]}" for i, d in enumerate(top_docs))
        prompt = f"{context}\n\nUser question: {question}\nAnswer concisely using the above documents:"
        resp = model.generate_content(prompt, generation_config={"max_output_tokens":400, "temperature":0.2})
        # extract text robustly
        txt = ""
        if hasattr(resp, "text") and resp.text:
            txt = resp.text
        elif hasattr(resp, "candidates") and resp.candidates:
            txt = resp.candidates[0].text
        else:
            txt = str(resp)
        return txt
    except Exception as e:
        return f"Gemini generation failed: {e}"

def synthesize_answer_locally(question: str, top_docs: List[Dict]) -> str:
    """
    Simple local fallback: concatenate top summaries and return a short synthesized text.
    """
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

        # Generate answer (Gemini if configured, else local synth)
        if GEMINI_AVAILABLE and GEMINI_API_KEY:
            with st.spinner("Generating answer with Gemini..."):
                ans = generate_answer_gemini(question, top)
            st.markdown("### Answer (Gemini)")
            st.write(ans)
        else:
            ans = synthesize_answer_locally(question, top)
            st.markdown("### Answer (Local synth)")
            st.write(ans)

st.markdown("---")
st.caption("FloatChat RAG — retrieves summaries from Supabase/Postgres and ranks with SBERT.")

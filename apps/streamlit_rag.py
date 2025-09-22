# apps/streamlit_rag.py
"""
Streamlit RAG chat for FloatChat (NetCDF metadata + summaries).
- Primary source: query Postgres (Supabase) for metadata rows and build/load FAISS from them.
- Fallback: if DB unavailable or returns no rows, use a local sample meta.json and build FAISS from it.
- Re-ranks with SBERT and cosine similarity.
- Optional: Gemini generation if package & GEMINI_API_KEY available (read from env or Streamlit secrets).
- Optional: gTTS TTS playback when available.

Notes:
- Do NOT hardcode secrets. Set DATABASE_URL and GEMINI_API_KEY as environment variables or Streamlit secrets.
- Add `faiss-cpu`, `sentence-transformers`, `sqlalchemy`, `psycopg2-binary`, and `gtts` to requirements if you need those features.
"""

import os
import json
import time
import tempfile
import html
from datetime import datetime
from typing import List, Dict, Optional

import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer

# Optional imports
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_PACKAGE_AVAILABLE = True
except Exception:
    GEMINI_PACKAGE_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except Exception:
    GTTS_AVAILABLE = False

# Try to import SQLAlchemy for DB access (optional)
try:
    import sqlalchemy
    from sqlalchemy import text
    DB_AVAILABLE = True
except Exception:
    DB_AVAILABLE = False

# ---------------- CONFIG ----------------
PERSIST_DIR = os.getenv("PERSIST_DIR", "./faiss_db_proto")
INDEX_PATH = os.path.join(PERSIST_DIR, "faiss_index.bin")
META_PATH = os.path.join(PERSIST_DIR, "meta.json")
os.makedirs(PERSIST_DIR, exist_ok=True)

# Read Gemini key from env / streamlit secrets (do NOT hardcode)
def get_secret(key: str) -> Optional[str]:
    try:
        if hasattr(st, "secrets") and st.secrets and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key)

GEMINI_API_KEY = get_secret("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY", "")

MODEL_HANDLE = None
if GEMINI_PACKAGE_AVAILABLE and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        MODEL_HANDLE = genai.GenerativeModel("gemini-1.5-pro")
    except Exception as e:
        # Do not fail app if Gemini init fails
        st.warning(f"Could not initialize Gemini model: {e}")
        MODEL_HANDLE = None

# ---------------- CSS ----------------
CSS = """<style>
header, footer {visibility:hidden; height:0;}
.topbar {position:fixed; top:0; left:0; right:0; height:60px;
 background:linear-gradient(90deg,#0077cc,#004e92); display:flex; align-items:center; justify-content:center;
 color:#fff; font-weight:700; font-family:"Segoe UI",sans-serif; z-index:9999;}
.messages {display:flex; flex-direction:column-reverse; overflow-y:auto; max-height:480px; padding:12px;
 background:linear-gradient(180deg,#cfeffd,#e7fbff); border-radius:12px; border:1px solid rgba(0,0,0,0.04);} 
.bubble {max-width:72%; padding:10px 12px; border-radius:12px; margin:8px 0; font-size:14px;}
.user {margin-left:auto; background:linear-gradient(135deg,#d6f7e0,#bff0c6);} 
.bot {margin-right:auto; background:#fff; border:1px solid rgba(0,0,0,0.03);} 
.meta {font-size:11px; color:#4b6b7a; margin-top:6px;}
</style>"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------- Init resources ----------------
@st.cache_resource(show_spinner=False)
def load_sbert_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# model variable kept to match earlier code
model = load_sbert_model()

# ---------------- Build/load FAISS index (DB-aware) ----------------
@st.cache_resource(show_spinner=False)
def build_or_load_index():
    """
    Build or load a FAISS index and metadata list.
    Priority:
      1) If DATABASE_URL present and SQLAlchemy available: fetch metadata rows (limited), build index from summaries, and persist meta locally.
      2) Else, if META_PATH exists: load it and build index.
      3) Else: fallback to a small set of sample docs.
    Returns: (index_or_None, meta_list)
    """
    meta_list: List[Dict] = []
    # 1) Try DB
    db_url = get_secret("DATABASE_URL") or os.getenv("DATABASE_URL")
    if db_url and DB_AVAILABLE:
        try:
            engine = sqlalchemy.create_engine(db_url, future=True)
            q = text("""
                SELECT file_url, file_name, coalesce(summary, '') AS summary, COALESCE(year, NULL) as year
                FROM netcdf_files
                ORDER BY id DESC
                LIMIT 5000
            """)
            with engine.connect() as conn:
                rows = conn.execute(q).fetchall()
            if rows:
                for r in rows:
                    rec = dict(r._mapping)
                    meta_list.append({
                        "id": rec.get("file_url"),
                        "file_name": rec.get("file_name"),
                        "file_url": rec.get("file_url"),
                        "summary": rec.get("summary") or "",
                        "year": rec.get("year")
                    })
                # persist a local copy for faster startup next time
                try:
                    with open(META_PATH, "w", encoding="utf-8") as f:
                        json.dump(meta_list, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
        except Exception as e:
            # don't raise; fallback to local
            st.warning(f"Could not fetch metadata from DB (falling back to local): {e}")
            meta_list = []

    # 2) If DB didn't populate meta_list, try file
    if not meta_list and os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                meta_list = json.load(f)
        except Exception:
            meta_list = []

    # 3) fallback sample
    if not meta_list:
        meta_list = [
            {"id": "R_sample_001", "file_name": "sample_2023_01.nc", "file_url": "local://sample_2023_01", "summary": "Salinity profile near 7N,75E recorded on 2023-03-15.", "year": 2023},
            {"id": "R_sample_002", "file_name": "sample_2023_02.nc", "file_url": "local://sample_2023_02", "summary": "Temperature profile near 10N,70E on 2023-02-10.", "year": 2023},
            {"id": "R_sample_003", "file_name": "sample_2023_03.nc", "file_url": "local://sample_2023_03", "summary": "BGC float near 5N,72E: chlorophyll elevated at surface.", "year": 2023},
        ]
        try:
            with open(META_PATH, "w", encoding="utf-8") as f:
                json.dump(meta_list, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # Build embeddings and FAISS index when possible
    texts = [m.get("summary", "") for m in meta_list]
    try:
        embs = model.encode(texts, show_progress_bar=False)
        arr = np.array(embs).astype("float32")
    except Exception as e:
        st.error(f"Failed to compute embeddings: {e}")
        return None, meta_list

    if FAISS_AVAILABLE:
        try:
            faiss.normalize_L2(arr)
            d = arr.shape[1]
            index = faiss.IndexFlatIP(d)
            index.add(arr)
            try:
                faiss.write_index(index, INDEX_PATH)
            except Exception:
                pass
            return index, meta_list
        except Exception as e:
            st.warning(f"FAISS build failed, will use numpy search instead: {e}")
            return None, meta_list
    else:
        # no FAISS: still return meta_list and None for index (we'll do numpy ranking)
        return None, meta_list

# build or load
index, meta = build_or_load_index()

# ---------------- Helpers ----------------

def _extract_text(resp):
    try:
        if hasattr(resp, "text") and resp.text:
            return resp.text
        if hasattr(resp, "candidates") and resp.candidates:
            c = resp.candidates[0]
            if hasattr(c, "text") and c.text:
                return c.text
        return str(resp)
    except Exception:
        return str(resp)


def generate_with_gemini(question, retrieved_texts):
    q_lower = (question or "").lower().strip()
    if q_lower in {"hi", "hello", "hey", "hey there", "hi there"}:
        return "Hello! I'm FloatChat. How can I help you today?"
    if MODEL_HANDLE is None:
        return "⚠️ Gemini not configured or not available. I can show retrieved summaries instead."

    context = "\n\n".join(f"[DOC {i+1}] {t.get('summary','')[:1000]}" for i, t in enumerate(retrieved_texts)) or ""
    prompt = f"{context}\n\nUser: {question}\nAnswer as FloatChat:"
    try:
        resp = MODEL_HANDLE.generate_content(prompt, generation_config={"max_output_tokens":300, "temperature":0.3})
        return _extract_text(resp)
    except Exception as e:
        return f"Gemini call failed: {e}"


def speak_text_tts(text):
    if not GTTS_AVAILABLE or not text or not text.strip():
        return None
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            path = tmp.name
        tts.save(path)
        data = open(path, "rb").read()
        os.remove(path)
        return data
    except Exception:
        return None


def build_messages_html(messages):
    parts = []
    for m in reversed(messages):
        ts = datetime.fromtimestamp(m.get("ts", time.time())).strftime("%I:%M %p")
        txt = html.escape(m.get("text", "") or "")
        cls = "user" if m["role"] == "user" else "bot"
        who = "You" if m["role"] == "user" else "FloatChat"
        parts.append(f'<div class="bubble {cls}"><div>{txt}</div><div class="meta">{who} • {ts}</div></div>')
    return "\n".join(parts)


def clear_conversation():
    st.session_state.messages = []

# ---------------- Ranking helpers (numpy fallback if no FAISS) ----------------

def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    embs = model.encode(texts, show_progress_bar=False)
    arr = np.array(embs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms
    return arr


def top_k_by_cosine(query: str, docs: List[Dict], k: int = 5):
    summaries = [d.get("summary") or "" for d in docs]
    if len(summaries) == 0:
        return []
    if FAISS_AVAILABLE and index is not None:
        # use FAISS for speed
        q_emb = model.encode([query], show_progress_bar=False)
        q_arr = np.array(q_emb, dtype=np.float32)
        faiss.normalize_L2(q_arr)
        D, I = index.search(q_arr, k)
        results = []
        for i, score in zip(I[0], D[0]):
            if i < 0 or i >= len(docs):
                continue
            results.append({"score": float(score), "doc": docs[i]})
        return results
    else:
        # numpy SBERT ranking
        doc_embs = embed_texts(summaries)
        q_emb = embed_texts([query])[0]
        sims = (doc_embs @ q_emb).astype(float)
        idx = np.argsort(-sims)[:k]
        results = []
        for i in idx:
            results.append({"score": float(sims[i]), "doc": docs[i]})
        return results

# ---------------- App Layout ----------------
st.set_page_config(page_title="FloatChat — RAG", layout="centered")
st.markdown('<div class="topbar">🌊 FLOATCHAT</div>', unsafe_allow_html=True)
if "messages" not in st.session_state:
    st.session_state.messages = []
messages_placeholder = st.empty()


def render_messages():
    messages_placeholder.markdown(f'<div class="messages">{build_messages_html(st.session_state.messages)}</div>', unsafe_allow_html=True)

render_messages()

with st.form("chat", clear_on_submit=True):
    cols = st.columns([8, 2])
    user_input = cols[0].text_input("Message", placeholder="Type your message...", key="text_input",
                                  label_visibility="collapsed")
    send = cols[1].form_submit_button("Send")
    if send and user_input.strip():
        text = user_input.strip()
        st.session_state.messages.append({"role": "user", "text": text, "ts": time.time()})

        # Build list of candidate docs from meta (already loaded)
        # We keep behavior simple: search the meta list (which was built from DB if available)
        candidates = meta[:]  # copy

        # If no candidates (shouldn't happen), show message
        if not candidates:
            bot_text = "No documents available to search."
        else:
            # get top docs by cosine
            top_docs = top_k_by_cosine(text, candidates, k=3)
            retrieved = [t['doc'] for t in top_docs]
            # Generate answer: Gemini if configured, else local synth
            if MODEL_HANDLE is not None and GEMINI_API_KEY:
                bot_text = generate_with_gemini(text, retrieved)
            else:
                # local synth: concatenate top summaries
                parts = [f"- {d.get('summary','')[:400]}..." for d in retrieved]
                bot_text = "I found the following top summaries:\n\n" + "\n\n".join(parts)

        st.session_state.messages.append({"role": "bot", "text": bot_text, "ts": time.time()})
        render_messages()
        mp3 = speak_text_tts(bot_text)
        if mp3:
            st.audio(mp3, format="audio/mp3")

if st.button("Clear Conversation"):
    clear_conversation()
    st.rerun()

st.markdown('<div class="footer">Powered by FloatChat 🌊</div>', unsafe_allow_html=True)

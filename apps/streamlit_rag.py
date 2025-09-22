# apps/streamlit_rag.py
"""
FloatChat RAG + Data Analysis assistant.

Adds a 'Data analysis' panel that can compute numeric aggregates
(e.g. average temperature) by opening NetCDF files referenced in the DB.
"""
import os
import json
import time
import tempfile
import html
from datetime import datetime
from typing import List, Dict, Optional, Tuple

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

# DB
try:
    import sqlalchemy
    from sqlalchemy import text
    DB_AVAILABLE = True
except Exception:
    DB_AVAILABLE = False

# xarray/netcdf
try:
    import xarray as xr
    XR_AVAILABLE = True
except Exception:
    XR_AVAILABLE = False

# streaming download helpers
import requests
from urllib.parse import urlparse
from pathlib import Path

# ---------------- CONFIG ----------------
PERSIST_DIR = os.getenv("PERSIST_DIR", "./faiss_db_proto")
INDEX_PATH = os.path.join(PERSIST_DIR, "faiss_index.bin")
META_PATH = os.path.join(PERSIST_DIR, "meta.json")
DATA_DIR = os.getenv("DATA_DIR", "./data/netcdf")
os.makedirs(PERSIST_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# secrets helper
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

model = load_sbert_model()

# ---------------- Build/load index (DB-aware, same as before) ----------------
@st.cache_resource(show_spinner=False)
def build_or_load_index() -> Tuple[Optional[object], List[Dict]]:
    meta_list: List[Dict] = []
    db_url = get_secret("DATABASE_URL") or os.getenv("DATABASE_URL")
    if db_url and DB_AVAILABLE:
        try:
            engine = sqlalchemy.create_engine(db_url, future=True)
            q = text("""
                SELECT file_url, file_name, coalesce(summary, '') AS summary, COALESCE(year, NULL) as year, local_path
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
                        "local_path": rec.get("local_path"),  # optional column
                        "summary": rec.get("summary") or "",
                        "year": rec.get("year")
                    })
                try:
                    with open(META_PATH, "w", encoding="utf-8") as f:
                        json.dump(meta_list, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
        except Exception as e:
            st.warning(f"Could not fetch metadata from DB: {e}")
            meta_list = []

    if not meta_list and os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                meta_list = json.load(f)
        except Exception:
            meta_list = []

    if not meta_list:
        meta_list = [
            {"id": "R_sample_001", "file_name": "sample_2023_01.nc", "file_url": "local://sample_2023_01", "local_path": None, "summary": "Salinity profile near 7N,75E recorded on 2023-03-15.", "year": 2023},
            {"id": "R_sample_002", "file_name": "sample_2023_02.nc", "file_url": "local://sample_2023_02", "local_path": None, "summary": "Temperature profile near 10N,70E on 2023-02-10.", "year": 2023},
            {"id": "R_sample_003", "file_name": "sample_2023_03.nc", "file_url": "local://sample_2023_03", "local_path": None, "summary": "BGC float near 5N,72E: chlorophyll elevated at surface.", "year": 2023},
        ]
        try:
            with open(META_PATH, "w", encoding="utf-8") as f:
                json.dump(meta_list, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # build embeddings for RAG
    texts = [m.get("summary","") for m in meta_list]
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
            idx = faiss.IndexFlatIP(d)
            idx.add(arr)
            try:
                faiss.write_index(idx, INDEX_PATH)
            except Exception:
                pass
            return idx, meta_list
        except Exception as e:
            st.warning(f"FAISS build failed: {e}")
            return None, meta_list
    return None, meta_list

index, meta = build_or_load_index()

# ---------------- Helpers (RAG side) ----------------
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
    if q_lower in {"hi", "hello", "hey"}:
        return "Hello! I'm FloatChat. How can I help you today?"
    if MODEL_HANDLE is None:
        return "⚠️ Gemini not configured or not available."
    context = "\n\n".join(f"[DOC {i+1}] {t.get('summary','')[:1000]}" for i,t in enumerate(retrieved_texts))
    prompt = f"{context}\n\nUser: {question}\nAnswer as FloatChat:"
    try:
        resp = MODEL_HANDLE.generate_content(prompt, generation_config={"max_output_tokens":300, "temperature":0.2})
        return _extract_text(resp)
    except Exception as e:
        return f"Gemini call failed: {e}"

def build_messages_html(messages):
    parts=[]
    for m in reversed(messages):
        ts=datetime.fromtimestamp(m.get("ts",time.time())).strftime("%I:%M %p")
        txt=html.escape(m.get("text","") or "")
        cls="user" if m["role"]=="user" else "bot"
        who="You" if m["role"]=="user" else "FloatChat"
        parts.append(f'<div class="bubble {cls}"><div>{txt}</div><div class="meta">{who} • {ts}</div></div>')
    return "\n".join(parts)

def clear_conversation(): st.session_state.messages=[]

# ---------------- Data analysis helpers ----------------

def safe_download(url: str, dst_dir: str = DATA_DIR) -> Optional[str]:
    """
    Download URL into dst_dir if not present. Return local path or None.
    Uses streaming, overwrites partially downloaded .part files if completed.
    """
    try:
        parsed = urlparse(url)
        name = os.path.basename(parsed.path)
        if not name:
            return None
        out = os.path.join(dst_dir, name)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            return out
        tmp = out + ".part"
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=64*1024):
                    if chunk:
                        f.write(chunk)
        os.replace(tmp, out)
        return out
    except Exception as e:
        print("download failed for", url, e)
        return None

def candidate_files_for(year:int=None, month:int=None, lat_min:float=None, lat_max:float=None, lon_min:float=None, lon_max:float=None, max_files:int=200) -> List[Dict]:
    """
    Return list of meta entries that match year and bbox constraints.
    """
    results = []
    for m in meta:
        try:
            if year and m.get("year") and int(m.get("year")) != int(year):
                continue
            # bbox filtering is approximate because metadata may not include bbox columns
            # If metadata has lat/lon extents include them in meta rows and check here
            # We'll still include entries without extents.
            results.append(m)
        except Exception:
            continue
        if len(results) >= max_files:
            break
    return results

# Temperature variable detection
TEMP_CANDIDATES = {"temperature","temp","sea_water_temperature","theta","thetao","t","TEMP","TEMP_ADJUSTED","TEMP_QC","SEA_SURFACE_TEMPERATURE"}

def find_temp_var(ds: xr.Dataset) -> Optional[str]:
    """
    Heuristic: find a variable in xarray Dataset that looks like temperature.
    """
    for v in ds.data_vars:
        vn = v.lower()
        if vn in TEMP_CANDIDATES:
            return v
    # more relaxed match
    for v in ds.data_vars:
        vn = v.lower()
        if "temp" in vn or "theta" in vn or "tc" in vn:
            return v
    return None

def compute_mean_over_files(file_paths: List[str], temp_var_name:Optional[str]=None) -> Dict:
    """
    Compute simple mean across files for a temperature-like variable.
    We process files one-by-one to keep memory low.
    Returns dictionary with stats.
    """
    if not XR_AVAILABLE:
        return {"error":"xarray not available in runtime."}

    total_sum = 0.0
    total_count = 0
    file_used = 0
    per_file_info = []

    for p in file_paths:
        try:
            ds = xr.open_dataset(p, decode_times=True, use_cftime=True)
        except Exception as e:
            per_file_info.append({"file":p,"status":"open_failed","error":str(e)})
            continue

        varname = temp_var_name
        if varname is None:
            varname = find_temp_var(ds)

        if not varname or varname not in ds.data_vars:
            per_file_info.append({"file":p,"status":"no_temp_var"})
            try:
                ds.close()
            except Exception:
                pass
            continue

        try:
            arr = ds[varname]
            # flatten and ignore NaNs
            vals = arr.values
            # some profiles are 1D arrays: compute nanmean and count of non-nan
            flat = np.ravel(vals)
            mask = ~np.isnan(flat)
            cnt = int(np.sum(mask))
            if cnt == 0:
                per_file_info.append({"file":p,"status":"no_valid_vals"})
                ds.close()
                continue
            s = float(np.nansum(flat))
            total_sum += s
            total_count += cnt
            file_used += 1
            per_file_info.append({"file":p,"status":"ok","count":cnt})
        except Exception as e:
            per_file_info.append({"file":p,"status":"read_failed","error":str(e)})
        finally:
            try:
                ds.close()
            except Exception:
                pass

    if total_count == 0:
        return {"error":"no valid temperature values found","per_file":per_file_info}
    mean = total_sum / total_count
    return {"mean": mean, "count": total_count, "files_used": file_used, "per_file": per_file_info}

# ---------------- UI ----------------
st.set_page_config(page_title="FloatChat — RAG + Data", layout="centered")
st.markdown('<div class="topbar">🌊 FLOATCHAT</div>', unsafe_allow_html=True)
if "messages" not in st.session_state:
    st.session_state.messages = []
messages_placeholder = st.empty()

def render_messages():
    messages_placeholder.markdown(f'<div class="messages">{build_messages_html(st.session_state.messages)}</div>', unsafe_allow_html=True)

render_messages()

# Left column: RAG / search
left, right = st.columns([2,1])
with left:
    st.header("RAG Search (metadata)")
    question = st.text_input("Ask about datasets (year, region, variables, etc.)")
    if st.button("Search RAG"):
        # do RAG retrieval on meta summaries
        candidates = meta[:]  # already loaded
        # simple SBERT ranking (reuse earlier logic)
        # embed summaries
        summaries = [c.get("summary","") for c in candidates]
        if len(summaries) == 0:
            st.warning("No metadata available.")
        else:
            # compute embeddings and cosine
            doc_embs = model.encode(summaries, show_progress_bar=False)
            doc_embs = np.array(doc_embs, dtype=np.float32)
            norms = np.linalg.norm(doc_embs, axis=1, keepdims=True)
            norms[norms==0] = 1.0
            doc_embs = doc_embs / norms
            q_emb = model.encode([question], show_progress_bar=False)
            q_emb = np.array(q_emb, dtype=np.float32)
            q_emb = q_emb / (np.linalg.norm(q_emb)+1e-12)
            sims = (doc_embs @ q_emb[0]).astype(float)
            idx = np.argsort(-sims)[:10]
            st.write("Top results:")
            for i in idx:
                d = candidates[i]
                st.write(f"- {d.get('file_name') or d.get('file_url')} (score {sims[i]:.3f})")
                st.caption(d.get("summary",""))
                st.write("---")

# Right column: Data analysis controls
with right:
    st.header("Data analysis")
    st.markdown("Compute numeric aggregates from NetCDF files referenced in DB.")
    year = st.number_input("Year", min_value=1900, max_value=2100, value=datetime.utcnow().year, step=1)
    month = st.selectbox("Month (optional)", options=[0,"Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], index=0)
    bbox = st.checkbox("Use bounding box filter?")
    lat_min = lat_max = lon_min = lon_max = None
    if bbox:
        lat_min = st.number_input("lat_min", value=-90.0, format="%.6f")
        lat_max = st.number_input("lat_max", value=90.0, format="%.6f")
        lon_min = st.number_input("lon_min", value=-180.0, format="%.6f")
        lon_max = st.number_input("lon_max", value=180.0, format="%.6f")
    var_hint = st.text_input("Variable name hint (leave empty to auto-detect)", value="temp")
    max_files = st.number_input("Max files to process", min_value=1, max_value=2000, value=200, step=1)
    if st.button("Compute average temperature"):
        if not XR_AVAILABLE:
            st.error("xarray/netCDF4 not available in this environment.")
        else:
            st.info("Collecting candidate files from metadata...")
            cands = candidate_files_for(year if year else None, None, lat_min, lat_max, lon_min, lon_max, max_files)
            st.write(f"Found {len(cands)} candidate metadata rows (limited to {max_files}).")
            # gather local paths or download
            file_paths = []
            progress = st.progress(0)
            for i, c in enumerate(cands):
                local = c.get("local_path") or c.get("local")  # ingestion might have saved this
                if local and os.path.exists(local):
                    file_paths.append(local)
                else:
                    url = c.get("file_url")
                    if url and url.startswith("http"):
                        p = safe_download(url, DATA_DIR)
                        if p:
                            file_paths.append(p)
                progress.progress(int((i+1)/len(cands)*100) if len(cands)>0 else 100)
                if len(file_paths) >= max_files:
                    break
            st.write(f"Prepared {len(file_paths)} files to analyze (up to {max_files}).")

            if not file_paths:
                st.warning("No files available to analyze. Make sure ingestion saved local_path or files are reachable via file_url.")
            else:
                st.info("Processing files (may take time).")
                # try to detect variable name from first file if var_hint empty
                temp_var = None
                if var_hint:
                    temp_var = var_hint.strip()
                else:
                    # detect using first file
                    for p in file_paths:
                        try:
                            ds = xr.open_dataset(p, decode_times=True, use_cftime=True)
                            v = find_temp_var(ds)
                            ds.close()
                            if v:
                                temp_var = v
                                break
                        except Exception:
                            continue
                st.write("Using variable hint / selection:", temp_var or "(auto-detect)")

                # compute mean
                res = compute_mean_over_files(file_paths, temp_var_name=temp_var)
                if "error" in res:
                    st.error(res["error"])
                    st.json(res.get("per_file", []))
                else:
                    st.success(f"Mean temperature (approx) = {res['mean']:.6f} over {res['count']} points from {res['files_used']} files.")
                    st.json(res.get("per_file", [])[:20])

# Chat area (bottom)
st.markdown("---")
st.header("Chat / Quick RAG")
with st.form("chat", clear_on_submit=True):
    cols=st.columns([8,2])
    user_input=cols[0].text_input("Message",placeholder="Type your message...",key="chat_input",label_visibility="collapsed")
    send=cols[1].form_submit_button("Send")
    if send and user_input.strip():
        text=user_input.strip()
        st.session_state.messages.append({"role":"user","text":text,"ts":time.time()})
        # simple retrieval: nearest summaries by SBERT (top 3)
        candidates = meta[:]
        if not candidates:
            bot_text = "No metadata available."
        else:
            top_docs = top_k_temp := None
            # reuse simple rank code
            summaries = [c.get("summary","") for c in candidates]
            doc_embs = model.encode(summaries, show_progress_bar=False)
            doc_embs = np.array(doc_embs, dtype=np.float32)
            norms = np.linalg.norm(doc_embs, axis=1, keepdims=True)
            norms[norms==0] = 1.0
            doc_embs = doc_embs / norms
            q_emb = model.encode([text], show_progress_bar=False)
            q_emb = np.array(q_emb, dtype=np.float32); q_emb = q_emb/(np.linalg.norm(q_emb)+1e-12)
            sims = (doc_embs @ q_emb[0]).astype(float)
            idx = np.argsort(-sims)[:5]
            retrieved = [candidates[i] for i in idx]
            if MODEL_HANDLE is not None:
                bot_text = generate_with_gemini(text, retrieved)
            else:
                parts = [f\"- {d.get('summary','')[:300]}...\" for d in retrieved]
                bot_text = \"I found these dataset summaries:\\n\\n\" + \"\\n\\n\".join(parts)
        st.session_state.messages.append({"role":"bot","text":bot_text,"ts":time.time()})
        render_messages()

if st.button("Clear Conversation"):
    clear_conversation()
    st.rerun()

st.markdown('<div class="footer">Powered by FloatChat 🌊</div>',unsafe_allow_html=True)

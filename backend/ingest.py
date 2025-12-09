# backend/ingest.py
from pathlib import Path
from typing import List, Dict, Any
import json
import re
import os

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

INDEX_LOCK = Lock()
# ---- Paths ----
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
INDEX_DIR = DATA_DIR / "index"

CHUNKS_PATH = INDEX_DIR / "chunks.json"
EMB_PATH = INDEX_DIR / "embeddings.npy"
PCA_PATH = INDEX_DIR / "pca_whitening.npz"

for d in (DATA_DIR, PDF_DIR, INDEX_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- Embedding model (free, local) ----
EMBED_MODEL = "BAAI/bge-base-en-v1.5"  # or "Alibaba-NLP/gte-base-en-v1.5"

# NOTE: if you have a GPU, uncomment device="cuda" to get a *huge* speed boost.
# st_model = SentenceTransformer(EMBED_MODEL, device="cuda")
st_model = SentenceTransformer(EMBED_MODEL)


# ---------- PDF TEXT + CHUNKING ----------

def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)

def prepare_pdf_for_index(pdf_path: Path):
    """
    Heavy part: read, extract, chunk, embed.
    This can be run in parallel for many PDFs.
    Returns (pdf_name, chunks_text, embeddings)
    """
    print(f"[ingest] Preparing {pdf_path.name}...")
    text = extract_text_from_pdf(pdf_path)
    chunks_text = chunk_by_paragraph(text)

    if not chunks_text:
        return pdf_path.name, [], None

    # Embed just this PDF's chunks
    emb = embed_texts(chunks_text)  # (m, d)
    return pdf_path.name, chunks_text, emb

def chunk_by_paragraph(
    text: str,
    max_chars: int = 1200,
    min_chars: int = 400,  # currently unused but kept for tuning
) -> List[str]:
    """
    Split text into paragraph-like units, then merge into chunks
    of roughly [min_chars, max_chars] characters.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 1) Try real paragraph split: one or more blank lines
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    # If we got basically nothing, fall back to line-based splitting:
    if len(paras) <= 2:
        paras = [line.strip() for line in text.split("\n") if line.strip()]

    chunks: List[str] = []
    buffer: List[str] = []
    buf_len = 0

    for para in paras:
        if buffer and buf_len + 1 + len(para) > max_chars:
            chunk = "\n".join(buffer).strip()
            if chunk:
                chunks.append(chunk)
            buffer = []
            buf_len = 0

        buffer.append(para)
        buf_len += len(para) + 1  # +1 for newline

    if buffer:
        chunk = "\n".join(buffer).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


# ---------- EMBEDDINGS + PCA ----------

def embed_texts(texts: List[str]) -> np.ndarray:
    emb = st_model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=False,
    )
    return emb.astype(np.float32)


def compute_pca_projection(
    embeddings: np.ndarray, target_dim: int = 256
) -> Dict[str, np.ndarray]:
    """
    Compute PCA projection matrix.
    For n >= 2: real PCA.
    For n < 2: fall back to identity projection (no PCA).
    """
    n, d = embeddings.shape

    # If we don't have enough samples for PCA, just use identity projection
    if n < 2:
        k = min(target_dim, d)
        mean = np.zeros((d,), dtype=np.float32)  # no centering
        W = np.eye(d, k, dtype=np.float32)       # (d, k) identity basis

        return {
            "mean": mean,
            "W": W,
            "k": np.array([k], dtype=np.int32),
            "d": np.array([d], dtype=np.int32),
        }

    # --- Normal PCA path for n >= 2 ---
    mean = embeddings.mean(axis=0)
    Xc = embeddings - mean

    cov = np.cov(Xc, rowvar=False, bias=True)

    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]

    k = min(target_dim, d)
    W = vecs[:, :k]

    return {
        "mean": mean.astype(np.float32),
        "W": W.astype(np.float32),
        "k": np.array([k], dtype=np.int32),
        "d": np.array([d], dtype=np.int32),
    }


# ---------- helpers for index file IO ----------

def _load_chunks() -> List[Dict[str, Any]]:
    """
    Safely load chunks.json.

    - Returns [] if file is missing or empty.
    - If file is corrupted (partial write or multiple JSON documents),
      we back it up and reset to [] so the app keeps working.
    """
    if not CHUNKS_PATH.exists():
        return []

    raw = CHUNKS_PATH.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        # Try to salvage: sometimes there are multiple JSON arrays concatenated
        print(f"[ingest] WARNING: chunks.json is corrupted ({e}). "
              f"Backing it up and resetting to empty list.")

        backup_path = CHUNKS_PATH.with_suffix(".corrupt.json")
        try:
            backup_path.write_text(raw, encoding="utf-8")
        except Exception as be:
            print(f"[ingest] Failed to write backup: {be}")

        # Reset the file to a valid empty list
        CHUNKS_PATH.write_text("[]", encoding="utf-8")
        return []


def _save_chunks(chunks: List[Dict[str, Any]]) -> None:
    """
    Atomically save chunks.json so we don't get partial writes
    while FastAPI background tasks are running.
    """
    tmp_path = CHUNKS_PATH.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    tmp_path.replace(CHUNKS_PATH)

def load_chunks_for_pdf(pdf_name: str) -> List[Dict[str, Any]]:
    """
    Return all chunk records for a single PDF.
    Each record has at least: id, source_pdf, chunk_index, text.
    """
    chunks = _load_chunks()
    return [c for c in chunks if c.get("source_pdf") == pdf_name]


def _save_full_index(
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray | None
) -> Dict[str, Any]:
    """
    Save chunks.json, embeddings.npy, and pca_whitening.npz
    and return metadata.
    """
    if embeddings is None:
        # Nothing embedded yet; just write chunks + empty embedding array
        _save_chunks(chunks)
        np.save(EMB_PATH, np.zeros((0, 0), dtype=np.float32))
        return {
            "total_chunks": 0,
            "per_pdf_counts": {},
        }

    total_chunks = len(chunks)

    # Save chunk metadata
    _save_chunks(chunks)

    # Save embeddings
    np.save(EMB_PATH, embeddings.astype(np.float32))

    # PCA
    print("[ingest] Computing PCA...")
    pca = compute_pca_projection(embeddings, target_dim=256)
    np.savez(
        PCA_PATH,
        mean=pca["mean"],
        W=pca["W"],
        k=pca["k"],
        d=pca["d"],
    )
    print("[ingest] Index save complete.")

    # per-PDF counts
    per_pdf_counts: Dict[str, int] = {}
    for rec in chunks:
        name = rec["source_pdf"]
        per_pdf_counts[name] = per_pdf_counts.get(name, 0) + 1

    return {
        "total_chunks": total_chunks,
        "per_pdf_counts": per_pdf_counts,
    }


# ---------- FULL REBUILD: parallel over PDFs ----------

def _extract_and_chunk_single(pdf_path: Path) -> Dict[str, Any]:
    """
    Worker function: read + chunk a single PDF.
    Used in ThreadPoolExecutor for parallelism.
    """
    print(f"[ingest] Reading {pdf_path.name}...")
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_by_paragraph(text)
    return {
        "pdf_name": pdf_path.name,
        "chunks": chunks,
    }


def build_index_from_pdfs(pdf_paths: list[Path]) -> Dict[str, Any]:
    """
    Full rebuild from a given list of PDFs:
    - prepare (chunk + embed) in parallel
    - commit once
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = []
    with ThreadPoolExecutor(max_workers=min(4, len(pdf_paths))) as ex:
        futures = {ex.submit(prepare_pdf_for_index, p): p for p in pdf_paths}
        for fut in as_completed(futures):
            pdf_name, chunks_text, emb = fut.result()
            if chunks_text and emb is not None:
                results.append((pdf_name, chunks_text, emb))

    if not results:
        return {"total_chunks": 0, "per_pdf_counts": {}}

    with INDEX_LOCK:
        existing = _load_chunks()
        if EMB_PATH.exists():
            old_emb = np.load(EMB_PATH)
        else:
            old_emb = None

        next_id = len(existing)
        all_chunks = list(existing)
        emb_list = [] if old_emb is None or old_emb.size == 0 else [old_emb]

        for pdf_name, chunks_text, emb in results:
            for i, chunk in enumerate(chunks_text):
                all_chunks.append(
                    {
                        "id": next_id,
                        "source_pdf": pdf_name,
                        "chunk_index": i,
                        "text": chunk,
                    }
                )
                next_id += 1
            emb_list.append(emb)

        embeddings = np.vstack(emb_list)
        meta = _save_full_index(all_chunks, embeddings)

    return meta


# ================== INCREMENTAL INDEX + METADATA ==================

def get_index_metadata() -> Dict[str, Any]:
    """
    Read chunks.json and derive:
      - total_chunks
      - per_pdf_counts: {filename: count}

    If the index doesn't exist yet, or the file is empty / invalid,
    return zeros instead of raising.
    """
    if not CHUNKS_PATH.exists() or CHUNKS_PATH.stat().st_size == 0:
        # Index not built yet
        return {
            "total_chunks": 0,
            "per_pdf_counts": {},
        }

    try:
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except json.JSONDecodeError:
        # File exists but is not valid JSON yet (e.g. write interrupted)
        return {
            "total_chunks": 0,
            "per_pdf_counts": {},
        }

    per_pdf_counts: Dict[str, int] = {}
    for rec in chunks:
        pdf = rec["source_pdf"]
        per_pdf_counts[pdf] = per_pdf_counts.get(pdf, 0) + 1

    return {
        "total_chunks": len(chunks),
        "per_pdf_counts": per_pdf_counts,
    }



def build_index_incremental_for_pdf(pdf_path: Path) -> Dict[str, Any]:
    """
    Incrementally update the index for ONE PDF.

    If a PDF with the same filename is already in the index, we skip
    re-ingesting it to avoid duplicating all of its chunks.
    """
    chunks_path = INDEX_DIR / "chunks.json"
    emb_path = INDEX_DIR / "embeddings.npy"

    # ---- Step 1: load existing chunks + embeddings (if any) ----
    if chunks_path.exists() and emb_path.exists():
        with open(chunks_path, "r", encoding="utf-8") as f:
            existing_chunks = json.load(f)  # list[dict]
        old_embeddings = np.load(emb_path)
    else:
        existing_chunks = []
        old_embeddings = None

    # ✅ NEW: if this PDF is already present, do NOT append another copy
    already_indexed = any(
        rec.get("source_pdf") == pdf_path.name for rec in existing_chunks
    )
    if already_indexed:
        print(f"[ingest] {pdf_path.name} already indexed, skipping re-ingest.")
        # Just return current metadata
        return get_index_metadata()

    next_id_start = len(existing_chunks)

    # ---- Step 2: extract + chunk this one PDF ----
    text = extract_text_from_pdf(pdf_path)
    new_chunks = chunk_by_paragraph(text)

    if not new_chunks:
        # no new chunks, return current metadata
        return get_index_metadata() if chunks_path.exists() else {
            "total_chunks": 0,
            "per_pdf_counts": {}
        }

    records = []
    for i, chunk in enumerate(new_chunks):
        rec = {
            "id": next_id_start + i,
            "source_pdf": pdf_path.name,
            "chunk_index": i,
            "text": chunk,
        }
        records.append(rec)

    all_chunks = existing_chunks + records

    # ---- Step 3: embed ONLY the new chunks, then stack ----
    new_emb = embed_texts(new_chunks)  # (m, d)
    if old_embeddings is not None:
        embeddings = np.vstack([old_embeddings, new_emb])
    else:
        embeddings = new_emb

    # ---- Step 4: save updated chunks + embeddings ----
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    np.save(emb_path, embeddings.astype(np.float32))

    # ---- Step 5: recompute PCA ----
    pca = compute_pca_projection(embeddings, target_dim=256)
    np.savez(
        INDEX_DIR / "pca_whitening.npz",
        mean=pca["mean"],
        W=pca["W"],
        k=pca["k"],
        d=pca["d"],
    )

    # ---- Step 6: return fresh metadata ----
    return get_index_metadata()


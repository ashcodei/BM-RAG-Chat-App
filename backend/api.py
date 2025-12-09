# backend/api.py
import os
from typing import List, Optional, Dict
import json
from ingest import get_index_metadata
import numpy as np
import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import dotenv

dotenv.load_dotenv()

from ingest import PDF_DIR, build_index_incremental_for_pdf, get_index_metadata, build_index_from_pdfs, load_chunks_for_pdf
from rag_engine import RAGEngine

# -------- Embedding model (same as ingest) --------
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
st_model = SentenceTransformer(EMBED_MODEL)

# -------- OpenRouter config --------
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("Please set OPENROUTER_API_KEY in your environment (.env)")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "openai/gpt-oss-20b:free"

OPENROUTER_HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": os.environ.get("OPENROUTER_SITE_URL", "http://localhost"),
    "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "Math RAG"),
}

# -------- FastAPI app --------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_engine: Optional[RAGEngine] = None
INDEX_BUILD_IN_PROGRESS: bool = False  # <--- new: avoid overlapping rebuilds


def try_load_engine() -> None:
    """
    (Re)initialize the global RAG engine from whatever index is on disk.
    """
    global rag_engine
    try:
        rag_engine = RAGEngine()
        print("[api] RAGEngine initialized with existing index.")
    except Exception as e:
        print(f"[api] No usable index yet: {e}")
        rag_engine = None


# Try to load an index if it already exists (e.g., from a previous run)
try_load_engine()


class IndexStatus(BaseModel):
    total_chunks: int
    per_pdf_counts: Dict[str, int]
    has_index: bool

class ChatRequest(BaseModel):
    query: str
    # list of PDF filenames attached to this message
    sources: Optional[List[str]] = None


class RetrievedChunk(BaseModel):
    id: int
    score: float
    text: str
    source_pdf: str
    chunk_index: int


class ChatResponse(BaseModel):
    answer: str
    retrieved: List[RetrievedChunk]

class FileChunksResponse(BaseModel):
    source_pdf: str
    chunks: List[RetrievedChunk] | List[dict]  # keep it loose if you want
from typing import List, Optional, Literal

class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    query: str
    sources: Optional[List[str]] = None
    history: Optional[List[HistoryMessage]] = None
def embed_query(query: str) -> np.ndarray:
    # BGE-style query instruction
    q_instruct = "Represent this sentence for searching relevant passages: " + query
    v = st_model.encode([q_instruct], normalize_embeddings=False)[0]
    return v.astype(np.float32)


def build_system_prompt(retrieved_chunks: List[RetrievedChunk]) -> str:
    if not retrieved_chunks:
        return (
            "You are a question-answering assistant, but no context was retrieved. "
            "If you don't know the answer from general knowledge, say you don't know."
        )

    context_parts = []
    for c in retrieved_chunks:
        context_parts.append(
            f"[Source: {c.source_pdf}, chunk {c.chunk_index}, score={c.score:.4f}]\n"
            f"{c.text}"
        )
    context_str = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        "You are a question-answering assistant for one or more technical PDFs.\n"
        "Use ONLY the provided context chunks to answer the user's question.\n"
        "If the context does not contain the answer, say that you don't know.\n\n"
        "Context:\n"
        f"{context_str}\n\n"
        "When answering, be concise but precise."
    )
    return system_prompt


def rebuild_index_background() -> None:
    """
    Heavy RAG index rebuild, run as a FastAPI BackgroundTask.

    It still calls build_index_from_pdfs(), but now it runs AFTER the HTTP
    response is returned, so the upload endpoint feels 'instant'.
    """
    global INDEX_BUILD_IN_PROGRESS

    if INDEX_BUILD_IN_PROGRESS:
        print("[api] Index build already in progress; skipping new rebuild request.")
        return

    INDEX_BUILD_IN_PROGRESS = True
    try:
        print("[api] Starting background index rebuild...")
        info = build_index_from_pdfs()
        print(
            f"[api] Index rebuild complete: total_chunks={info.get('total_chunks')} "
            f"per_pdf_counts={info.get('per_pdf_counts')}"
        )
        try_load_engine()
    except Exception as e:
        print(f"[api] Background index rebuild failed: {e}")
    finally:
        INDEX_BUILD_IN_PROGRESS = False


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    global rag_engine

    # Do we actually want to use PDFs for this turn?
    use_docs = bool(req.sources)  # True only if list and non-empty
    retrieved_models: List[RetrievedChunk] = []

    if use_docs:
        if rag_engine is None:
            raise HTTPException(
                status_code=400,
                detail="RAG index not built yet. Upload at least one PDF and attach it to your message.",
            )

        # 1) Embed query in original embedding space
        q_raw = embed_query(req.query)  # (768,)

        # 2) Project into PCA-whitened space
        q_vec = rag_engine.embed_query_vec(q_raw)  # (k,)

        # 3) Search only among allowed sources (the attached PDFs)
        retrieved = rag_engine.search(
            q_vec,
            top_k=3,
            candidate_k=20,
            use_rocchio=True,
            feedback_k=5,
            alpha=1.0,
            beta=0.75,
            allowed_sources=req.sources,
        )
        retrieved_models = [RetrievedChunk(**r) for r in retrieved]

    # Build system prompt from whatever we retrieved (possibly empty)
    system_prompt = build_system_prompt(retrieved_models)

    # Build chat history messages (same as we added before)
    messages = [{"role": "system", "content": system_prompt}]

    if req.history:
        for m in req.history:
            content = (m.content or "").strip()
            if not content:
                continue
            role = "assistant" if m.role == "assistant" else "user"
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": req.query})

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": 0.2,
    }

    try:
        resp = requests.post(
            OPENROUTER_URL,
            headers=OPENROUTER_HEADERS,
            json=payload,
            timeout=60,
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"OpenRouter error: {e}")

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"OpenRouter returned {resp.status_code}: {resp.text}",
        )

    data = resp.json()
    try:
        answer = data["choices"][0]["message"]["content"]
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Unexpected OpenRouter response format: {e}; data={data}",
        )

    return ChatResponse(answer=answer, retrieved=retrieved_models)

@app.get("/index_status")
def index_status():
    try:
        meta = get_index_metadata()
        return meta
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "total_chunks": 0,
            "per_pdf_counts": {},
        }
    
from fastapi import Query

@app.get("/file_chunks")
def file_chunks(pdf: str = Query(..., description="PDF filename")):
    """
    Return all chunks for a single PDF (by filename).
    Used by the 'All Contexts' view to show every chunk.
    """
    try:
        chunks = load_chunks_for_pdf(pdf)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Index not built yet.")

    return {
        "source_pdf": pdf,
        "chunks": chunks,
    }

@app.post("/upload_pdf")
async def upload_pdf(
    background: BackgroundTasks,
    file: UploadFile = File(...),
):
    global rag_engine

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")

    dest_path = PDF_DIR / file.filename
    contents = await file.read()
    with open(dest_path, "wb") as f:
        f.write(contents)

    # Run the incremental index update in the background so the response is fast.
    background.add_task(build_index_incremental_for_pdf, dest_path)
    # After that, we also want to refresh the RAG engine.
    background.add_task(try_load_engine)

    # We don't yet know the final chunk count (it's being computed),
    # so we return nulls and let the frontend poll /index_status.
    return {
        "status": "ok",
        "filename": file.filename,
        "total_chunks": None,
        "pdf_chunks": {file.filename: None},
    }



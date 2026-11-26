"""FastAPI routes for the company knowledge assistant."""

import os
import json
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.rag import answer_question, load_vectorstore, sync_vectorstore

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_FILE = BASE_DIR / "index.html"

app = FastAPI(title="Advanced Impactor Knowledge Assistant", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _initialize_index():
    if os.getenv("AUTO_SYNC_INDEX", "true").lower() in {"1", "true", "yes"}:
        return sync_vectorstore()
    return load_vectorstore()


try:
    vectorstore = _initialize_index()
except Exception as exc:
    raise RuntimeError(
        "Knowledge index initialization failed. Check GOOGLE_API_KEY and source files; "
        "run `python -m app.rag --sync` to rebuild the index. "
        f"Details: {exc}"
    ) from exc


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class Query(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[ConversationMessage] = Field(default_factory=list, max_length=20)
    category: str | None = None
    product: str | None = None
    verification_status: str | None = None


@app.get("/")
def home():
    return FileResponse(INDEX_FILE)


@app.get("/health")
def health():
    manifest = BASE_DIR / "vectorstore" / "manifest.json"
    details = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    return {
        "status": "ok",
        "index_manifest": manifest.exists(),
        "indexed_sources": len(details.get("sources", {})),
        "source_errors": len(details.get("source_errors", [])),
    }


@app.get("/knowledge-filters")
def knowledge_filters():
    values = {"category": set(), "product": set(), "verification_status": set()}
    for doc_id in vectorstore.index_to_docstore_id.values():
        doc = vectorstore.docstore.search(doc_id)
        if not hasattr(doc, "metadata"):
            continue
        for field in values:
            item = doc.metadata.get(field)
            if item:
                values[field].add(str(item))
    return {key: sorted(items) for key, items in values.items()}


@app.post("/ask")
def ask(q: Query):
    try:
        if not q.question.strip():
            raise HTTPException(status_code=422, detail="Question cannot be blank.")
        history = [item.model_dump() for item in q.history]
        filters = {key: value for key, value in {
            "category": q.category,
            "product": q.product,
            "verification_status": q.verification_status,
        }.items() if value}
        return answer_question(q.question.strip(), vectorstore, history=history, filters=filters)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"The AI request failed: {exc}") from exc

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel


from app.rag import (
    load_documents,
    clean_documents,
    split_docs,
    filter_chunks,
    create_vectorstore,
    load_vectorstore,
    create_rag_chain
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later you can restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_FILE = BASE_DIR / "index.html"


# LOAD OR CREATE VECTOR STORE
try:
    vectorstore = load_vectorstore()
except Exception as exc:
    print(f"Could not load saved vector store ({exc}); building it from data files.")
    docs = load_documents()
    cleaned = clean_documents(docs)
    chunks = split_docs(cleaned)
    filtered = filter_chunks(chunks)
    vectorstore = create_vectorstore(filtered)

rag_chain = create_rag_chain(vectorstore)


class Query(BaseModel):
    question: str


@app.get("/")
def home():
    return FileResponse(INDEX_FILE)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(q: Query):
    try:
        answer = rag_chain.invoke(q.question)
        return {"answer": answer}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"The AI request failed: {exc}") from exc

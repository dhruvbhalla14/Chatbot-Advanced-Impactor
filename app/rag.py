"""Document loading, incremental FAISS indexing, hybrid retrieval and RAG answers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from zipfile import ZipFile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pptx import Presentation
from lxml import etree

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_DIR = BASE_DIR / "company_knowledge"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
MANIFEST_PATH = VECTORSTORE_DIR / "manifest.json"
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5.4-mini")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 140
MAX_CONTEXT_CHARS = 14000
MAX_RETRIEVAL = 8
EMBEDDING_BATCH_SIZE = 50
EMBEDDING_PAUSE_SECONDS = float(os.getenv("EMBEDDING_PAUSE_SECONDS", "1"))
LOAD_ERRORS: list[dict[str, str]] = []

SYSTEM_PROMPT = """You are the Advanced Impactor company knowledge assistant.
Answer from the supplied company documents only. Do not invent facts, fill gaps with general knowledge, or present recommendations as company statements.
Treat document status as meaningful: VERIFIED_PUBLIC is supported by a cited public source; INTERNAL_CONTEXT comes from internal/user-provided material; HISTORICAL may have changed; ANALYTICAL is interpretation; UNVERIFIED needs confirmation. State these distinctions when relevant, especially for launches, partnerships, architecture, metrics, and product relationships.
If documents conflict, explain the conflict and cite both. If evidence is missing, say you do not have enough documented information.
Explain technical components in plain language, including what they do and how they relate. Be concise unless the user requests detail. Keep conversational context for follow-ups.
Cite claims using only the provided source labels such as [S1] and [S2]. Never fabricate a citation. When useful, identify the source document and section in the answer."""


def _relative(path: Path) -> str:
    return path.resolve().relative_to(BASE_DIR).as_posix()


def discover_sources() -> list[Path]:
    """Find supported knowledge and legacy source documents deterministically."""
    paths: list[Path] = []
    for root in (KNOWLEDGE_DIR, DATA_DIR):
        if root.exists():
            paths.extend(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".md", ".pdf", ".txt", ".docx", ".pptx"})
    return sorted(paths, key=lambda p: _relative(p).casefold())


def _read_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end < 0:
        return {}, text
    raw = text[4:end]
    try:
        values = yaml.safe_load(raw) or {}
    except yaml.YAMLError:
        values = {}
    return (values if isinstance(values, dict) else {}), text[end + 4:].lstrip("\r\n")


def _base_metadata(path: Path, frontmatter: dict[str, Any] | None = None) -> dict[str, Any]:
    fm = frontmatter or {}
    rel = _relative(path)
    keywords = fm.get("keywords", [])
    if isinstance(keywords, str):
        keywords = [keywords]
    return {
        "source": rel,
        "document_title": str(fm.get("document_title") or path.stem.replace("_", " ")),
        "category": str(fm.get("category") or ("knowledge" if "company_knowledge" in rel else "source_material")),
        "product": str(fm.get("product") or "company"),
        "source_type": str(fm.get("source_type") or ("structured_knowledge" if path.suffix.lower() == ".md" else "internal_document")),
        "verification_status": str(fm.get("verification_status") or "internal_context"),
        "last_updated": str(fm.get("last_updated") or "unknown"),
        "keywords": ", ".join(str(k) for k in keywords),
        "source_urls": json.dumps(fm.get("source_urls", []), ensure_ascii=False),
    }


def _markdown_sections(body: str, title: str) -> list[tuple[str, str]]:
    headings = list(re.finditer(r"(?m)^(#{1,3})\s+(.+?)\s*$", body))
    if not headings:
        return [("Introduction", f"# {title}\n\n{body.strip()}")]
    sections: list[tuple[str, str]] = []
    prefix = body[: headings[0].start()].strip()
    if prefix:
        sections.append(("Introduction", f"# {title}\n\n{prefix}"))
    for i, match in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
        heading = match.group(2).strip()
        content = body[match.start():end].strip()
        sections.append((heading, f"# {title}\n\n{content}"))
    return sections


def load_source(path: Path) -> list[Document]:
    """Load a source file while retaining document and section attribution."""
    suffix = path.suffix.lower()
    if suffix == ".md":
        text = path.read_text(encoding="utf-8-sig")
        frontmatter, body = _read_frontmatter(text)
        metadata = _base_metadata(path, frontmatter)
        title = metadata["document_title"]
        return [Document(page_content=content, metadata={**metadata, "section": section})
                for section, content in _markdown_sections(body, title) if content.strip()]
    if suffix == ".txt":
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        return [Document(page_content=text, metadata={**_base_metadata(path), "section": "Document"})]
    if suffix == ".pdf":
        docs = PyPDFLoader(str(path)).load()
    elif suffix == ".docx":
        try:
            docs = Docx2txtLoader(str(path)).load()
        except Exception:
            # Some Word exports contain illegal XML control bytes. Recover readable
            # paragraph text without modifying the source file.
            with ZipFile(path) as archive:
                xml = archive.read("word/document.xml")
            root = etree.fromstring(xml, parser=etree.XMLParser(recover=True))
            namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
            paragraphs = []
            for paragraph in root.iter(f"{namespace}p"):
                text = "".join(node.text or "" for node in paragraph.iter(f"{namespace}t")).strip()
                if text:
                    paragraphs.append(text)
            if not paragraphs:
                raise
            docs = [Document(page_content="\n".join(paragraphs), metadata={})]
    elif suffix == ".pptx":
        deck = Presentation(str(path))
        docs = []
        for slide_number, slide in enumerate(deck.slides, 1):
            parts = []
            for shape in slide.shapes:
                if getattr(shape, "has_text_frame", False) and shape.text.strip():
                    parts.append(shape.text.strip())
                if getattr(shape, "has_table", False):
                    parts.extend(" | ".join(cell.text.strip() for cell in row.cells) for row in shape.table.rows)
            if parts:
                docs.append(Document(page_content="\n".join(parts), metadata={"slide": slide_number}))
    else:
        return []
    base = _base_metadata(path)
    result = []
    for index, doc in enumerate(docs):
        page = doc.metadata.get("page")
        slide = doc.metadata.get("slide")
        section = f"Page {int(page) + 1}" if isinstance(page, int) else f"Slide {int(slide)}" if isinstance(slide, int) else f"Section {index + 1}"
        result.append(Document(page_content=doc.page_content, metadata={**base, **doc.metadata, "source": _relative(path), "document_title": base["document_title"], "section": section}))
    return result


def load_documents(paths: Iterable[Path] | None = None, *, strict: bool = False) -> list[Document]:
    documents: list[Document] = []
    for path in paths or discover_sources():
        try:
            loaded = load_source(path)
            documents.extend(loaded)
            print(f"Loaded {len(loaded)} section(s): {_relative(path)}")
        except Exception as exc:
            message = f"Could not read source {_relative(path)}: {exc}"
            LOAD_ERRORS.append({"source": _relative(path), "error": str(exc)})
            if strict:
                raise RuntimeError(message) from exc
            print(f"WARNING {message}; skipping this source for now.")
    return documents


def clean_documents(documents: Iterable[Document]) -> list[Document]:
    cleaned: list[Document] = []
    seen: set[tuple[str, str, str]] = set()
    for doc in documents:
        text = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", doc.page_content)).strip()
        if not text:
            continue
        key = (str(doc.metadata.get("source", "")), str(doc.metadata.get("section", "")), text)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(Document(page_content=text, metadata=dict(doc.metadata)))
    return cleaned


def split_docs(docs: Iterable[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n### ", "\n## ", "\n# ", "\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(list(docs))
    for i, chunk in enumerate(chunks):
        source = str(chunk.metadata.get("source", "unknown"))
        start = chunk.metadata.get("start_index", i)
        chunk.metadata["chunk_id"] = hashlib.sha256(f"{source}:{start}:{chunk.page_content}".encode("utf-8")).hexdigest()
    return chunks


def _source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_uses_current_embeddings(manifest: dict[str, Any]) -> bool:
    return (
        manifest.get("embedding_provider") == "openai"
        and manifest.get("embedding_model") == EMBEDDING_MODEL
    )


def _embeddings() -> OpenAIEmbeddings:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Add a valid OpenAI API key to .env before indexing or starting the API.")
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def sync_vectorstore() -> FAISS:
    """Update changed sources, delete stale chunks and persist a source manifest."""
    LOAD_ERRORS.clear()
    paths = discover_sources()
    if not paths:
        raise RuntimeError(f"No Markdown, PDF, TXT, DOCX or PPTX sources found under {DATA_DIR} or {KNOWLEDGE_DIR}.")
    current = {_relative(path): (path, _source_hash(path)) for path in paths}
    old_manifest: dict[str, Any] = {}
    if MANIFEST_PATH.exists():
        old_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    source_map: dict[str, Any] = old_manifest.get("sources", {})
    index_exists = (VECTORSTORE_DIR / "index.faiss").exists()
    legacy_exists = index_exists and not MANIFEST_PATH.exists()
    embedding_changed = index_exists and not _manifest_uses_current_embeddings(old_manifest)
    rebuild_index = legacy_exists or embedding_changed
    vectorstore: FAISS | None = None
    embeddings = _embeddings()
    if not rebuild_index and index_exists:
        vectorstore = FAISS.load_local(str(VECTORSTORE_DIR), embeddings, allow_dangerous_deserialization=True)

    if rebuild_index:
        reason = "embedding model changed" if embedding_changed else "legacy index has no source manifest"
        print(f"FAISS {reason}; rebuilding all vectors with {EMBEDDING_MODEL}.")
        vectorstore = None
        source_map = {}
        changed = set(current)
        removed = set()
    else:
        removed = set(source_map) - set(current)
        changed = {name for name, (_, digest) in current.items() if source_map.get(name, {}).get("sha256") != digest}
        changed.update(name for name, record in source_map.items() if name in current and not record.get("ids"))
        LOAD_ERRORS.extend(
            item for item in old_manifest.get("source_errors", [])
            if item.get("source") not in removed | changed
        )
        to_delete: list[str] = []
        for name in removed | changed:
            to_delete.extend(source_map.get(name, {}).get("ids", []))
        if vectorstore is not None and to_delete:
            vectorstore.delete(ids=to_delete)

    docs_by_source: dict[str, list[Document]] = {}
    if changed:
        changed_paths = [current[name][0] for name in sorted(changed)]
        loaded = clean_documents(load_documents(changed_paths))
        for chunk in split_docs(loaded):
            docs_by_source.setdefault(str(chunk.metadata["source"]), []).append(chunk)

    updated_map = {name: record for name, record in source_map.items() if name not in removed | changed}
    documents_to_embed: list[Document] = []
    ids_to_embed: list[str] = []
    for source, docs in docs_by_source.items():
        digest = current[source][1]
        ids = [hashlib.sha256(f"{source}:{digest}:{i}:{doc.page_content}".encode("utf-8")).hexdigest() for i, doc in enumerate(docs)]
        for doc, chunk_id in zip(docs, ids):
            doc.metadata["chunk_id"] = chunk_id
        documents_to_embed.extend(docs)
        ids_to_embed.extend(ids)
        updated_map[source] = {"sha256": digest, "ids": ids}

    import time
    for offset in range(0, len(documents_to_embed), EMBEDDING_BATCH_SIZE):
        batch_docs = documents_to_embed[offset:offset + EMBEDDING_BATCH_SIZE]
        batch_ids = ids_to_embed[offset:offset + EMBEDDING_BATCH_SIZE]
        try:
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch_docs, embeddings, ids=batch_ids)
            else:
                vectorstore.add_documents(batch_docs, ids=batch_ids)
        except Exception as exc:
            first_source = batch_docs[0].metadata.get("source", "unknown")
            raise RuntimeError(
                f"OpenAI embedding failed near {first_source}, chunk batch starting at {offset}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        next_offset = offset + EMBEDDING_BATCH_SIZE
        if next_offset < len(documents_to_embed) and EMBEDDING_PAUSE_SECONDS > 0:
            print(f"Embedded {min(next_offset, len(documents_to_embed))}/{len(documents_to_embed)} chunks; pausing {EMBEDDING_PAUSE_SECONDS:g}s for API quota.", flush=True)
            time.sleep(EMBEDDING_PAUSE_SECONDS)

    for source in changed - set(docs_by_source):
        updated_map[source] = {"sha256": current[source][1], "ids": []}

    if vectorstore is None:
        raise RuntimeError("The source files contained no indexable text.")
    for source in current:
        if source not in updated_map:
            updated_map[source] = source_map[source]
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))
    tmp_manifest = MANIFEST_PATH.with_suffix(".tmp")
    tmp_manifest.write_text(json.dumps({
        "version": 3,
        "embedding_provider": "openai",
        "embedding_model": EMBEDDING_MODEL,
        "sources": updated_map,
        "source_errors": LOAD_ERRORS,
    }, indent=2), encoding="utf-8")
    tmp_manifest.replace(MANIFEST_PATH)
    print(f"Index synchronized: {len(changed)} changed/new source(s), {len(removed)} removed source(s), {len(updated_map)} total.")
    return vectorstore


def load_vectorstore(*, lexical_only: bool = False) -> FAISS:
    if not (VECTORSTORE_DIR / "index.faiss").exists():
        raise FileNotFoundError("No FAISS index exists. Run `python -m app.rag --sync` first.")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.exists() else {}
    if not lexical_only and not _manifest_uses_current_embeddings(manifest):
        raise RuntimeError(
            f"The FAISS index does not use {EMBEDDING_MODEL}. Run `python -m app.rag --sync` to rebuild it."
        )
    embeddings = FakeEmbeddings(size=1) if lexical_only else _embeddings()
    return FAISS.load_local(str(VECTORSTORE_DIR), embeddings, allow_dangerous_deserialization=True)


def _all_indexed_documents(vectorstore: FAISS) -> list[Document]:
    documents: list[Document] = []
    for doc_id in vectorstore.index_to_docstore_id.values():
        doc = vectorstore.docstore.search(doc_id)
        if isinstance(doc, Document):
            documents.append(doc)
    return documents


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)?", re.I)
_STOPWORDS = {"a", "an", "the", "is", "are", "was", "were", "what", "how", "does", "do", "of", "to", "and", "in", "for", "with", "on", "it", "this", "that", "explain", "tell", "me", "about", "company", "please"}


def _tokens(text: str) -> list[str]:
    values = []
    for raw in _TOKEN_RE.findall(text):
        token = raw.lower()
        if token in _STOPWORDS:
            continue
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        values.append(token)
    return values


def bm25_search(query: str, documents: list[Document], k: int = 12) -> list[Document]:
    """Small dependency-free BM25 ranker used alongside FAISS retrieval."""
    query_terms = _tokens(query)
    expansions = {
        "product": ["portfolio", "comparison", "coachplex", "pro"],
        "source": ["register", "reference", "citation", "listing", "link"],
        "architecture": ["backend", "pipeline", "infrastructure", "system"],
        "quality": ["testing", "qa", "validation", "regression"],
        "launch": ["date", "availability", "release", "app", "listing", "announcement"],
        "differ": ["difference", "comparison", "compare", "coachplex", "pro"],
        "difference": ["differ", "comparison", "compare", "coachplex", "pro"],
        "compare": ["difference", "comparison", "differ", "coachplex", "pro"],
        "customer": ["count", "business", "analysis", "open", "question", "revenue", "market"],
        "revenue": ["customer", "count", "business", "analysis", "open", "question", "market"],
    }
    query_terms.extend(term for token in list(query_terms) for term in expansions.get(token, []))
    if not query_terms or not documents:
        return []
    tokenized = [_tokens(doc.page_content) for doc in documents]
    metadata_tokens = [
        _tokens(" ".join(str(doc.metadata.get(key, "")) for key in ("keywords", "document_title", "source", "category", "product")))
        for doc in documents
    ]
    avg_len = sum(map(len, tokenized)) / max(len(tokenized), 1)
    document_frequency = Counter(term for row in tokenized for term in set(row))
    n = len(tokenized)
    scores: list[tuple[float, int]] = []
    for i, terms in enumerate(tokenized):
        counts = Counter(terms)
        score = 0.0
        for term in set(query_terms):
            tf = counts[term]
            if tf:
                idf = max(0.0, math.log(1 + (n - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5)))
                score += idf * (tf * 2.0) / (tf + 1.2 * (0.25 + 0.75 * len(terms) / max(avg_len, 1)))
            meta_tf = metadata_tokens[i].count(term)
            if meta_tf:
                score += 1.5 * math.log1p(meta_tf) * math.log1p(n / (1 + document_frequency[term]))
        if str(documents[i].metadata.get("source", "")).startswith("company_knowledge/"):
            score *= 1.15
        if score:
            scores.append((score, i))
    scores.sort(reverse=True)
    return [documents[index] for _, index in scores[:k]]


def _matches_filters(doc: Document, filters: dict[str, str] | None) -> bool:
    for key, value in (filters or {}).items():
        if value and str(doc.metadata.get(key, "")).casefold() != value.casefold():
            return False
    return True


def _contextual_query(question: str, history: list[dict[str, str]] | None) -> str:
    if not history or len(_tokens(question)) > 8:
        return question
    if not re.search(r"\b(it|this|that|they|these|those|their|there|what about|and the|how about)\b", question, re.I):
        return question
    recent = [f"{item.get('role', 'user')}: {item.get('content', '')}" for item in history[-4:] if item.get("content")]
    return " ".join(recent + [f"current question: {question}"])


def retrieve_documents(
    query: str,
    vectorstore: FAISS,
    *,
    history: list[dict[str, str]] | None = None,
    filters: dict[str, str] | None = None,
    k: int = MAX_RETRIEVAL,
    dense: bool = True,
) -> list[Document]:
    """Fuse dense FAISS and lexical BM25 rankings using reciprocal rank fusion."""
    rewritten = _contextual_query(query, history)
    all_documents = _all_indexed_documents(vectorstore)
    corpus = [doc for doc in all_documents if _matches_filters(doc, filters)]
    lexical = bm25_search(rewritten, corpus, k=16)
    semantic: list[Document] = []
    if dense:
        try:
            search_options: dict[str, Any] = {"k": 16}
            if filters:
                search_options.update({"filter": filters, "fetch_k": max(16, int(vectorstore.index.ntotal))})
            semantic = [d for d in vectorstore.similarity_search(rewritten, **search_options) if _matches_filters(d, filters)]
        except Exception as exc:
            if not lexical:
                raise RuntimeError(f"Semantic retrieval failed and no lexical results were available: {exc}") from exc
    fused: dict[str, tuple[float, Document]] = {}
    for ranking in (semantic, lexical):
        for rank, doc in enumerate(ranking, 1):
            source = str(doc.metadata.get("source", ""))
            chunk_id = str(doc.metadata.get("chunk_id", ""))
            identity = chunk_id or hashlib.sha256(f"{source}:{doc.page_content}".encode("utf-8")).hexdigest()
            prior = fused.get(identity, (0.0, doc))[0]
            fused[identity] = (prior + 1 / (60 + rank), doc)
    results = [doc for _, doc in sorted(fused.values(), key=lambda item: item[0], reverse=True)]
    deduped: list[Document] = []
    seen_content: set[str] = set()
    source_counts: Counter[str] = Counter()
    for doc in results:
        normalized = re.sub(r"\W+", " ", doc.page_content.lower()).strip()
        if normalized in seen_content:
            continue
        source = str(doc.metadata.get("source", "unknown"))
        if source_counts[source] >= 2:
            continue
        seen_content.add(normalized)
        source_counts[source] += 1
        deduped.append(doc)
        if len(deduped) >= k:
            break
    return deduped


def source_payload(documents: Iterable[Document], max_sources: int = MAX_RETRIEVAL) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for doc in documents:
        metadata = doc.metadata
        source = str(metadata.get("source", "unknown"))
        section = str(metadata.get("section", "Document"))
        key = (source, section)
        if key in seen:
            continue
        seen.add(key)
        try:
            urls = json.loads(metadata.get("source_urls", "[]"))
        except (TypeError, json.JSONDecodeError):
            urls = []
        sources.append({
            "source": source,
            "title": str(metadata.get("document_title", Path(source).stem)),
            "section": section,
            "category": str(metadata.get("category", "source_material")),
            "product": str(metadata.get("product", "company")),
            "verification_status": str(metadata.get("verification_status", "internal_context")),
            "source_type": str(metadata.get("source_type", "internal_document")),
            "urls": urls if isinstance(urls, list) else [],
        })
        if len(sources) >= max_sources:
            break
    return sources


def answer_question(
    question: str,
    vectorstore: FAISS,
    *,
    history: list[dict[str, str]] | None = None,
    filters: dict[str, str] | None = None,
) -> dict[str, Any]:
    docs = retrieve_documents(question, vectorstore, history=history, filters=filters)
    if not docs:
        return {"answer": "I don't have enough documented information to answer that.", "sources": []}
    source_ids: dict[tuple[str, str], str] = {}
    context_parts: list[str] = []
    context_length = 0
    for doc in docs:
        meta = doc.metadata
        key = (str(meta.get("source", "unknown")), str(meta.get("section", "Document")))
        sid = source_ids.setdefault(key, f"S{len(source_ids) + 1}")
        header = f"[{sid}] {meta.get('document_title', key[0])} — {key[1]} | status={meta.get('verification_status', 'internal_context')} | source={key[0]}"
        block = f"{header}\n{doc.page_content}"
        if context_length + len(block) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - context_length
            if remaining > 400:
                context_parts.append(block[:remaining])
            break
        context_parts.append(block)
        context_length += len(block)
    conversation = "\n".join(f"{item.get('role', 'user')}: {item.get('content', '')}" for item in (history or [])[-8:])
    context = "\n\n".join(context_parts)
    prompt = f"{SYSTEM_PROMPT}\n\nConversation so far (may be empty):\n{conversation or '(none)'}\n\nRetrieved company context:\n{context}\n\nCurrent question: {question}\n\nAnswer with grounded citations like [S1]."
    try:
        response = ChatOpenAI(model=CHAT_MODEL, max_retries=1).invoke(prompt)
    except Exception as exc:
        message = str(exc)
        if any(marker in message.casefold() for marker in ("429", "resource_exhausted", "quota")):
            excerpts = []
            for doc in docs[:3]:
                meta = doc.metadata
                key = (str(meta.get("source", "unknown")), str(meta.get("section", "Document")))
                sid = source_ids.setdefault(key, f"S{len(source_ids) + 1}")
                excerpts.append(f"[{sid}] {meta.get('document_title', key[0])} — {key[1]}\n{doc.page_content[:900]}")
            return {
                "answer": "The answer generator is temporarily rate-limited. Here are the most relevant retrieved passages; each has a source reference:\n\n" + "\n\n".join(excerpts),
                "sources": source_payload(docs),
                "degraded": True,
            }
        raise
    answer = response.content if isinstance(response.content, str) else str(response.content)
    return {"answer": answer, "sources": source_payload(docs), "degraded": False}


if __name__ == "__main__" and "--sync" in sys.argv:
    sync_vectorstore()

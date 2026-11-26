# Advanced Impactor Company Knowledge Assistant

FastAPI chatbot backed by LangChain, Gemini embeddings/chat, a local FAISS index, and a structured company knowledge library. Dense retrieval is fused with a lightweight BM25 keyword ranker. Existing PDF, DOCX, and PowerPoint sources remain supported.

## Run on Windows

Create the Python environment, install dependencies, then run the app:

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install -r requirements.txt
.\.venv311\Scripts\python.exe run.py
```

The `.env` file needs a valid `GOOGLE_API_KEY`. Set `GEMINI_EMBEDDING_MODEL` or `GEMINI_CHAT_MODEL` there to override the defaults. The app is at http://127.0.0.1:8000; `/health` reports service and manifest status. Startup automatically syncs changed source files and embeds only changed/new files. On the first run, the legacy index is rebuilt once to add source metadata. If port 8000 is already in use, set another port before launch: `$env:PORT=8001; .\.venv311\Scripts\python.exe run.py`.

GitHub Pages hosts static files only; it cannot run this FastAPI backend. For a public Pages deployment, deploy the backend separately using `render.yaml`, configure `GOOGLE_API_KEY` as a secret in the backend host, then set the backend's public origin in `api-config.js` as `window.CHATBOT_API_BASE_URL`. Do not put API keys in `api-config.js`. The Pages frontend sends `/ask` requests to that configured API origin.

## Knowledge and ingestion

Company knowledge lives under `company_knowledge/`, grouped into company, products, features, technology, operations, business, and documentation. The original supported source files remain in `data/`.

To add a Markdown, PDF, or TXT source, place it under `company_knowledge/` or `data/`. Markdown files should start with YAML front matter using `document_title`, `category`, `product`, `source_type`, `verification_status`, `last_updated`, `keywords`, and optional `source_urls`. The supported status labels are `VERIFIED_PUBLIC`, `INTERNAL_CONTEXT`, `HISTORICAL`, `ANALYTICAL`, and `UNVERIFIED`.

Run a sync manually with:

```powershell
.\.venv311\Scripts\python.exe -m app.rag --sync
```

The index manifest tracks each source hash and its chunk IDs. Unchanged files keep their vectors; changed files have their old chunks removed before replacement; deleted files are removed from the index. Heading-aware chunking keeps Markdown sections intact and records source, title, section, category, product, verification status, and source URLs on each chunk.

## Retrieval and conversation

The app combines Gemini/FAISS dense similarity with BM25 lexical ranking using reciprocal-rank fusion. It deduplicates repeated chunks, supports category/product/status filters on `/ask`, uses recent browser conversation turns to resolve short follow-ups, limits context size, and returns document/section sources with status and links. The browser stores the latest 40 messages in local storage; the API does not maintain shared server-side chat logs.

The system prompt is maintained as `SYSTEM_PROMPT` in `app/rag.py`. It requires grounded answers, document citations, clear disclosure of historical/internal/unverified claims, and abstention when evidence is missing.

## Evaluation

Run the unit tests and offline source-retrieval evaluation:

```powershell
.\.venv311\Scripts\python.exe -m unittest discover -s tests -v
.\.venv311\Scripts\python.exe evaluation\evaluate_retrieval.py
```

The 33-question set is `evaluation/questions.json`. Offline mode checks expected-source hit rate and MRR using BM25 without external API calls. Add `--semantic` to evaluate the FAISS+BM25 hybrid; this calls the configured embedding API. Add `--sync` to refresh the index before evaluation. The current offline run scored 100% expected-source hit@5, MRR 0.909, and 29.7% source-precision proxy. The source relevance labels are deliberately narrow, so see `evaluation/review_rubric.md` before interpreting this as complete context precision.

For answer-level checks (fact-key coverage, cited source validity, expected source coverage, context precision, and explicit uncertainty on unknown questions), run `evaluation\evaluate_answers.py --limit 15`. It sends Gemini requests. Results are saved locally in an ignored JSONL file; use `--resume` after quota resets to continue. These automatic checks are a screening aid; answer quality and hallucination risk still need human review. If generation is rate-limited during normal chat use, the app returns the retrieved passages with their citations instead of hanging on retries.

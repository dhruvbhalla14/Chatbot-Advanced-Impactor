---
document_title: Internal RAG System
category: technology
product: internal_chatbot
source_type: repository_code_inspection_and_user_provided_context
verification_status: INTERNAL_CONTEXT
last_updated: 2026-11-26
keywords: [RAG, LangChain, FAISS, OpenAI, embeddings, retrieval, chatbot, ChromaDB]
---
# Internal RAG system

## Current repository implementation
[INTERNAL_CONTEXT] This standalone project uses FastAPI and LangChain. It uses OpenAI embeddings (OPENAI_EMBEDDING_MODEL, default text-embedding-3-small), OpenAI chat (OPENAI_CHAT_MODEL, default gpt-5.4-mini), and a local FAISS vector store. It combines dense FAISS similarity with lexical BM25 ranking and reciprocal-rank fusion. Its index reads company_knowledge and supported original files in data. Changing the embedding provider or model triggers a full FAISS rebuild.

## Earlier chatbot context
[HISTORICAL] The supplied brief refers to an earlier standalone internal chatbot using LangChain, ChromaDB, and RAG. That does not match this repository's current FAISS implementation.

## Conversation and scope
The web frontend sends recent conversation turns to the ask endpoint and keeps its transcript in browser local storage. The API does not create a shared server-side chat log. This internal assistant is not established as a production feature of the public mobile app.


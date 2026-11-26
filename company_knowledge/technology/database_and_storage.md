---
document_title: Database and Storage
category: technology
product: company
source_type: user_provided_internal_context_and_repository_inspection
verification_status: HISTORICAL
last_updated: 2026-11-26
keywords: [MongoDB, S3, storage, video, application data, FAISS, vector database]
---
# Database and storage

## Previously documented company architecture
[HISTORICAL] The supplied architecture identifies MongoDB for application data and processing-recovery information, and Amazon S3 for video storage. Current production usage and retention rules have not been independently verified.

## This chatbot repository
[INTERNAL_CONTEXT] This repository stores RAG vectors locally using FAISS under the vectorstore directory. A source manifest records hashes and chunk IDs for incremental updates. Knowledge Markdown and original files in data are read from disk. This chatbot does not use ChromaDB.

## Unknowns
No provided source documents retention periods, encryption, backups, database schema, or production storage access controls.


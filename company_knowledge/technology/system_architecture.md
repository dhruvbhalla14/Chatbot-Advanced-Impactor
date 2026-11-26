---
document_title: System Architecture (Previously Documented)
category: technology
product: company
source_type: user_provided_internal_context
verification_status: HISTORICAL
last_updated: 2026-11-26
keywords: [system architecture, AWS Lambda, Node.js, Python, MongoDB, S3, queue, processing server]
---
# System architecture

## Status and scope
[HISTORICAL] The supplied company brief describes a previous architecture, not independently verified current production. Services, ownership, deployments, and data flows may have changed.

## Previously documented components
- Application backend: Node.js on AWS Lambda, CRUD operations for players, coaches, parents, and video-analysis results; MongoDB for application data.
- A separate Node.js AWS Lambda analysis backend, queue-based processing, with recovery information in MongoDB.
- A Linux processing server with Python video rotation correction and ball-by-ball segmentation.
- Amazon S3 for video storage.
- Separate workers for biomechanics, shot classification, and Gemini-based analysis.

## Unknowns
The brief does not identify queues, exact functions, schemas, security controls, deployment regions, current service status, or ownership boundaries. Do not infer those details.

## This repository's chatbot
[INTERNAL_CONTEXT] The inspected standalone chatbot uses FastAPI, LangChain, Google Gemini embeddings/chat, and local FAISS. It is not evidence of the public app's production architecture.


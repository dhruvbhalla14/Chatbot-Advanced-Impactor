---
document_title: Backend Architecture (Previously Documented)
category: technology
product: company
source_type: user_provided_internal_context
verification_status: HISTORICAL
last_updated: 2026-11-26
keywords: [backend, Node.js, AWS Lambda, CRUD, MongoDB, queue, recovery]
---
# Backend architecture

## Application backend
[HISTORICAL] The supplied internal architecture describes a Node.js backend deployed on AWS Lambda for CRUD operations involving players, coaches, parents, and video-analysis results, with MongoDB for application data.

## Analysis backend
[HISTORICAL] It describes a separate Node.js AWS Lambda backend for video analysis. Processing is queue-based; recovery information is stored in MongoDB.

## Unknown details
No source provided here names the queue provider, API gateway, authentication mechanism, schema, retry policy, region, or present-day deployment. Confirm these details with engineering before using this as an implementation diagram or runbook.


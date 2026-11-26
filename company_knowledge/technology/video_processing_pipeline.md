---
document_title: Video Processing Pipeline (Previously Documented)
category: technology
product: advanced_impactor
source_type: user_provided_internal_context
verification_status: HISTORICAL
last_updated: 2026-11-26
keywords: [video, processing, segmentation, rotation correction, ball-by-ball, S3, Linux, Python]
---
# Video processing pipeline

## Previously documented sequence
[HISTORICAL] The supplied architecture describes video stored in Amazon S3, then processed by a Linux-based server running Python services. Processing includes correcting video rotation and segmenting the recording into ball-by-ball clips. A queue-based Node.js/Lambda backend coordinates work and stores recovery information in MongoDB.

## Analysis workers
The same brief describes separate workers for biomechanics, shot classification, and Gemini-based analysis. It does not document their exact inputs, output schemas, scheduling, retries, or whether this sequence remains current.

## Public product behavior
[VERIFIED_PUBLIC] App listings describe recording or uploading video and reviewing batting, bowling, ball-tracking, biomechanical, and performance insights. They do not verify the internal pipeline above.


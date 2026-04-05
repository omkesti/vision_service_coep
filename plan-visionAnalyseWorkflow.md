## Plan: Vision Analyze + Webhook Flow

Build an asynchronous Vision Service around the current anomaly pipeline so backend can submit analyze requests, receive immediate acceptance, and later get final incident results via callback with snapshot evidence.

**Steps**

1. Phase 1: API contract and async acceptance
2. Define POST /vision/analyze request schema with session_id, camera_id, stream_url, lat, lng, callback_url.
3. Define callback payload schema with camera_id, incident_type, risk_score, confidence, lat, lng, snapshot_url.
4. Return immediate accepted response with job_id and status while processing runs in background.
5. Phase 2: Reuse current analyzer as a service
6. Extract logic from test_anomaly.py into a reusable analysis runner.
7. Reuse detector methods from anomaly_engine/anomaly_detector.py: calibrate_baseline_from_video and get_score.
8. Implement incident decision mapping from anomaly score to incident_type, risk_score, confidence.
9. Phase 3: Media retrieval + snapshot generation
10. Add media ingestion from stream_url into temporary local media.
11. For stream sources, analyze bounded duration/window to control cost.
12. Capture the highest-score frame and publish it to a retrievable snapshot_url.
13. Phase 4: Background worker + callback delivery
14. Worker flow: retrieve media -> analyze -> build payload -> POST to callback_url.
15. Add retry/backoff for callback failures and structured failure logging.
16. Add idempotency guard using session_id + camera_id to reduce duplicate callback emission.
17. Phase 5: Reliability and configuration
18. Centralize config for model path, thresholds, URL timeouts, retries, temp storage, snapshot base URL.
19. Use structured logs with correlation keys: session_id, camera_id, job_id.
20. Phase 6: Testing and verification
21. Unit tests for request validation, scoring-to-incident mapping, callback payload construction.
22. Integration test for full async path with mocked stream source and mocked callback endpoint.
23. Failure tests for unreachable stream_url, unreadable media, callback timeout/5xx.

**Relevant files**

- anomaly_engine/anomaly_detector.py: detector scoring and baseline calibration reuse.
- test_anomaly.py: reference inference loop to convert into service logic.
- anomaly_engine/autoencoder.py: model dependency used by detector.
- anomaly_engine/trained_model.pth: model artifact to make configurable.

**Decisions captured**

1. callback_url should be included per request payload.
2. /vision/analyze should be asynchronous (immediate ack + background processing).
3. Initial incident_type mapping can start with fire and remain extendable.

**Verification checklist**

1. POST /vision/analyze returns quickly with accepted status and job_id.
2. Callback payload always includes required fields and valid risk_score/confidence range.
3. snapshot_url points to the exact detection frame (max anomaly score frame).
4. Callback retries work on transient failures and stop after configured max attempts.

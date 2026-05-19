# Vision Service (Urban Safety AI)

This repository contains the vision-service component of an AI-powered urban safety platform. It ingests a video stream, runs three parallel AI branches (anomaly risk, incident classification, and crowd detection), fuses the results, and writes incident alerts with geolocation to Supabase.

## Problem Summary (Brief)

Cities need faster, proactive safety monitoring. The goal is to automatically detect incidents from video, associate them with locations, dispatch the nearest drone (simulated), and show live movement on a map. This service handles the first step: AI detection and alert generation from video feeds.

## Skills Needed (Overall System)

- Computer vision and deep learning (video preprocessing, detection, classification)
- Backend APIs (FastAPI, async processing, data storage)
- Geospatial concepts (lat/lng mapping, distance, routing)
- Frontend dashboards (maps, real-time updates, UI/UX)
- DevOps basics (model artifacts, deployment, environment config)

### Specifically for Video Analysis

- Video decoding and frame sampling (OpenCV)
- Anomaly detection using autoencoders (PyTorch)
- Video classification (VideoMAE via Hugging Face)
- Object detection for crowd estimation (YOLO via Ultralytics)
- Result fusion and threshold calibration

### Specifically for Frontend (Dashboard)

- Map visualization (e.g., Mapbox, Leaflet, or Google Maps)
- Real-time updates (WebSocket, polling, or SSE)
- Incident list with severity and status
- Drone markers with live position, ETA, and route
- Video preview widgets (source feed and response drone)

## How This Service Works

Flow:

1. Download video from `stream_url`.
2. Decode once and build shared frame sets.
3. Run three branches in parallel:
   - Risk scoring (autoencoder reconstruction error)
   - Incident classification (VideoMAE)
   - Crowd detection (YOLO person detection + clustering)
4. Fuse results into a single incident type + risk score.
5. Insert incident into Supabase with lat/lng.

## Key Components

- `main.py` - FastAPI app with `/analyze` endpoint, Supabase insert.
- `analysis_service.py` - Orchestrates download, frame prep, parallel branches, and fusion.
- `risk_scoring_service.py` - Autoencoder-based anomaly scoring.
- `incident_classification_service.py` - VideoMAE-based incident classification with rejection rules.
- `crowd_detection_service.py` - YOLO-based crowd scoring.
- `fusion_service.py` - Combines branch outputs into final incident decision.
- `anomaly_engine/` - Autoencoder model and utilities.
- `yolo11m.pt`, `yolo11n.pt` - YOLO model weights.

## API

### POST /analyze

Request body:

```json
{
  "session_id": "sess-123",
  "camera_id": "cam-01",
  "stream_url": "https://.../video.mp4",
  "lat": 18.5204,
  "lng": 73.8567
}
```

Response body:

```json
{
  "id": 1,
  "camera_id": "cam-01",
  "incident_type": "vehicle_collision",
  "risk_score": 0.72,
  "confidence": 0.81,
  "decision_source": "agreed",
  "crowd_score": 0.0,
  "lat": 18.5204,
  "lng": 73.8567,
  "snapshot_url": null
}
```

## Environment Variables

Required:

- `SUPABASE_URL`
- `SUPABASE_SERVICE_KEY`

Optional (tuning):

- `ALERT_THRESHOLD` (default 0.08)
- `MAX_CONCURRENT_DOWNLOADS`, `MAX_CONCURRENT_RISK`, `MAX_CONCURRENT_CLASSIFIER`, `MAX_CONCURRENT_CROWD`
- `DOWNLOAD_TIMEOUT_SECONDS`, `REQUEST_TIMEOUT_SECONDS`
- `VIDEO_CLS_MODEL_NAME`, `VIDEO_CLS_TOP_K`, `VIDEO_CLS_DEVICE`
- `YOLO_CROWD_MODEL`, `YOLO_PERSON_CONF`

## Local Run (FastAPI)

1. Install dependencies (example):
   - `fastapi`, `uvicorn`, `torch`, `opencv-python`, `numpy`, `requests`, `python-dotenv`, `transformers`, `ultralytics`, `supabase`
2. Set environment variables in `.env`.
3. Run:
   - `uvicorn main:app --host 0.0.0.0 --port 8000`

## Notes

- This service expects a direct video URL for `stream_url`.
- The autoencoder model weights are stored at `anomaly_engine/trained_model.pth`.
- The fusion logic ensures classifier outputs are only trusted when backed by risk or confidence thresholds.

## Project Context

This service is part of a larger hackathon project for an AI-powered urban safety platform. The complete system includes simulated drone dispatch and a command dashboard, which live outside this repository.

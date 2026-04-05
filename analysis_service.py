"""
AnalysisService — parallel pipeline orchestrator.
Mirrors test_anomaly.py behavior exactly:
  1. Download video
  2. Decode once -> shared frame sets
  3. Run RiskScoringService + IncidentClassificationService in parallel
  4. Fuse results
  5. Return FusionResult
"""

import os
import time
import tempfile
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import cv2
import numpy as np

from risk_scoring_service import RiskScoringService
from incident_classification_service import IncidentClassificationService
from fusion_service import fuse, FusionResult

logger = logging.getLogger("analysis_service")

# ── Config ────────────────────────────────────────────────────────────────────
MAX_CONCURRENT_DOWNLOADS  = int(os.getenv("MAX_CONCURRENT_DOWNLOADS",  "5"))
MAX_CONCURRENT_RISK       = int(os.getenv("MAX_CONCURRENT_RISK",        "2"))
MAX_CONCURRENT_CLASSIFIER = int(os.getenv("MAX_CONCURRENT_CLASSIFIER",  "2"))
DOWNLOAD_TIMEOUT_SECONDS  = int(os.getenv("DOWNLOAD_TIMEOUT_SECONDS",  "30"))
REQUEST_TIMEOUT_SECONDS   = int(os.getenv("REQUEST_TIMEOUT_SECONDS",   "120"))
ALERT_THRESHOLD           = float(os.getenv("ALERT_THRESHOLD",          "0.08"))

CALIBRATION_FRAMES = int(os.getenv("CALIBRATION_FRAMES", "30"))
ANALYSIS_FRAMES    = int(os.getenv("ANALYSIS_FRAMES",    "60"))
CLS_NUM_FRAMES     = int(os.getenv("VIDEO_CLS_NUM_FRAMES", "16"))
FRAME_SIZE         = int(os.getenv("VIDEO_CLS_FRAME_SIZE", "224"))

MODEL_PATH = Path(__file__).resolve().parent / "anomaly_engine" / "trained_model.pth"


# ── Frame preparation (blocking, runs in thread pool) ─────────────────────────
def _prepare_frames(video_path: str) -> tuple:
    """
    Decode video once. Returns (cal_frames, score_frames, cls_frames, all_frames).
    Mirrors prepare_frames() in test_anomaly.py exactly.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        all_frames.append(frame)
    cap.release()

    if not all_frames:
        raise RuntimeError("No frames decoded from video.")

    cal_frames   = all_frames[:CALIBRATION_FRAMES]
    score_frames = all_frames[CALIBRATION_FRAMES:CALIBRATION_FRAMES + ANALYSIS_FRAMES]

    if not score_frames:
        score_frames = all_frames  # fallback for short videos

    # 16 evenly sampled RGB 224x224 frames for VideoMAE
    indices = np.linspace(0, len(score_frames) - 1, CLS_NUM_FRAMES, dtype=int)
    cls_frames = []
    for idx in indices:
        f = cv2.resize(score_frames[idx], (FRAME_SIZE, FRAME_SIZE))
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        cls_frames.append(f)

    while len(cls_frames) < CLS_NUM_FRAMES:
        cls_frames.append(cls_frames[-1])

    return cal_frames, score_frames, cls_frames[:CLS_NUM_FRAMES], all_frames


# ── Main service ──────────────────────────────────────────────────────────────
class AnalysisService:

    def __init__(self):
        logger.info("Initializing RiskScoringService...")
        self.risk_svc = RiskScoringService(str(MODEL_PATH), alert_threshold=ALERT_THRESHOLD)

        logger.info("Initializing IncidentClassificationService...")
        self.cls_svc = IncidentClassificationService()

        self._download_sem  = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
        self._risk_sem      = asyncio.Semaphore(MAX_CONCURRENT_RISK)
        self._cls_sem       = asyncio.Semaphore(MAX_CONCURRENT_CLASSIFIER)
        self._thread_pool   = ThreadPoolExecutor(max_workers=4, thread_name_prefix="vision")

        logger.info("AnalysisService ready | alert_threshold=%.2f", ALERT_THRESHOLD)

    # ── Download ──────────────────────────────────────────────────────────────
    async def _download(self, url: str, session_id: str) -> str:
        if not self._download_sem._value:
            raise RuntimeError("download_queue_full")
        async with self._download_sem:
            t0 = time.monotonic()
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp_path = tmp.name
            tmp.close()

            def _fetch(url: str, path: str):
                clean_url = url.strip()
                for attempt in range(1, 4):
                    resp = requests.get(clean_url, stream=True, timeout=DOWNLOAD_TIMEOUT_SECONDS)
                    resp.raise_for_status()
                    with open(path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=1024 * 64):
                            f.write(chunk)
                    size = os.path.getsize(path)
                    logger.info("[%s] Attempt %d: %d bytes", session_id, attempt, size)
                    if size >= 1024:
                        return
                    logger.warning("[%s] File not ready, retrying in 2s...", session_id)
                    time.sleep(2)
                raise RuntimeError("File still empty after 3 attempts")

            loop = asyncio.get_running_loop()
            await asyncio.wait_for(
                loop.run_in_executor(self._thread_pool, _fetch, url, tmp_path),
                timeout=DOWNLOAD_TIMEOUT_SECONDS + 10,
            )
            logger.info("[%s] download=%.2fs", session_id, time.monotonic() - t0)
            return tmp_path

    # ── Branch runners (blocking, for executor) ───────────────────────────────
    def _run_risk(self, cal_frames, score_frames):
        return self.risk_svc.analyze(cal_frames, score_frames)

    def _run_cls(self, cls_frames):
        return self.cls_svc.analyze(cls_frames)

    # ── Public entry point ────────────────────────────────────────────────────
    async def analyze(self, session_id: str, camera_id: str, stream_url: str) -> FusionResult:
        timings = {}
        tmp_path = None
        try:
            # Step 1 — download
            timings["t_download"] = time.monotonic()
            tmp_path = await self._download(stream_url, session_id)
            timings["download_s"] = round(time.monotonic() - timings["t_download"], 3)

            # Step 2 — decode once (shared frames)
            timings["t_prepare"] = time.monotonic()
            loop = asyncio.get_running_loop()
            cal_frames, score_frames, cls_frames, _ = await loop.run_in_executor(
                self._thread_pool, _prepare_frames, tmp_path
            )
            timings["prepare_s"] = round(time.monotonic() - timings["t_prepare"], 3)
            logger.info("[%s] frames: cal=%d score=%d cls=%d",
                        session_id, len(cal_frames), len(score_frames), len(cls_frames))

            # Step 3 — run both branches in parallel
            timings["t_branches"] = time.monotonic()
            risk_result = cls_result = risk_error = cls_error = None

            async def run_risk():
                nonlocal risk_result, risk_error
                async with self._risk_sem:
                    try:
                        risk_result = await loop.run_in_executor(
                            self._thread_pool, self._run_risk, cal_frames, score_frames
                        )
                    except Exception as e:
                        risk_error = e
                        logger.error("[%s] risk branch failed: %s", session_id, e)

            async def run_cls():
                nonlocal cls_result, cls_error
                async with self._cls_sem:
                    try:
                        cls_result = await loop.run_in_executor(
                            self._thread_pool, self._run_cls, cls_frames
                        )
                    except Exception as e:
                        cls_error = e
                        logger.error("[%s] classifier branch failed: %s", session_id, e)

            await asyncio.gather(run_risk(), run_cls())
            timings["branches_s"] = round(time.monotonic() - timings["t_branches"], 3)

            if risk_error and cls_error:
                raise RuntimeError(f"Both branches failed: risk={risk_error} cls={cls_error}")

            # Step 4 — fuse
            degraded = "risk_failed" if risk_error else ("cls_failed" if cls_error else None)
            fusion = fuse(risk_result, cls_result, degraded=degraded)

            logger.info(
                "[%s] %s | incident=%s risk=%.4f conf=%.4f source=%s | "
                "download=%.2fs prepare=%.2fs branches=%.2fs",
                session_id, camera_id,
                fusion.incident_type, fusion.risk_score, fusion.confidence, fusion.decision_source,
                timings["download_s"], timings["prepare_s"], timings["branches_s"],
            )

            return fusion

        except RuntimeError as e:
            raise
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

"""
CrowdDetectionService — YOLO person detection branch.
Runs on score_frames (every Nth frame), outputs crowd heuristics.
"""

import os
import time
import logging
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger("crowd_detection")

YOLO_MODEL          = os.getenv("YOLO_CROWD_MODEL",         "yolo11m.pt")
YOLO_PERSON_CONF    = float(os.getenv("YOLO_PERSON_CONF",   "0.25"))
CROWD_MIN_PEOPLE    = int(os.getenv("CROWD_MIN_PEOPLE",     "10"))
CROWD_MIN_SUSTAINED = int(os.getenv("CROWD_MIN_SUSTAINED_FRAMES", "3"))
CROWD_MIN_CLUSTER   = float(os.getenv("CROWD_MIN_CLUSTER_RATIO",  "0.20"))
FRAME_STRIDE        = int(os.getenv("CROWD_FRAME_STRIDE",   "3"))   # sample every Nth frame


@dataclass
class CrowdResult:
    crowd_detected: bool
    crowd_score: float
    max_person_count: int
    avg_person_count: float
    cluster_frames_ratio: float
    crowd_confidence: float
    timing_s: float = 0.0


def _centroid(bbox) -> tuple:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _cluster_score(centers: list, frame_w: int, frame_h: int) -> float:
    """
    Measure how spatially concentrated people are.
    Returns 0 (spread out) to 1 (tightly clustered).
    Uses std deviation of centroid positions normalized by frame size.
    """
    if len(centers) < 2:
        return 0.0
    xs = [c[0] / max(frame_w, 1) for c in centers]
    ys = [c[1] / max(frame_h, 1) for c in centers]
    spread = float(np.std(xs) + np.std(ys))
    # low spread = high cluster score, but dense crowds fill the frame so use gentler penalty
    return round(max(0.0, 1.0 - spread * 1.5), 4)


class CrowdDetectionService:

    def __init__(self):
        from ultralytics import YOLO
        self.model = YOLO(YOLO_MODEL)
        self.model.fuse()
        logger.info("CrowdDetectionService ready | model=%s conf=%.2f", YOLO_MODEL, YOLO_PERSON_CONF)

    def analyze(self, score_frames: list) -> CrowdResult:
        t0 = time.monotonic()

        sampled = score_frames[::FRAME_STRIDE] if len(score_frames) > FRAME_STRIDE else score_frames

        per_frame_counts  = []
        per_frame_cluster = []
        sustained_count   = 0

        for frame in sampled:
            h, w = frame.shape[:2]
            results = self.model(frame, conf=YOLO_PERSON_CONF, classes=[0], verbose=False)

            persons = []
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    persons.append(boxes.xyxy[i].tolist())

            count = len(persons)
            per_frame_counts.append(count)

            if count >= CROWD_MIN_PEOPLE:
                sustained_count += 1
                centers = [_centroid(b) for b in persons]
                per_frame_cluster.append(_cluster_score(centers, w, h))
            else:
                per_frame_cluster.append(0.0)

        if not per_frame_counts:
            return CrowdResult(False, 0.0, 0, 0.0, 0.0, 0.0,
                               round(time.monotonic() - t0, 3))

        max_count  = int(max(per_frame_counts))
        avg_count  = round(float(np.mean(per_frame_counts)), 2)
        total      = len(sampled)
        cluster_ratio = round(float(np.mean(per_frame_cluster)), 4)
        sustained_ratio = round(sustained_count / max(total, 1), 4)

        # ── Crowd score ───────────────────────────────────────────────────────
        count_score    = min(max_count / max(CROWD_MIN_PEOPLE * 2, 1), 1.0)
        cluster_score  = cluster_ratio
        duration_score = sustained_ratio

        crowd_score = round(
            0.40 * count_score + 0.35 * cluster_score + 0.25 * duration_score, 4
        )

        crowd_detected = (
            max_count >= CROWD_MIN_PEOPLE
            and sustained_count >= CROWD_MIN_SUSTAINED
            and cluster_ratio >= CROWD_MIN_CLUSTER
        )

        logger.info(
            "Crowd: detected=%s score=%.3f max=%d avg=%.1f cluster=%.3f sustained=%d/%d",
            crowd_detected, crowd_score, max_count, avg_count,
            cluster_ratio, sustained_count, total
        )

        return CrowdResult(
            crowd_detected=crowd_detected,
            crowd_score=round(crowd_score, 4),
            max_person_count=max_count,
            avg_person_count=avg_count,
            cluster_frames_ratio=cluster_ratio,
            crowd_confidence=round(crowd_score, 4),
            timing_s=round(time.monotonic() - t0, 3),
        )

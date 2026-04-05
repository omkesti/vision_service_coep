"""
Parallel two-branch pipeline — shared frame preparation.
Both branches read from the same decoded frames.
"""

import cv2
import time
import threading
import numpy as np
from pathlib import Path

from risk_scoring_service import RiskScoringService
from incident_classification_service import IncidentClassificationService
from fusion_service import fuse

ROOT       = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "anomaly_engine" / "trained_model.pth"
VIDEO_PATH = ROOT / "test_videos" / "test11.mp4"

ALERT_THRESHOLD    = 0.08
CALIBRATION_FRAMES = 30
ANALYSIS_FRAMES    = 60
CLS_NUM_FRAMES     = 16
FRAME_SIZE         = 224


# ── Shared frame preparation ──────────────────────────────────────────────────
def prepare_frames(video_path: str):
    """
    Decode video once into:
      cal_frames   — first CALIBRATION_FRAMES BGR frames (for baseline)
      score_frames — next ANALYSIS_FRAMES BGR frames (for risk scoring)
      cls_frames   — CLS_NUM_FRAMES RGB 224x224 frames sampled from score window
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        all_frames.append(frame)
    cap.release()

    if not all_frames:
        raise RuntimeError("No frames decoded.")

    cal_frames   = all_frames[:CALIBRATION_FRAMES]
    score_frames = all_frames[CALIBRATION_FRAMES:CALIBRATION_FRAMES + ANALYSIS_FRAMES]

    if not score_frames:
        score_frames = all_frames  # fallback if video is short

    # Sample CLS_NUM_FRAMES evenly from score window for VideoMAE
    indices = np.linspace(0, len(score_frames) - 1, CLS_NUM_FRAMES, dtype=int)
    cls_frames = []
    for idx in indices:
        f = cv2.resize(score_frames[idx], (FRAME_SIZE, FRAME_SIZE))
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        cls_frames.append(f)

    # pad if short
    while len(cls_frames) < CLS_NUM_FRAMES:
        cls_frames.append(cls_frames[-1])

    return cal_frames, score_frames, cls_frames[:CLS_NUM_FRAMES], all_frames


def main():
    print("Loading risk scoring service...")
    risk_svc = RiskScoringService(str(MODEL_PATH), alert_threshold=ALERT_THRESHOLD)

    print("Loading VideoMAE classifier...")
    cls_svc = IncidentClassificationService()

    print(f"\nPreparing frames from: {VIDEO_PATH.name}")
    cal_frames, score_frames, cls_frames, all_frames = prepare_frames(str(VIDEO_PATH))
    print(f"  cal={len(cal_frames)} score={len(score_frames)} cls={len(cls_frames)} total={len(all_frames)}")

    # ── Run both branches in parallel ─────────────────────────────────────────
    risk_result = cls_result = risk_error = cls_error = None

    def run_risk():
        nonlocal risk_result, risk_error
        try:
            risk_result = risk_svc.analyze(cal_frames, score_frames)
        except Exception as e:
            risk_error = e

    def run_cls():
        nonlocal cls_result, cls_error
        try:
            cls_result = cls_svc.analyze(cls_frames)
        except Exception as e:
            cls_error = e

    t1 = threading.Thread(target=run_risk)
    t2 = threading.Thread(target=run_cls)
    t1.start(); t2.start()
    t1.join();  t2.join()

    # ── Print branch results ──────────────────────────────────────────────────
    print("\n── Branch A: Risk Scoring ──────────────────")
    if risk_result:
        print(f"  autoencoder_score_raw        : {risk_result.autoencoder_score_raw}")
        print(f"  autoencoder_score_calibrated : {risk_result.autoencoder_score_calibrated}")
        print(f"  risk_band                    : {risk_result.risk_band}")
        print(f"  risk_confidence              : {risk_result.risk_confidence}")
        print(f"  is_anomaly                   : {risk_result.is_anomaly}")
        print(f"  peak_frame                   : {risk_result.peak_frame_index}")
        print(f"  time                         : {risk_result.timing_s}s")
    else:
        print(f"  FAILED: {risk_error}")

    print("\n── Branch B: VideoMAE Classifier ───────────")
    if cls_result:
        print(f"  raw_label         : {cls_result.raw_label}")
        print(f"  mapped_label      : {cls_result.mapped_label}")
        print(f"  confidence        : {cls_result.classifier_confidence}")
        print(f"  margin (top1-top2): {cls_result.classifier_margin}")
        print(f"  accepted          : {cls_result.classifier_accepted}")
        print(f"  rejection_reason  : {cls_result.classifier_rejection_reason}")
        print(f"  top_k             :")
        for e in cls_result.top_k:
            print(f"    {e['label']:<25} {e['confidence']:.4f}  (raw: {e['raw_label']})")
        print(f"  time              : {cls_result.timing_s}s")
    else:
        print(f"  FAILED: {cls_error}")

    # ── Fusion ────────────────────────────────────────────────────────────────
    if risk_error and cls_error:
        print("\nBoth branches failed.")
        return

    degraded = "risk_failed" if risk_error else ("cls_failed" if cls_error else None)
    fusion = fuse(risk_result, cls_result, degraded=degraded)

    print("\n── Fusion Result ───────────────────────────")
    print(f"  incident_type   : {fusion.incident_type}")
    print(f"  risk_score      : {fusion.risk_score}")
    print(f"  confidence      : {fusion.confidence}")
    print(f"  decision_source : {fusion.decision_source}")

    # ── Playback with overlay ─────────────────────────────────────────────────
    # Recompute per-frame scores for display
    score_map = {}
    buf = []
    baseline_errors = [risk_svc._get_error(f) for f in cal_frames]
    baseline = float(np.mean(baseline_errors))
    for i, frame in enumerate(score_frames):
        err = risk_svc._get_error(frame)
        s = max(0.0, (err - baseline) / max(baseline, 1e-8))
        s = min(s, 1.0)
        buf.append(s)
        if len(buf) > 15:
            buf.pop(0)
        score_map[CALIBRATION_FRAMES + i] = float(np.mean(buf))

    for i, frame in enumerate(all_frames):
        score = score_map.get(i, 0.0)
        color = (0, 0, 255) if score >= ALERT_THRESHOLD else (0, 255, 0)

        cv2.putText(frame, f"Risk: {score:.3f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"Type: {fusion.incident_type}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        cv2.putText(frame, f"{fusion.decision_source} | conf:{fusion.confidence:.2f}",
                    (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

        cv2.imshow("Parallel Pipeline", frame)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""
RiskScoringService — autoencoder branch.
Accepts pre-sampled BGR frames (shared with classifier branch).
"""

import logging
import time
import numpy as np
import torch
from dataclasses import dataclass
from anomaly_engine.autoencoder import ConvolutionalAutoencoder
from risk_calibration import calibrate, risk_band as compute_risk_band

logger = logging.getLogger("risk_scoring")

MODEL_PATH_DEFAULT = "anomaly_engine/trained_model.pth"


@dataclass
class RiskResult:
    autoencoder_score_raw: float
    autoencoder_score_calibrated: float
    risk_band: str
    risk_confidence: float
    is_anomaly: bool
    peak_frame_index: int
    timing_s: float = 0.0

    # alias for backward compat
    @property
    def risk_score(self):
        return self.autoencoder_score_raw


class RiskScoringService:

    def __init__(self, model_path: str, alert_threshold: float = 0.08):
        import cv2
        self.cv2 = cv2
        self.alert_threshold = alert_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ConvolutionalAutoencoder().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state_dict)
        ckpt_thresh = checkpoint.get("threshold") if isinstance(checkpoint, dict) else None
        self.model_threshold = float(ckpt_thresh) if ckpt_thresh is not None else 0.005
        self.model.eval()
        logger.info("RiskScoringService ready on %s | model_threshold=%.5f", self.device, self.model_threshold)

    def _get_error(self, frame) -> float:
        img = self.cv2.resize(frame, (64, 64))
        img = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2GRAY)
        img = img / 255.0
        tensor = torch.tensor(img[None, None], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            recon = self.model(tensor)
        return float(torch.mean((tensor - recon) ** 2).item())

    def analyze(self, cal_frames: list, score_frames: list) -> RiskResult:
        """
        cal_frames:   BGR frames used to compute baseline (first N frames)
        score_frames: BGR frames to score against baseline
        """
        t0 = time.monotonic()

        # baseline from calibration frames
        cal_errors = [self._get_error(f) for f in cal_frames]
        baseline = float(np.mean(cal_errors)) if cal_errors else self.model_threshold

        # score frames
        raw_errors = [self._get_error(f) for f in score_frames]

        if not raw_errors:
            raise RuntimeError("No score frames provided.")

        scores, buf = [], []
        for err in raw_errors:
            s = max(0.0, (err - baseline) / max(baseline, 1e-8))
            s = min(s, 1.0)
            buf.append(s)
            if len(buf) > 15:
                buf.pop(0)
            scores.append(float(np.mean(buf)))

        risk_score_raw  = round(float(np.max(scores)), 4)
        risk_confidence = round(float(np.mean(scores)), 4)
        peak_frame      = int(np.argmax(scores)) + len(cal_frames)
        cal_score       = calibrate(risk_score_raw)
        band            = compute_risk_band(cal_score)

        logger.info("Risk: raw=%.4f calibrated=%.4f band=%s conf=%.4f peak=%d",
                    risk_score_raw, cal_score, band, risk_confidence, peak_frame)

        return RiskResult(
            autoencoder_score_raw=risk_score_raw,
            autoencoder_score_calibrated=cal_score,
            risk_band=band,
            risk_confidence=risk_confidence,
            is_anomaly=risk_score_raw >= self.alert_threshold,
            peak_frame_index=peak_frame,
            timing_s=round(time.monotonic() - t0, 3),
        )

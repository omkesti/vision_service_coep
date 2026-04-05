"""
IncidentClassificationService — VideoMAE branch.
Accepts pre-sampled frames. Adds rejection layer (margin + per-class thresholds).
Does NOT make final incident_type decisions — that's fusion's job.
"""

import os
import logging
import time
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from label_mapping import normalize_label

logger = logging.getLogger("incident_cls")

MODEL_NAME  = os.getenv("VIDEO_CLS_MODEL_NAME", "OPear/videomae-large-finetuned-UCF-Crime")
TOP_K       = int(os.getenv("VIDEO_CLS_TOP_K",  "3"))
DEVICE_CFG  = os.getenv("VIDEO_CLS_DEVICE",     "auto")

# Rejection thresholds
MIN_CLASSIFIER_MARGIN = float(os.getenv("MIN_CLASSIFIER_MARGIN", "0.20"))

# Per-class minimum confidence to be accepted
CLASS_CONF_THRESHOLDS = {
    "vehicle_collision": float(os.getenv("VEHICLE_COLLISION_CONF_THRESHOLD", "0.90")),
    "theft":             float(os.getenv("THEFT_CONF_THRESHOLD",             "0.60")),
    "fire":              float(os.getenv("FIRE_CONF_THRESHOLD",              "0.60")),
    "fighting":          float(os.getenv("FIGHTING_CONF_THRESHOLD",         "0.60")),
    "assault":           float(os.getenv("ASSAULT_CONF_THRESHOLD",          "0.60")),
    "abuse":             float(os.getenv("ABUSE_CONF_THRESHOLD",             "0.60")),
    "shooting":          float(os.getenv("SHOOTING_CONF_THRESHOLD",         "0.60")),
    "robbery":           float(os.getenv("ROBBERY_CONF_THRESHOLD",          "0.60")),
    "vandalism":         float(os.getenv("VANDALISM_CONF_THRESHOLD",        "0.60")),
    "arrest":            float(os.getenv("ARREST_CONF_THRESHOLD",           "0.60")),
}
DEFAULT_CONF_THRESHOLD = float(os.getenv("DEFAULT_CONF_THRESHOLD", "0.60"))


@dataclass
class ClassifierResult:
    raw_label: str
    mapped_label: str
    classifier_confidence: float
    classifier_margin: float          # top1 - top2
    classifier_accepted: bool
    classifier_rejection_reason: str  # None if accepted
    top_k: list = field(default_factory=list)
    timing_s: float = 0.0


class IncidentClassificationService:

    def __init__(self):
        from transformers import VideoMAEVideoProcessor, VideoMAEForVideoClassification

        self.device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
                       if DEVICE_CFG == "auto" else torch.device(DEVICE_CFG))

        logger.info("Loading VideoMAE: %s on %s", MODEL_NAME, self.device)
        self.processor = VideoMAEVideoProcessor.from_pretrained(MODEL_NAME)
        self.model     = VideoMAEForVideoClassification.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()
        logger.info("VideoMAE ready | labels=%d", self.model.config.num_labels)

    def _reject(self, mapped_label: str, conf: float, margin: float):
        """
        Returns (accepted: bool, reason: str|None).
        Rejection rules:
          1. mapped label is normal -> reject
          2. top1-top2 margin too small -> uncertain prediction
          3. confidence below per-class threshold
        """
        if mapped_label == "normal":
            return False, "label_is_normal"

        if margin < MIN_CLASSIFIER_MARGIN:
            return False, f"margin_too_small({margin:.3f}<{MIN_CLASSIFIER_MARGIN})"

        threshold = CLASS_CONF_THRESHOLDS.get(mapped_label, DEFAULT_CONF_THRESHOLD)
        if conf < threshold:
            return False, f"confidence_below_threshold({conf:.3f}<{threshold})"

        return True, None

    def analyze(self, frames: list) -> ClassifierResult:
        t0 = time.monotonic()

        inputs = self.processor(list(frames), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = F.softmax(outputs.logits, dim=-1)[0]
        top_k_vals, top_k_ids = torch.topk(probs, min(max(TOP_K, 2), len(probs)))

        top_k = []
        for val, idx in zip(top_k_vals.tolist(), top_k_ids.tolist()):
            raw = self.model.config.id2label[idx]
            top_k.append({
                "raw_label":  raw,
                "label":      normalize_label(raw),
                "confidence": round(float(val), 4),
            })

        top1   = top_k[0]
        top2   = top_k[1] if len(top_k) > 1 else {"confidence": 0.0}
        margin = round(top1["confidence"] - top2["confidence"], 4)

        accepted, rejection_reason = self._reject(top1["label"], top1["confidence"], margin)

        logger.info("Classifier: %s->%s conf=%.3f margin=%.3f accepted=%s reason=%s",
                    top1["raw_label"], top1["label"], top1["confidence"],
                    margin, accepted, rejection_reason)

        return ClassifierResult(
            raw_label=top1["raw_label"],
            mapped_label=top1["label"],
            classifier_confidence=round(top1["confidence"], 4),
            classifier_margin=margin,
            classifier_accepted=accepted,
            classifier_rejection_reason=rejection_reason,
            top_k=top_k[:TOP_K],
            timing_s=round(time.monotonic() - t0, 3),
        )

"""
FusionService — support-aware fusion with false-positive hardening.
Classifier can only promote a label when classifier_accepted=True.
"""

import os
import logging
from dataclasses import dataclass, field

logger = logging.getLogger("fusion")

VEHICLE_COLLISION_MIN_RISK  = float(os.getenv("VEHICLE_COLLISION_MIN_RISK",  "0.20"))
SEMANTIC_MIN_RISK_SUPPORT   = float(os.getenv("SEMANTIC_MIN_RISK_SUPPORT",   "0.25"))
UNKNOWN_ANOMALY_RISK_THRESH = float(os.getenv("UNKNOWN_ANOMALY_RISK_THRESH", "0.75"))
CROWD_FUSION_THRESHOLD      = float(os.getenv("CROWD_FUSION_THRESHOLD",      "0.45"))

# Labels where classifier wins over crowd_gathering
CLASSIFIER_PRIORITY_OVER_CROWD = {"fighting", "assault", "vehicle_collision", "fire", "shooting"}


@dataclass
class FusionResult:
    incident_type: str
    risk_score: float
    confidence: float
    decision_source: str
    crowd_score: float = 0.0
    risk_model_output: dict = field(default_factory=dict)
    classifier_output: dict = field(default_factory=dict)


def fuse(risk_result, cls_result, crowd_result=None, degraded: str = None) -> FusionResult:
    result = _fuse_inner(risk_result, cls_result, crowd_result, degraded)
    result.crowd_score = round(crowd_result.crowd_score, 4) if crowd_result else 0.0
    return result


def _fuse_inner(risk_result, cls_result, crowd_result=None, degraded: str = None) -> FusionResult:

    risk_out = {
        "autoencoder_score_raw":        risk_result.autoencoder_score_raw        if risk_result else None,
        "autoencoder_score_calibrated": risk_result.autoencoder_score_calibrated if risk_result else None,
        "risk_band":                    risk_result.risk_band                    if risk_result else None,
        "risk_confidence":              risk_result.risk_confidence              if risk_result else None,
        "is_anomaly":                   risk_result.is_anomaly                   if risk_result else None,
        "peak_frame_index":             risk_result.peak_frame_index             if risk_result else None,
    }
    cls_out = {
        "raw_label":               cls_result.raw_label               if cls_result else None,
        "mapped_label":            cls_result.mapped_label            if cls_result else None,
        "classifier_confidence":   cls_result.classifier_confidence   if cls_result else None,
        "classifier_margin":       cls_result.classifier_margin       if cls_result else None,
        "classifier_accepted":     cls_result.classifier_accepted     if cls_result else None,
        "classifier_rejection_reason": cls_result.classifier_rejection_reason if cls_result else None,
        "top_k":                   cls_result.top_k                   if cls_result else None,
    }

    # ── Degraded path ─────────────────────────────────────────────────────────
    if degraded:
        if risk_result and not cls_result:
            label = "unknown_anomaly" if risk_result.is_anomaly else "normal"
            return FusionResult(label, risk_result.autoencoder_score_raw,
                                risk_result.risk_confidence, "degraded", risk_out, cls_out)
        if cls_result and not risk_result:
            return FusionResult(cls_result.mapped_label if cls_result.classifier_accepted else "unknown_anomaly",
                                0.0, cls_result.classifier_confidence, "degraded", risk_out, cls_out)
        return FusionResult("unknown_anomaly", 0.0, 0.0, "degraded", risk_out, cls_out)

    ae_raw    = risk_result.autoencoder_score_raw
    ae_cal    = risk_result.autoencoder_score_calibrated
    cls_label = cls_result.mapped_label
    cls_conf  = cls_result.classifier_confidence
    accepted  = cls_result.classifier_accepted

    # ── Vehicle collision path ────────────────────────────────────────────────
    if cls_label == "vehicle_collision":
        if accepted and ae_cal >= VEHICLE_COLLISION_MIN_RISK:
            accident_score = round(0.6 * cls_conf + 0.4 * ae_cal, 4)
            if accident_score >= 0.65:
                # fused_risk = accident_score (both models contributed)
                logger.info("Fusion: vehicle_collision | acc_score=%.3f ae_cal=%.3f", accident_score, ae_cal)
                return FusionResult("vehicle_collision", accident_score, accident_score,
                                    "agreed", risk_out, cls_out)
        # not accepted or insufficient support -> normal
        # fused_risk = low score reflecting weak combined signal
        fused_risk = round(min(ae_cal * 0.5 + cls_conf * 0.1, 0.4), 4)
        reason = "rejected" if not accepted else "insufficient_risk_support"
        logger.info("Fusion: normal (vehicle_collision %s) fused_risk=%.3f", reason, fused_risk)
        return FusionResult("normal", fused_risk, round(1 - fused_risk, 4), "agreed", risk_out, cls_out)

    # ── All other semantic labels ─────────────────────────────────────────────
    if accepted:
        if ae_cal >= SEMANTIC_MIN_RISK_SUPPORT:
            # fused_risk = weighted blend of calibrated risk and classifier confidence
            fused_risk = round(0.5 * ae_cal + 0.5 * cls_conf, 4)
            source = "agreed" if risk_result.is_anomaly else "classifier_override"
            logger.info("Fusion: %s | fused_risk=%.3f cls_conf=%.3f ae_cal=%.3f",
                        source, fused_risk, cls_conf, ae_cal)
            return FusionResult(cls_label, fused_risk, cls_conf, source, risk_out, cls_out)
        else:
            # classifier accepted but ae support too weak — don't fully trust it
            # fused_risk = ae_cal only (classifier not backed by risk signal)
            logger.info("Fusion: unknown_anomaly (weak ae support) ae_cal=%.3f", ae_cal)
            return FusionResult("unknown_anomaly", ae_cal, ae_cal,
                                "classifier_override_weak", risk_out, cls_out)

    # ── Classifier rejected ───────────────────────────────────────────────────
    if ae_cal >= UNKNOWN_ANOMALY_RISK_THRESH:
        # high calibrated risk but classifier couldn't identify type
        logger.info("Fusion: unknown_anomaly | risk_override ae_cal=%.3f", ae_cal)
        return FusionResult("unknown_anomaly", ae_cal, ae_cal, "risk_override", risk_out, cls_out)

    # ── Crowd gathering check (before falling to normal) ─────────────────────
    if crowd_result and crowd_result.crowd_detected and \
       crowd_result.crowd_score >= CROWD_FUSION_THRESHOLD:
        # Only use crowd label if classifier didn't already win with a priority label
        if not (accepted and cls_label in CLASSIFIER_PRIORITY_OVER_CROWD):
            fused_risk = round(0.5 * crowd_result.crowd_score + 0.5 * ae_cal, 4)
            logger.info("Fusion: crowd_gathering | crowd_score=%.3f ae_cal=%.3f",
                        crowd_result.crowd_score, ae_cal)
            return FusionResult("crowd_gathering", fused_risk, crowd_result.crowd_score,
                                "crowd_override", risk_out, cls_out)

    # both weak -> normal
    fused_risk = round(ae_cal * 0.3 + cls_conf * 0.1, 4)
    logger.info("Fusion: normal | cls_rejected ae_cal=%.3f fused_risk=%.3f", ae_cal, fused_risk)
    return FusionResult("normal", fused_risk, round(1 - fused_risk, 4), "agreed", risk_out, cls_out)

"""
Piecewise linear calibration for Aryan's autoencoder score.
Maps raw [0, 0.5] operating range to fusion-friendly [0, 1].

Anchors (raw -> calibrated):
  0.00 -> 0.00
  0.01 -> 0.05   (very low)
  0.10 -> 0.25   (low-mid)
  0.25 -> 0.60   (medium)
  0.50 -> 1.00   (very high)
  >0.50 -> 1.00  (clip)
"""

# (raw, calibrated) anchor pairs — must be sorted by raw ascending
ANCHORS = [
    (0.00, 0.00),
    (0.01, 0.05),
    (0.10, 0.25),
    (0.25, 0.60),
    (0.50, 1.00),
]


def calibrate(raw_score: float) -> float:
    """Map raw autoencoder score to calibrated [0,1] via piecewise linear interpolation."""
    raw = float(raw_score)

    # clip below/above anchor range
    if raw <= ANCHORS[0][0]:
        return ANCHORS[0][1]
    if raw >= ANCHORS[-1][0]:
        return ANCHORS[-1][1]

    # find segment
    for i in range(len(ANCHORS) - 1):
        r0, c0 = ANCHORS[i]
        r1, c1 = ANCHORS[i + 1]
        if r0 <= raw <= r1:
            t = (raw - r0) / (r1 - r0)
            return round(c0 + t * (c1 - c0), 4)

    return 1.0  # fallback


def risk_band(calibrated: float) -> str:
    if calibrated < 0.25:
        return "low"
    if calibrated < 0.60:
        return "medium"
    return "high"

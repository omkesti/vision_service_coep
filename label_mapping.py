"""
Label mapping with robust normalization.
Strips spaces, underscores, hyphens and lowercases before lookup.
"""

LABEL_MAP = {
    # theft group
    "burglary":    "theft",
    "robbery":     "theft",
    "stealing":    "theft",
    "shoplifting": "theft",
    # vehicle collision — all variants
    "roadaccident":   "vehicle_collision",
    "roadaccidents":  "vehicle_collision",
    "road_accident":  "vehicle_collision",
    "road_accidents": "vehicle_collision",
    # fire group
    "explosion": "fire",
    "arson":     "fire",
    # direct mappings
    "abuse":     "abuse",
    "arrest":    "arrest",
    "assault":   "assault",
    "fighting":  "fighting",
    "shooting":  "shooting",
    "vandalism": "vandalism",
    # normal
    "normalvideos": "normal",
    "normal":       "normal",
}


def _normalize_key(raw: str) -> str:
    """Lowercase, strip spaces/underscores/hyphens for robust matching."""
    return raw.lower().replace(" ", "").replace("_", "").replace("-", "").strip()


def normalize_label(raw_label: str) -> str:
    return LABEL_MAP.get(_normalize_key(raw_label), "unknown_anomaly")

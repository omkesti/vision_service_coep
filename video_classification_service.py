"""
VideoMAE-based anomaly type classifier.
Loads model once, classifies a short clip around the peak anomaly frame.
"""

import os
import logging
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from label_mapping import normalize_label

logger = logging.getLogger("video_cls")

MODEL_NAME      = os.getenv("VIDEO_CLS_MODEL_NAME",   "OPear/videomae-large-finetuned-UCF-Crime")
NUM_FRAMES      = int(os.getenv("VIDEO_CLS_NUM_FRAMES",    "16"))
FRAME_SIZE      = int(os.getenv("VIDEO_CLS_FRAME_SIZE",   "224"))
TOP_K           = int(os.getenv("VIDEO_CLS_TOP_K",          "3"))
MIN_CONFIDENCE  = float(os.getenv("VIDEO_CLS_MIN_CONFIDENCE", "0.40"))
DEVICE_CFG      = os.getenv("VIDEO_CLS_DEVICE", "auto")

# Window around peak to sample from (frames)
WINDOW_BEFORE   = int(os.getenv("VIDEO_CLS_WINDOW_BEFORE", "30"))
WINDOW_AFTER    = int(os.getenv("VIDEO_CLS_WINDOW_AFTER",  "30"))


class VideoClassificationService:

    def __init__(self):
        from transformers import VideoMAEVideoProcessor, VideoMAEForVideoClassification

        if DEVICE_CFG == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(DEVICE_CFG)

        logger.info("Loading VideoMAE: %s on %s", MODEL_NAME, self.device)
        self.processor = VideoMAEVideoProcessor.from_pretrained(MODEL_NAME)
        self.model = VideoMAEForVideoClassification.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()
        logger.info("VideoMAE ready | labels: %d", self.model.config.num_labels)

    def _sample_frames(self, video_path: str, peak_frame: int) -> list:
        """
        Sample NUM_FRAMES evenly from the window around peak_frame.
        Falls back to full video if window is too short.
        Pads by repeating last frame if needed.
        """
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start = max(0, peak_frame - WINDOW_BEFORE)
        end   = min(total - 1, peak_frame + WINDOW_AFTER)

        # fallback to full video if window too small
        if (end - start + 1) < NUM_FRAMES:
            start, end = 0, total - 1

        indices = np.linspace(start, end, NUM_FRAMES, dtype=int)
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        cap.release()

        # pad with last frame if short
        while len(frames) < NUM_FRAMES:
            frames.append(frames[-1] if frames else np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8))

        return frames[:NUM_FRAMES]

    def classify(self, video_path: str, peak_frame: int) -> dict:
        """
        Run VideoMAE on the peak window.
        Returns dict with anomaly_type, anomaly_type_confidence, anomaly_type_top_k.
        """
        frames = self._sample_frames(video_path, peak_frame)

        inputs = self.processor(list(frames), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = F.softmax(outputs.logits, dim=-1)[0]
        top_k_vals, top_k_ids = torch.topk(probs, min(TOP_K, len(probs)))

        top_k = []
        for val, idx in zip(top_k_vals.tolist(), top_k_ids.tolist()):
            raw_label = self.model.config.id2label[idx]
            top_k.append({
                "label": normalize_label(raw_label),
                "raw_label": raw_label,
                "confidence": round(val, 4),
            })

        top1 = top_k[0]
        anomaly_type = top1["label"]
        confidence   = top1["confidence"]

        # conflict: risk model says anomaly but VideoMAE says normal
        if anomaly_type == "normal":
            anomaly_type = "unknown_or_normal_conflict"

        logger.info("VideoMAE top-1: %s (%.3f) | top-k: %s", anomaly_type, confidence, top_k)

        return {
            "anomaly_type":            anomaly_type,
            "anomaly_type_confidence": round(confidence, 4),
            "anomaly_type_top_k":      top_k,
        }

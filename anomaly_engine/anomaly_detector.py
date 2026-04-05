import torch
import cv2
import numpy as np
from collections import deque

from .autoencoder import ConvolutionalAutoencoder

class AnomalyDetector:
    def __init__(self, model_path, threshold=0.005, smoothing_window=15):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # load model
        self.model = ConvolutionalAutoencoder().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state_dict)
        ckpt_threshold = checkpoint.get("threshold") if isinstance(checkpoint, dict) else None
        self.threshold = float(ckpt_threshold) if ckpt_threshold is not None else float(threshold)
        self.baseline = None
        self.buffer = deque(maxlen=max(1, int(smoothing_window)))
        self.model.eval()

    def preprocess(self, frame):
        frame = cv2.resize(frame, (64, 64))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame / 255.0

        frame = np.expand_dims(frame, axis=0)  # (1, 64, 64)
        frame = np.expand_dims(frame, axis=0)  # (1, 1, 64, 64)

        return torch.tensor(frame, dtype=torch.float32).to(self.device)

    def get_error(self, frame):
        input_tensor = self.preprocess(frame)

        with torch.no_grad():
            reconstructed = self.model(input_tensor)

        error = torch.mean((input_tensor - reconstructed) ** 2).item()
        return error

    def calibrate_baseline_from_video(self, video_path, num_frames=30):
        cap = cv2.VideoCapture(str(video_path))
        errors = []

        while len(errors) < num_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            errors.append(self.get_error(frame))

        cap.release()

        if not errors:
            raise RuntimeError("Could not calibrate baseline: no readable frames found.")

        self.baseline = float(np.mean(errors))
        return self.baseline

    def _normalize(self, error):
        # Map error to [0, 1] using threshold when baseline is unavailable.
        return min(error / max(self.threshold, 1e-8), 1.0)

    def _apply_smoothing(self, score):
        self.buffer.append(float(score))
        return float(sum(self.buffer) / len(self.buffer))

    def get_score(self, frame):
        error = self.get_error(frame)

        if self.baseline is not None:
            score = max(0.0, (error - self.baseline) / max(self.baseline, 1e-8))
            score = min(score, 1.0)
        else:
            score = self._normalize(error)

        return self._apply_smoothing(score)
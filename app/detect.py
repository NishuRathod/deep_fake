import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

class DeepFakeDetector:
    def __init__(self, model_path: str, frame_sample_rate: int = 5):
        self.model_path = Path(model_path)
        self.frame_sample_rate = frame_sample_rate
        self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.model = tf.keras.models.load_model(self.model_path)

        # Model expects (None, 3, 160, 160, 3)
        self.frames_required = self.model.input_shape[1]
        self.input_h = self.model.input_shape[2]
        self.input_w = self.model.input_shape[3]

    def preprocess_frame(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_w, self.input_h))
        img = img.astype("float32") / 255.0
        return img  # shape: (H, W, 3)

    def predict_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        scores = []
        frames_buffer = []
        frame_idx = 0

        if not cap.isOpened():
            raise RuntimeError("Unable to open video file.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.frame_sample_rate == 0:
                processed = self.preprocess_frame(frame)
                frames_buffer.append(processed)

                # When enough frames collected → predict
                if len(frames_buffer) == self.frames_required:
                    x = np.array(frames_buffer)                 # (3, H, W, 3)
                    x = np.expand_dims(x, axis=0)              # (1, 3, H, W, 3)

                    pred = self.model.predict(x, verbose=0)
                    score = float(pred.squeeze())
                    scores.append(score)

                    # Clear buffer for next chunk of frames
                    frames_buffer = []

            frame_idx += 1

        cap.release()

        if not scores:
            raise RuntimeError("No valid frame batches processed.")

        return float(sum(scores) / len(scores))


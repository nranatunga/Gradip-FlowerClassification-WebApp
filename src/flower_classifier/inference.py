"""Load saved model and run predictions (Gradio-compatible)."""

from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from flower_classifier.config import CLASS_NAMES, IMG_HEIGHT, IMG_WIDTH, default_model_path


class FlowerClassifier:
    def __init__(self, model_path: Path | str | None = None):
        path = Path(model_path) if model_path else default_model_path()
        if not path.is_file():
            raise FileNotFoundError(
                f"Model not found at {path}. Train first: "
                f"python -m flower_classifier.train"
            )
        self.model = tf.keras.models.load_model(path)
        self.class_names = CLASS_NAMES

    def predict_proba(self, image: np.ndarray) -> dict[str, float]:
        """`image`: HWC RGB uint8 or float; resized to model input if needed."""
        x = self._prepare_batch(image)
        pred = self.model.predict(x, verbose=0)[0]
        return {self.class_names[i]: float(pred[i]) for i in range(len(self.class_names))}

    def prepare_rgb_uint8(self, image: np.ndarray) -> np.ndarray:
        """Resize/crop to model input; HWC RGB uint8 (same geometry as model sees)."""
        if image is None:
            raise ValueError("No image provided.")
        if image.ndim != 3 or image.shape[-1] not in (3, 4):
            raise ValueError("Expected an HWC image array.")
        rgb = image[..., :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        pil = Image.fromarray(rgb)
        pil = pil.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.BILINEAR)
        return np.asarray(pil, dtype=np.uint8)

    def _prepare_batch(self, image: np.ndarray) -> np.ndarray:
        arr = self.prepare_rgb_uint8(image).astype(np.float32)
        return arr.reshape(1, IMG_HEIGHT, IMG_WIDTH, 3)


def load_classifier(model_path: Path | str | None = None) -> FlowerClassifier:
    return FlowerClassifier(model_path)

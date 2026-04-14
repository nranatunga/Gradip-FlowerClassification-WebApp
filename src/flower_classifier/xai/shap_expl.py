"""SHAP image explanations via ``shap.Explainer`` + ``shap.maskers.Image``."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from flower_classifier.config import IMG_HEIGHT, IMG_WIDTH

if TYPE_CHECKING:
    import tensorflow as tf


def run_shap_image(
    model: "tf.keras.Model",
    rgb_uint8: np.ndarray,
    class_names: tuple[str, ...],
    class_idx: int,
    *,
    max_evals: int = 150,
    blur_size: int = 11,
) -> tuple[np.ndarray, str]:
    """
    Build a matplotlib SHAP image plot and return it as RGB uint8 for Gradio.

    Uses ``shap.Explainer`` with a blur ``Image`` masker (see SHAP docs). The
    explainer partitions superpixels and can be slow; lower ``max_evals`` for speed.
    """
    import shap
    import shap.plots

    if rgb_uint8.shape != (IMG_HEIGHT, IMG_WIDTH, 3):
        raise ValueError(f"Expected {(IMG_HEIGHT, IMG_WIDTH, 3)}, got {rgb_uint8.shape}")

    def predict_fn(x: np.ndarray) -> np.ndarray:
        return np.asarray(model.predict(x, verbose=0), dtype=np.float64)

    b = max(3, int(blur_size))
    if b % 2 == 0:
        b += 1
    masker = shap.maskers.Image(
        f"blur({b},{b})",
        (IMG_HEIGHT, IMG_WIDTH, 3),
    )
    explainer = shap.Explainer(
        predict_fn,
        masker,
        output_names=list(class_names),
    )
    batch = rgb_uint8[np.newaxis, ...].astype(np.float32)
    explanation = explainer(batch, max_evals=max_evals)

    # One panel: SHAP values for the selected class only (shape H×W×C)
    shap_for_class = explanation.values[0, :, :, :, class_idx]
    pixels = explanation.data[0]

    shap.plots.image(
        [shap_for_class[np.newaxis, ...]],
        pixel_values=pixels[np.newaxis, ...],
        labels=np.array([[class_names[class_idx]]]),
        show=False,
    )
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    plot_rgb = np.asarray(Image.open(buf).convert("RGB"))

    note = (
        f"**Class:** {class_names[class_idx]} - **max_evals:** {max_evals} - "
        f"**mask:** blur({b},{b}) on {IMG_HEIGHT}x{IMG_WIDTH} RGB."
    )
    return plot_rgb, note

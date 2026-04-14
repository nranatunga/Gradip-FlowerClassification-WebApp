"""Grad-CAM on the last convolutional layer (portfolio-friendly, no extra DL deps)."""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def _forward_conv_and_logits(
    model: tf.keras.Model, inputs: tf.Tensor, layer_name: str
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Forward through all layers except the final activation, then apply the last
    `Dense` as **linear** to get pre-softmax logits.

    Grad-CAM uses logits, not softmax outputs: ∂(softmax prob)/∂(conv) often
    vanishes when the model is confident, which yields an empty heatmap (all
    zeros after ReLU → solid dark blue in jet).
    """
    layers_list = model.layers
    if len(layers_list) < 2:
        raise ValueError("Model must have at least two layers.")
    last = layers_list[-1]
    if not isinstance(last, tf.keras.layers.Dense):
        raise TypeError("Grad-CAM logit path expects the last layer to be Dense.")

    x = inputs
    conv_out = None
    for layer in layers_list[:-1]:
        x = layer(x)
        if layer.name == layer_name:
            conv_out = x
    if conv_out is None:
        raise ValueError(f"Layer {layer_name!r} not found on model.")
    # `x` is the vector fed into the final Dense (pre-softmax).
    logits = tf.nn.bias_add(tf.matmul(x, last.kernel), last.bias)
    return conv_out, logits


def _jet_colormap(hmap: np.ndarray) -> np.ndarray:
    try:
        from matplotlib import colormaps

        cmap = colormaps["jet"]
    except Exception:
        from matplotlib import pyplot as plt

        cmap = plt.cm.jet
    rgba = cmap(hmap)
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    return rgb


def compute_gradcam(
    model: tf.keras.Model,
    image_batch: tf.Tensor,
    class_idx: int,
    layer_name: str,
) -> np.ndarray:
    """Return HxW heatmap (float 0–1) at the spatial size of `conv_out`, then resized to input."""
    with tf.GradientTape() as tape:
        conv_out, logits = _forward_conv_and_logits(model, image_batch, layer_name)
        class_channel = logits[0, class_idx]

    grads = tape.gradient(class_channel, conv_out)
    if grads is None:
        raise RuntimeError("Gradients are None; check model and class index.")

    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
    heatmap = tf.reduce_sum(
        tf.multiply(conv_out, pooled_grads[:, tf.newaxis, tf.newaxis, :]),
        axis=-1,
    )
    heatmap = tf.nn.relu(heatmap)
    h = heatmap[0].numpy()
    h_min, h_max = np.min(h), np.max(h)
    if h_max - h_min < 1e-8:
        h = np.zeros_like(h, dtype=np.float32)
    else:
        h = (h - h_min) / (h_max - h_min)

    h_tensor = tf.constant(h, dtype=tf.float32)[..., tf.newaxis]
    target_h = tf.shape(image_batch)[1]
    target_w = tf.shape(image_batch)[2]
    h_up = tf.image.resize(h_tensor, [target_h, target_w], method="bilinear")
    return h_up.numpy().squeeze()


def heatmap_to_rgb(hmap: np.ndarray) -> np.ndarray:
    """HxW float 0–1 -> HxW3 uint8 colormap."""
    return _jet_colormap(hmap)


def overlay_heatmap(
    base_rgb_uint8: np.ndarray, heatmap_rgb_uint8: np.ndarray, alpha: float = 0.45
) -> np.ndarray:
    """`base_rgb_uint8`: HW3. Blend with heatmap."""
    base = base_rgb_uint8.astype(np.float32)
    heat = heatmap_rgb_uint8.astype(np.float32)
    out = np.clip(base * (1.0 - alpha) + heat * alpha, 0, 255).astype(np.uint8)
    return out


def explain_image(
    model: tf.keras.Model,
    image_hwc_rgb_uint8: np.ndarray,
    class_idx: int,
    layer_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        overlay_rgb uint8 HW3,
        heatmap_rgb uint8 HW3 (jet),
        raw_float_hw2 grayscale heatmap 0–1 at input resolution.
    """
    x = tf.constant(image_hwc_rgb_uint8[np.newaxis, ...], dtype=tf.float32)
    hmap = compute_gradcam(model, x, class_idx, layer_name)
    heat_rgb = heatmap_to_rgb(hmap)
    overlay = overlay_heatmap(image_hwc_rgb_uint8, heat_rgb)
    return overlay, heat_rgb, hmap

"""Gradio UI: classification + Grad-CAM + SHAP explain tabs."""

from __future__ import annotations

import os

import gradio as gr


def _gradio_share() -> bool:
    """
    Public *.gradio.live (or similar) tunnel via Gradio's servers.

    Set ``GRADIO_SHARE=0`` (or ``false`` / ``no``) for localhost-only.
    Unset or ``1`` / ``true``: enable sharing (default: on for easy demos).

    Tunnel lifetime is controlled by Gradio's infrastructure, not this app—often
    on the order of a few days, not guaranteed. There is no API to set exactly 3 days.
    """
    v = os.environ.get("GRADIO_SHARE")
    if v is None:
        return True
    return v.strip().lower() in ("1", "true", "yes", "on")

from flower_classifier.config import CLASS_NAMES, LAST_CONV_LAYER_NAME
from flower_classifier.inference import FlowerClassifier, load_classifier
from flower_classifier.xai.gradcam import explain_image
from flower_classifier.xai.shap_expl import run_shap_image

EXPLAIN_BLURB = """
### Grad-CAM (Gradient-weighted Class Activation Mapping)

- **What you see:** A heatmap over the input image showing which regions most influenced the **selected class** score (via the last conv layer, `{layer}`). Gradients are taken w.r.t. **pre-softmax logits** for that class (standard Grad-CAM), not softmax probabilities—so maps stay informative when the model is confident.
- **Limitation:** This is a *local* explanation for one image and one class label—not a proof of causal reasoning or dataset-wide fairness.

""".format(
    layer=LAST_CONV_LAYER_NAME,
)

SHAP_BLURB = """
### SHAP ([`shap.Explainer`](https://shap.readthedocs.io/en/latest/generated/shap.Explainer.html))

- **What you see:** Approximate **Shapley attributions** for pixels (via a partition explainer and a blur **Image** masker), for the **selected class** output. The figure shows the input and a heatmap of SHAP values for that class.
- **Speed:** Larger **max_evals** is more accurate but slower. This is much heavier than Grad-CAM.
- **Limitation:** Estimates depend on the masker and evaluation budget; not an exact Shapley value for every pixel.

"""


def main() -> None:
    try:
        clf: FlowerClassifier | None = load_classifier()
        load_error: str | None = None
    except FileNotFoundError as e:
        clf = None
        load_error = str(e)

    def classify_fn(img):
        if clf is None:
            return {}, gr.update(), gr.update(), gr.update()
        if img is None:
            return {}, gr.update(), gr.update(), gr.update()
        probs = clf.predict_proba(img)
        top = max(probs, key=probs.get)
        u = gr.update(value=top)
        return probs, u, u, u

    def explain_fn(img, class_name: str):
        if clf is None:
            return None, None, f"**Model not loaded.**\n\n{load_error or ''}\n\n{EXPLAIN_BLURB}"
        if img is None:
            return None, None, EXPLAIN_BLURB
        idx = CLASS_NAMES.index(class_name)
        rgb = clf.prepare_rgb_uint8(img)
        overlay, heat_rgb, _ = explain_image(
            clf.model, rgb, idx, LAST_CONV_LAYER_NAME
        )
        return overlay, heat_rgb, EXPLAIN_BLURB

    def shap_fn(img, class_name: str, max_evals: float, blur_size: float):
        if clf is None:
            return None, f"**Model not loaded.**\n\n{load_error or ''}\n\n{SHAP_BLURB}"
        if img is None:
            return None, SHAP_BLURB
        idx = CLASS_NAMES.index(class_name)
        rgb = clf.prepare_rgb_uint8(img)
        try:
            plot_img, note = run_shap_image(
                clf.model,
                rgb,
                CLASS_NAMES,
                idx,
                max_evals=int(max_evals),
                blur_size=int(blur_size),
            )
            return plot_img, f"{note}\n\n{SHAP_BLURB}"
        except Exception as e:
            return None, f"**SHAP error:** `{e!s}`\n\n{SHAP_BLURB}"

    with gr.Blocks(title="Flower classification") as demo:
        if load_error:
            gr.Markdown(
                f"**Warning:** {load_error}\n\n"
                "Train a model first:\n\n"
                "`python -m flower_classifier.train --epochs 10`"
            )
        gr.Markdown("# Flower classifier + explainability (Grad-CAM & SHAP)")
        gr.Markdown(
            "Upload **one image** below; use the tabs for probabilities, Grad-CAM, or SHAP."
        )

        img_in = gr.Image(type="numpy", label="Flower image")

        with gr.Tabs():
            with gr.Tab("Classify"):
                out_label = gr.Label(label="Class probabilities", num_top_classes=5)
                class_guess = gr.Dropdown(
                    choices=list(CLASS_NAMES),
                    value=CLASS_NAMES[0],
                    label="Top guess (updates after classify)",
                )

            with gr.Tab("Explain (Grad-CAM)"):
                gr.Markdown(
                    "Pick which **class** to explain (syncs with the top guess when you upload)."
                )
                ex_class = gr.Dropdown(
                    choices=list(CLASS_NAMES),
                    value=CLASS_NAMES[0],
                    label="Explain class",
                )
                with gr.Row():
                    ex_overlay = gr.Image(type="numpy", label="Overlay")
                    ex_heat = gr.Image(type="numpy", label="Heatmap")
                ex_md = gr.Markdown(EXPLAIN_BLURB)
                ex_btn = gr.Button("Run Grad-CAM")
                ex_btn.click(
                    explain_fn,
                    [img_in, ex_class],
                    [ex_overlay, ex_heat, ex_md],
                )

            with gr.Tab("Explain (SHAP)"):
                gr.Markdown(
                    "Uses [`shap.Explainer`](https://shap.readthedocs.io/en/latest/generated/shap.Explainer.html) "
                    "with an **Image** blur masker. Can take **tens of seconds to a few minutes** on CPU."
                )
                shap_class = gr.Dropdown(
                    choices=list(CLASS_NAMES),
                    value=CLASS_NAMES[0],
                    label="Class output to explain",
                )
                with gr.Row():
                    shap_max_evals = gr.Slider(
                        40,
                        350,
                        value=120,
                        step=10,
                        label="max_evals (higher = slower, often more stable)",
                    )
                    shap_blur = gr.Slider(
                        5,
                        21,
                        value=11,
                        step=2,
                        label="Blur mask kernel size (odd)",
                    )
                shap_plot = gr.Image(type="numpy", label="SHAP plot")
                shap_md = gr.Markdown(SHAP_BLURB)
                shap_btn = gr.Button("Run SHAP")
                shap_btn.click(
                    shap_fn,
                    [img_in, shap_class, shap_max_evals, shap_blur],
                    [shap_plot, shap_md],
                )

        img_in.change(
            classify_fn,
            img_in,
            [out_label, class_guess, ex_class, shap_class],
        )

        gr.Markdown(
            "Same CNN as the TensorFlow flowers tutorial: 180×180 RGB, five flower classes."
        )

    share = _gradio_share()
    if share:
        print(
            "Gradio public sharing is ON. After launch, use the printed 'Running on public URL' "
            "link or QR from the terminal. Set GRADIO_SHARE=0 for localhost only. "
            "Tunnel expiry is decided by Gradio (often ~1–3 days; not configurable here)."
        )
    demo.launch(share=share)


if __name__ == "__main__":
    main()

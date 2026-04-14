# Flower classification with Gradio and explainable AI (XAI)

A small **portfolio-style** project that trains a convolutional neural network on the TensorFlow **flower photos** dataset and serves it through a **Gradio** web UI. Beyond raw predictions, the app compares two **explainability** approaches—**Grad-CAM** (gradient-based, local) and **SHAP** (game-theoretic attributions via `shap.Explainer`)—so you can see *where* the model looks in an image, not only *what* it predicts.

<img width="716" height="367" alt="gradio " src="https://github.com/user-attachments/assets/5db2055c-63b3-4f69-8cc2-f6c2b74e507c" />


---

## What this project demonstrates

This work is meant to show hands-on familiarity with ideas that matter in applied ML and responsible AI:

| Area | What we practiced |
|------|-------------------|
| **Deep learning** | End-to-end image pipeline: data loading, CNN (`Conv2D` / pooling / dense), training, saving artifacts (`.keras`). |
| **Model serving** | Packaging inference behind a clean API and loading weights in a reproducible layout. |
| **Gradio** | Building a multi-tab **Blocks** UI: classification outputs, wiring events, and optional **public share links** for demos. |
| **Explainable AI (XAI)** | **Grad-CAM**: class-discriminative saliency on a target conv layer using **pre-softmax logits** (avoids vanishing gradients through softmax). **SHAP**: `shap.Explainer` with an **Image** masker—understanding Shapley-style attributions, cost vs. accuracy (`max_evals`), and honest limitations of each method. |
| **Software engineering** | Modular Python package under `src/`, pinned dependencies, training script vs. app entry points, `.gitignore` for large model files. |

We can articulate **trade-offs**: Grad-CAM is fast and standard for CNNs; SHAP image explanations are heavier and depend on the masker and evaluation budget. Neither is “ground truth”—both support **human review** and **debugging**, not automatic fairness or causality.


<img width="685" height="850" alt="gradio 1 daisy" src="https://github.com/user-attachments/assets/45e57edf-e630-47ac-990c-153fe28a7acb" />

<img width="708" height="623" alt="gradio 2 daisy grad cam" src="https://github.com/user-attachments/assets/2e0aeea7-1380-4d74-9b16-34fb8c346d90" />

---

## Features

- **Classify** — Upload a 180×180 RGB flower image; view calibrated class probabilities across five labels (daisy, dandelion, roses, sunflowers, tulips).
- **Explain (Grad-CAM)** — Heatmap and overlay for a **selected class**, tied to the last convolutional layer.
- **Explain (SHAP)** — Partition explainer + blur **Image** masker; configurable `max_evals` and blur kernel; matplotlib figure embedded in the UI.
- **Optional public URL** — Gradio tunnel for temporary sharing (see below).

---

## Tech stack

- Python 3.10+
- TensorFlow / Keras (CNN training and inference)
- Gradio (web UI)
- NumPy, Pillow, Matplotlib
- SHAP, OpenCV (headless) for image masker support

---

## Quick start

### 1. Install

From the repository root:

```bash
python -m pip install -e .
```

Or: `python -m pip install -r requirements.txt`

### 2. Train (produces `models/flower_model.keras`)

The model file is **not** committed to git (large binary). Train once locally:

```bash
python -m flower_classifier.train --epochs 10
```

For a quick smoke test: `--epochs 1`.

### 3. Run the app

```bash
python -m flower_classifier.app
```

Or without an editable install:

```bash
python gradiodemo.py
```

Open the local URL Gradio prints (usually `http://127.0.0.1:7860`).

### Public demo link (temporary)

Gradio can create a short-lived public URL (e.g. `*.gradio.live`) for sharing or QR codes. **On by default** unless you disable it:

- **Enable** (default): unset `GRADIO_SHARE`, or set `GRADIO_SHARE=1` / `true`.
- **Localhost only**: `GRADIO_SHARE=0` (or `false` / `no`).

Example (PowerShell, local only):

```powershell
$env:GRADIO_SHARE="0"; python -m flower_classifier.app
```

Tunnel lifetime is controlled by **Gradio’s servers**, not this repo—often on the order of **one to a few days**, not guaranteed.

---

## Project layout

```
src/flower_classifier/
  app.py          # Gradio UI (Classify, Grad-CAM, SHAP)
  config.py       # Image size, class names, paths
  model.py        # CNN definition
  dataset.py      # TF flowers download + tf.data pipelines
  train.py        # Training CLI
  inference.py    # Load model, predict, preprocess for XAI
  xai/
    gradcam.py    # Grad-CAM (logit target, eager layer loop for Keras 3)
    shap_expl.py  # shap.Explainer + Image masker + plot
models/           # Put trained `flower_model.keras` here (gitignored)
```

---

## Original tutorial context

This project extends the workflow from the classic TensorFlow **flower classification** tutorial (small CNN on the public flower_photos archive). An older walkthrough video from the repo author: [YouTube](https://www.youtube.com/watch?v=aZ4wV4V_p9E).

---

## License

Add a `LICENSE` file if you open-source the repo; none is bundled here by default.

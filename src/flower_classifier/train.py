"""Train the flower CNN and save weights (Colab notebook workflow)."""

import argparse
from pathlib import Path

import tensorflow as tf

from flower_classifier.config import EPOCHS_DEFAULT, default_model_path
from flower_classifier.dataset import load_train_val_datasets
from flower_classifier.model import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train flower classifier on TF flowers.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS_DEFAULT,
        help=f"Training epochs (default {EPOCHS_DEFAULT}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save .keras model (default: models/flower_model.keras).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Optional path to flower_photos directory; otherwise download via TF.",
    )
    args = parser.parse_args()

    out = args.output or default_model_path()
    out.parent.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = load_train_val_datasets(args.data_dir)
    model = build_model()
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)
    model.save(out)
    print(f"Saved model to {out.resolve()}")


if __name__ == "__main__":
    main()

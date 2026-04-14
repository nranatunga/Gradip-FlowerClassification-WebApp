"""CNN architecture from the TensorFlow flowers tutorial / Colab notebook."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from flower_classifier.config import (
    IMG_HEIGHT,
    IMG_WIDTH,
    LAST_CONV_LAYER_NAME,
    NUM_CLASSES,
)


def build_model() -> keras.Sequential:
    return keras.Sequential(
        [
            layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(
                64, 3, padding="same", activation="relu", name=LAST_CONV_LAYER_NAME
            ),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="flower_cnn",
    )

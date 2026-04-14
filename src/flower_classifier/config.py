"""Hyperparameters and paths aligned with the original Colab notebook."""

from pathlib import Path

DATASET_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
)
DATASET_LOCAL_NAME = "flower_photos"

IMG_HEIGHT = 180
IMG_WIDTH = 180
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
SEED = 123
NUM_CLASSES = 5
EPOCHS_DEFAULT = 10

# Sorted as in tf.keras.preprocessing.image_dataset_from_directory
CLASS_NAMES = ("daisy", "dandelion", "roses", "sunflowers", "tulips")

LAST_CONV_LAYER_NAME = "last_conv"


def project_root() -> Path:
    """Repository root (parent of `src`)."""
    return Path(__file__).resolve().parents[2]


def default_model_path() -> Path:
    return project_root() / "models" / "flower_model.keras"

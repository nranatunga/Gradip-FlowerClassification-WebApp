"""Download TF flowers and build train/validation `tf.data` pipelines."""

import pathlib

import tensorflow as tf

from flower_classifier.config import (
    BATCH_SIZE,
    DATASET_LOCAL_NAME,
    DATASET_URL,
    IMG_HEIGHT,
    IMG_WIDTH,
    SEED,
    VALIDATION_SPLIT,
)


def download_flowers_tgz() -> pathlib.Path:
    archive = tf.keras.utils.get_file(
        DATASET_LOCAL_NAME, origin=DATASET_URL, untar=True
    )
    return pathlib.Path(archive)


def load_train_val_datasets(
    data_dir: pathlib.Path | str | None = None,
):
    """Return `(train_ds, val_ds)` with batching and prefetching."""
    if data_dir is None:
        data_dir = download_flowers_tgz()
    data_dir = pathlib.Path(data_dir)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)
    return train_ds, val_ds

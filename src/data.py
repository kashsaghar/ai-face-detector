import json
import os

import tensorflow as tf
from tensorflow.keras import layers

from src.config import BATCH_SIZE, DATA_DIR, METRICS_DIR, SEED, TRAIN_SUBSET

AUGMENTATION = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomBrightness(0.1),
        layers.RandomContrast(0.1),
    ],
    name="augmentation",
)


def _load_split(split, img_size, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, split),
        label_mode="binary",
        image_size=img_size,
        batch_size=BATCH_SIZE,
        seed=SEED,
        shuffle=shuffle,
    )


def get_datasets(img_size, preprocess=None):
    """Returns train_ds, val_ds, test_ds, class_names.

    Classes are alphabetical, so class_names == ["fake", "real"]:
    fake = 0, real = 1. The model outputs P(real).
    """
    train_ds = _load_split("train", img_size, shuffle=True)
    val_ds = _load_split("valid", img_size, shuffle=False)
    test_ds = _load_split("test", img_size, shuffle=False)

    class_names = train_ds.class_names
    print("class_names:", class_names)
    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(os.path.join(METRICS_DIR, "class_names.json"), "w") as f:
        json.dump(class_names, f)

    if TRAIN_SUBSET:
        train_ds = train_ds.unbatch().take(TRAIN_SUBSET).batch(BATCH_SIZE)

    train_ds = train_ds.map(
        lambda x, y: (AUGMENTATION(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if preprocess is not None:
        train_ds = train_ds.map(
            lambda x, y: (preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )
        val_ds = val_ds.map(
            lambda x, y: (preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )
        test_ds = test_ds.map(
            lambda x, y: (preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )

    # The full dataset does not fit comfortably in memory, so only prefetch.
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


if __name__ == "__main__":
    from src.config import CNN_IMG_SIZE

    train_ds, val_ds, test_ds, class_names = get_datasets(CNN_IMG_SIZE)
    for x, y in train_ds.take(1):
        print("batch shape:", x.shape, "label shape:", y.shape)
        print("label values:", y.numpy().flatten()[:10])

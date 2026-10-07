"""``tf.data`` input pipelines built from the manifest.

Design notes
------------
* Images are decoded in parallel, resized once, and (optionally) cached in
  RAM. On EuroSAT (27k 64x64 tiles) the whole cache is ~1.3 GB at 128x128
  float32, so ``data.cache`` can be turned off in ``config.yaml`` on small
  machines.
* Augmentation runs **only on the training split** and is applied as a
  batched Keras preprocessing stack, which keeps it on the same device as
  training and off the Python main thread.
* Pixels stay in ``[0, 255]``. Backbone-specific normalisation lives
  inside the model (see :mod:`src.models.build`) so that exported models
  are self-contained: the web app can feed them raw uint8 images.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from ..config import Config, load_config
from ..utils import LOGGER
from .preprocess import load_manifest, resolve_path

AUTOTUNE = tf.data.AUTOTUNE


# ----------------------------------------------------------------------
# Decoding
# ----------------------------------------------------------------------
def _decode(path: tf.Tensor, image_size: Tuple[int, int]) -> tf.Tensor:
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, image_size, method="bilinear")
    img = tf.cast(img, tf.float32)
    img.set_shape([image_size[0], image_size[1], 3])
    return img


def build_augmenter(cfg: Config) -> tf.keras.Sequential:
    """Keras random-augmentation stack described by ``augment:`` in config."""
    a = cfg["augment"]
    layers = tf.keras.layers
    stack = []

    if a.get("horizontal_flip", True) and a.get("vertical_flip", True):
        stack.append(layers.RandomFlip("horizontal_and_vertical"))
    elif a.get("horizontal_flip", True):
        stack.append(layers.RandomFlip("horizontal"))
    elif a.get("vertical_flip", False):
        stack.append(layers.RandomFlip("vertical"))

    if float(a.get("rotation", 0)) > 0:
        stack.append(layers.RandomRotation(float(a["rotation"]), fill_mode="reflect"))
    if float(a.get("zoom", 0)) > 0:
        stack.append(layers.RandomZoom(float(a["zoom"]), fill_mode="reflect"))
    if float(a.get("translation", 0)) > 0:
        t = float(a["translation"])
        stack.append(layers.RandomTranslation(t, t, fill_mode="reflect"))
    if float(a.get("brightness", 0)) > 0:
        stack.append(layers.RandomBrightness(float(a["brightness"]), value_range=(0.0, 255.0)))
    if float(a.get("contrast", 0)) > 0:
        stack.append(layers.RandomContrast(float(a["contrast"])))

    return tf.keras.Sequential(stack, name="augmentation")


def _encode_labels(cfg: Config, label_idx: np.ndarray) -> np.ndarray:
    if cfg.is_binary:
        return label_idx.astype("float32")
    return tf.keras.utils.to_categorical(label_idx, num_classes=cfg.num_classes).astype("float32")


# ----------------------------------------------------------------------
# Public builders
# ----------------------------------------------------------------------
def make_dataset(
    cfg: Optional[Config] = None,
    split: str = "train",
    df: Optional[pd.DataFrame] = None,
    shuffle: Optional[bool] = None,
    augment: Optional[bool] = None,
    batch_size: Optional[int] = None,
    repeat: bool = False,
    drop_remainder: bool = False,
) -> tf.data.Dataset:
    """Build the ``tf.data.Dataset`` for one split of the manifest."""
    cfg = cfg or load_config()
    df = load_manifest(cfg) if df is None else df
    sub = df[df["split"] == split].reset_index(drop=True)
    if len(sub) == 0:
        raise ValueError(f"Split '{split}' is empty in the manifest.")

    shuffle = (split == "train") if shuffle is None else shuffle
    augment = (split == "train") if augment is None else augment
    batch_size = int(batch_size or cfg["data"]["batch_size"])
    image_size = cfg.image_size

    paths = np.array([str(resolve_path(p)) for p in sub["path"]], dtype=object)
    labels = _encode_labels(cfg, sub["label_idx"].to_numpy())

    ds = tf.data.Dataset.from_tensor_slices((paths.astype(str), labels))
    ds = ds.map(lambda p, y: (_decode(p, image_size), y), num_parallel_calls=AUTOTUNE)

    if bool(cfg["data"].get("cache", True)):
        ds = ds.cache()
    if shuffle:
        buf = min(int(cfg["data"].get("shuffle_buffer", 4096)), len(sub))
        ds = ds.shuffle(buf, seed=cfg.seed, reshuffle_each_iteration=True)
    if repeat:
        ds = ds.repeat()

    ds = ds.batch(batch_size, drop_remainder=drop_remainder)

    if augment:
        augmenter = build_augmenter(cfg)
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)

    LOGGER.info("dataset[%s]: %d images, batch=%d, augment=%s, shuffle=%s",
                split, len(sub), batch_size, augment, shuffle)
    return ds.prefetch(AUTOTUNE)


def make_all_datasets(cfg: Optional[Config] = None) -> dict:
    """Convenience: ``{'train': ds, 'val': ds, 'test': ds}``."""
    cfg = cfg or load_config()
    df = load_manifest(cfg)
    return {
        "train": make_dataset(cfg, "train", df=df),
        "val": make_dataset(cfg, "val", df=df, shuffle=False, augment=False),
        "test": make_dataset(cfg, "test", df=df, shuffle=False, augment=False),
    }


def dataset_from_arrays(
    images: Sequence[np.ndarray],
    image_size: Tuple[int, int],
    batch_size: int = 64,
) -> tf.data.Dataset:
    """Batched dataset from in-memory images (used by scene scan / batch UI)."""
    arr = np.stack([np.asarray(im, dtype=np.float32) for im in images], axis=0)
    ds = tf.data.Dataset.from_tensor_slices(arr)
    ds = ds.map(lambda x: tf.image.resize(x, image_size, method="bilinear"),
                num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE)


def dataset_from_paths(
    paths: Iterable[str | Path],
    image_size: Tuple[int, int],
    batch_size: int = 64,
) -> tf.data.Dataset:
    """Batched dataset from a list of image file paths (batch analysis page)."""
    str_paths = [str(p) for p in paths]
    ds = tf.data.Dataset.from_tensor_slices(str_paths)
    ds = ds.map(lambda p: _decode(p, image_size), num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE)


def peek(ds: tf.data.Dataset, n: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Grab one batch as numpy - handy for sanity checks and the app."""
    for x, y in ds.take(1):
        return x.numpy()[:n].astype("uint8"), y.numpy()[:n]
    raise ValueError("Dataset produced no batches.")


__all__ = [
    "make_dataset",
    "make_all_datasets",
    "dataset_from_arrays",
    "dataset_from_paths",
    "build_augmenter",
    "peek",
    "AUTOTUNE",
]

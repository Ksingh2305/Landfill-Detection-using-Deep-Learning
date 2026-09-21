"""Grad-CAM explainability.

Answers "which pixels made the model call this tile a landfill?" by
weighting the backbone's final feature maps with the gradient of the
class score. Because :mod:`src.models.build` keeps the backbone and the
head as separate nested models, we can run them in two steps and watch
the feature map in between - no fragile layer-graph surgery, and it works
identically on Keras 2 and Keras 3.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from .build import get_backbone, get_head, get_rescaler


def _as_batch(images) -> tf.Tensor:
    arr = np.asarray(images, dtype="float32")
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.ndim != 4:
        raise ValueError(f"Expected an image or batch of images, got shape {arr.shape}")
    return tf.convert_to_tensor(arr)


def compute_heatmaps(
    model,
    images,
    class_index: Optional[int] = None,
) -> np.ndarray:
    """Grad-CAM heatmaps, normalised to ``[0, 1]``.

    Parameters
    ----------
    model:
        A model built by :func:`src.models.build.build_model`.
    images:
        ``(H, W, 3)`` or ``(N, H, W, 3)`` array with pixels in ``[0, 255]``.
    class_index:
        Which output to explain. ``None`` means "the predicted class"
        (and, for a single-sigmoid binary model, the positive class).

    Returns
    -------
    ``(N, h, w)`` array at the backbone's feature-map resolution.
    """
    x = _as_batch(images)
    rescaler = get_rescaler(model)
    backbone = get_backbone(model)
    head = get_head(model)

    with tf.GradientTape() as tape:
        z = rescaler(x) if rescaler is not None else x
        fmap = backbone(z, training=False)
        tape.watch(fmap)
        preds = head(fmap, training=False)

        if preds.shape[-1] == 1:                      # binary sigmoid
            score = preds[:, 0]
        elif class_index is None:                     # explain the argmax
            idx = tf.argmax(preds, axis=-1, output_type=tf.int32)
            score = tf.gather(preds, idx, batch_dims=1)
        else:
            score = preds[:, int(class_index)]

    grads = tape.gradient(score, fmap)
    if grads is None:  # pragma: no cover - defensive
        return np.zeros((x.shape[0], 1, 1), dtype="float32")

    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)   # (N,1,1,C)
    cam = tf.reduce_sum(weights * fmap, axis=-1)                  # (N,h,w)
    cam = tf.nn.relu(cam)

    cam = cam.numpy().astype("float32")
    flat_max = cam.reshape(cam.shape[0], -1).max(axis=1)
    flat_max[flat_max <= 1e-8] = 1.0
    return cam / flat_max[:, None, None]


def resize_heatmap(heatmap: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Bilinear-resize one ``(h, w)`` heatmap to ``(H, W)``."""
    hm = np.asarray(heatmap, dtype="float32")[None, ..., None]
    out = tf.image.resize(hm, (int(size[0]), int(size[1])), method="bilinear")
    return out.numpy()[0, ..., 0]


def colorize(heatmap: np.ndarray, colormap: str = "jet") -> np.ndarray:
    """Map a ``[0, 1]`` heatmap to an RGB uint8 image using a matplotlib colormap."""
    import matplotlib

    cmap = matplotlib.colormaps.get_cmap(colormap)
    rgba = cmap(np.clip(np.asarray(heatmap, dtype="float32"), 0.0, 1.0))
    return (rgba[..., :3] * 255.0).astype("uint8")


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> np.ndarray:
    """Blend a heatmap over the original image; returns uint8 RGB."""
    img = np.asarray(image)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype("uint8")

    hm = resize_heatmap(heatmap, img.shape[:2])
    coloured = colorize(hm, colormap).astype("float32")
    blended = (1.0 - alpha) * img.astype("float32") + alpha * coloured
    return np.clip(blended, 0, 255).astype("uint8")


def explain(
    model,
    image: np.ndarray,
    class_index: Optional[int] = None,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> Tuple[np.ndarray, np.ndarray]:
    """One-shot helper: returns ``(overlay_uint8, heatmap_float_at_image_size)``."""
    heatmap = compute_heatmaps(model, image, class_index=class_index)[0]
    full = resize_heatmap(heatmap, np.asarray(image).shape[:2])
    return overlay_heatmap(image, heatmap, alpha=alpha, colormap=colormap), full


def batch_explain(
    model,
    images: Sequence[np.ndarray],
    class_index: Optional[int] = None,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> list:
    """Grad-CAM overlays for a list of equally sized images."""
    heatmaps = compute_heatmaps(model, np.stack(images), class_index=class_index)
    return [
        overlay_heatmap(img, hm, alpha=alpha, colormap=colormap)
        for img, hm in zip(images, heatmaps)
    ]


__all__ = [
    "compute_heatmaps",
    "resize_heatmap",
    "colorize",
    "overlay_heatmap",
    "explain",
    "batch_explain",
]

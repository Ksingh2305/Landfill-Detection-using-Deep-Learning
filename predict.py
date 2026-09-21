"""Inference API shared by the CLI, the evaluator and the Streamlit app.

Loading a Keras model takes several seconds, so :func:`load_detector`
caches by (path, mtime): the app gets an instant model after the first
call, but a freshly retrained checkpoint is still picked up automatically.
"""

from __future__ import annotations

import argparse
import functools
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .utils import quiet_tensorflow

quiet_tensorflow()

import numpy as np  # noqa: E402

from .config import Config, load_config  # noqa: E402
from .utils import LOGGER, list_images, load_image_rgb, load_json  # noqa: E402


class ModelNotTrainedError(FileNotFoundError):
    """No checkpoint on disk yet."""


@dataclass
class Prediction:
    """One image's verdict."""

    label: str
    confidence: float                 # probability of the reported label
    positive_score: float             # probability of the landfill/positive class
    is_positive: bool
    probabilities: Dict[str, float] = field(default_factory=dict)
    source: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
@functools.lru_cache(maxsize=4)
def _load_cached(model_path: str, mtime: float):
    import tensorflow as tf  # noqa: WPS433

    LOGGER.info("Loading model: %s", model_path)
    return tf.keras.models.load_model(model_path, compile=False)


def load_detector(cfg: Optional[Config] = None,
                  model_path: Optional[Path] = None) -> Tuple[object, dict]:
    """Return ``(keras_model, metadata_dict)``.

    Raises :class:`ModelNotTrainedError` with actionable instructions when
    the checkpoint is missing.
    """
    cfg = cfg or load_config()
    path = Path(model_path) if model_path else cfg.model_path
    if not path.exists():
        raise ModelNotTrainedError(
            f"No trained model at {path}.\n"
            "Train one first:\n"
            "  python scripts/01_download_data.py\n"
            "  python scripts/02_train.py"
        )
    model = _load_cached(str(path), path.stat().st_mtime)
    meta = load_json(cfg.model_meta_path, default={}) or {}
    meta.setdefault("class_names", cfg.class_names)
    meta.setdefault("threshold", cfg.threshold)
    meta.setdefault("image_size", list(cfg.image_size))
    return model, meta


def model_is_available(cfg: Optional[Config] = None) -> bool:
    cfg = cfg or load_config()
    return cfg.model_path.exists()


# ----------------------------------------------------------------------
# Raw probability inference
# ----------------------------------------------------------------------
def _prepare_batch(images: Sequence[np.ndarray], image_size: Tuple[int, int]) -> np.ndarray:
    import tensorflow as tf  # noqa: WPS433

    arrays = []
    for im in images:
        a = np.asarray(im)
        if a.ndim == 2:
            a = np.stack([a] * 3, axis=-1)
        if a.shape[-1] == 4:
            a = a[..., :3]
        arrays.append(a.astype("float32"))

    same_shape = len({a.shape for a in arrays}) == 1
    if same_shape and arrays[0].shape[:2] == tuple(image_size):
        return np.stack(arrays)

    resized = [tf.image.resize(a[None, ...], image_size, method="bilinear").numpy()[0]
               for a in arrays]
    return np.stack(resized)


def predict_probabilities(
    model,
    images: Sequence[np.ndarray],
    image_size: Tuple[int, int],
    batch_size: int = 64,
) -> np.ndarray:
    """Return ``(N, C)`` probabilities; binary models are expanded to 2 columns."""
    if len(images) == 0:
        return np.zeros((0, 2), dtype="float32")

    batch = _prepare_batch(images, tuple(image_size))
    probs = model.predict(batch, batch_size=batch_size, verbose=0)
    probs = np.asarray(probs, dtype="float32")

    if probs.ndim == 1:
        probs = probs[:, None]
    if probs.shape[-1] == 1:  # sigmoid -> [P(neg), P(pos)]
        p = np.clip(probs[:, 0], 0.0, 1.0)
        probs = np.stack([1.0 - p, p], axis=1)
    return probs


def to_predictions(
    probs: np.ndarray,
    class_names: Sequence[str],
    threshold: float = 0.5,
    sources: Optional[Sequence[str]] = None,
    binary: bool = True,
) -> List[Prediction]:
    """Turn a probability matrix into readable :class:`Prediction` objects."""
    out: List[Prediction] = []
    names = list(class_names)
    sources = list(sources) if sources is not None else [""] * len(probs)

    for row, src in zip(probs, sources):
        pos_score = float(row[1]) if binary and len(row) == 2 else float(row.max())
        if binary and len(row) == 2:
            is_pos = pos_score >= threshold
            idx = 1 if is_pos else 0
        else:
            idx = int(np.argmax(row))
            is_pos = False
        out.append(
            Prediction(
                label=names[idx] if idx < len(names) else f"class_{idx}",
                confidence=float(row[idx]),
                positive_score=pos_score,
                is_positive=bool(is_pos),
                probabilities={
                    (names[i] if i < len(names) else f"class_{i}"): float(v)
                    for i, v in enumerate(row)
                },
                source=str(src),
            )
        )
    return out


# ----------------------------------------------------------------------
# High-level helpers
# ----------------------------------------------------------------------
def predict_images(
    images: Sequence[np.ndarray],
    cfg: Optional[Config] = None,
    model=None,
    meta: Optional[dict] = None,
    threshold: Optional[float] = None,
    sources: Optional[Sequence[str]] = None,
) -> List[Prediction]:
    cfg = cfg or load_config()
    if model is None:
        model, meta = load_detector(cfg)
    meta = meta or {}
    image_size = tuple(meta.get("image_size", cfg.image_size))
    class_names = list(meta.get("class_names", cfg.class_names))
    thr = float(threshold if threshold is not None else meta.get("threshold", cfg.threshold))

    probs = predict_probabilities(model, images, image_size)
    return to_predictions(probs, class_names, thr, sources=sources, binary=cfg.is_binary)


def predict_files(
    paths: Sequence[str | Path],
    cfg: Optional[Config] = None,
    threshold: Optional[float] = None,
    batch_size: int = 64,
) -> List[Prediction]:
    """Classify image files on disk, streaming in batches to bound memory."""
    cfg = cfg or load_config()
    model, meta = load_detector(cfg)
    image_size = tuple(meta.get("image_size", cfg.image_size))
    class_names = list(meta.get("class_names", cfg.class_names))
    thr = float(threshold if threshold is not None else meta.get("threshold", cfg.threshold))

    results: List[Prediction] = []
    paths = [Path(p) for p in paths]
    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]
        images = [load_image_rgb(p, image_size) for p in chunk]
        probs = predict_probabilities(model, images, image_size, batch_size=batch_size)
        results.extend(
            to_predictions(probs, class_names, thr,
                           sources=[str(p) for p in chunk], binary=cfg.is_binary)
        )
    return results


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Classify satellite image tiles.")
    p.add_argument("inputs", nargs="+", help="image files and/or folders")
    p.add_argument("--config", default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--csv", default=None, help="write results to this CSV")
    args = p.parse_args(argv)

    cfg = load_config(args.config)

    paths: List[Path] = []
    for item in args.inputs:
        path = Path(item)
        paths.extend(list_images(path) if path.is_dir() else [path])
    if not paths:
        LOGGER.error("No images found in: %s", args.inputs)
        return 1

    preds = predict_files(paths, cfg, threshold=args.threshold)
    for pred in preds:
        flag = "LANDFILL" if pred.is_positive else "clean    "
        LOGGER.info("%s  p=%.3f  %s", flag, pred.positive_score, Path(pred.source).name)

    n_pos = sum(1 for p in preds if p.is_positive)
    LOGGER.info("-" * 60)
    LOGGER.info("%d/%d tiles flagged as landfill candidates (%.1f%%)",
                n_pos, len(preds), 100.0 * n_pos / max(len(preds), 1))

    if args.csv:
        import pandas as pd

        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([pr.as_dict() for pr in preds]).to_csv(out, index=False)
        LOGGER.info("Results written to %s", out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

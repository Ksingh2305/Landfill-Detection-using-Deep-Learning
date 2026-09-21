"""Shared helpers: logging, seeding, JSON I/O, image loading, timing."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def get_logger(name: str = "landfill", level: int = logging.INFO) -> logging.Logger:
    """A console logger that does not duplicate handlers on re-import."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


LOGGER = get_logger()


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy and (if importable) tensorflow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf  # noqa: WPS433 (deliberate late import)

        tf.random.set_seed(seed)
        try:
            tf.keras.utils.set_random_seed(seed)
        except Exception:  # pragma: no cover - older/newer keras
            pass
    except ImportError:
        pass


def quiet_tensorflow() -> None:
    """Silence TensorFlow's very chatty C++ logging. Call BEFORE importing tf."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")


# ----------------------------------------------------------------------
# JSON helpers
# ----------------------------------------------------------------------
class _NpEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:  # noqa: D102
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(obj: Any, path: str | os.PathLike, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, cls=_NpEncoder)
    return path


def load_json(path: str | os.PathLike, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ----------------------------------------------------------------------
# Images
# ----------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(directory: str | os.PathLike, recursive: bool = True) -> list[Path]:
    """All image files under ``directory``, sorted for reproducibility."""
    directory = Path(directory)
    if not directory.exists():
        return []
    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in directory.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_image_rgb(path_or_file, target_size: Optional[Sequence[int]] = None) -> np.ndarray:
    """Load any image as an ``uint8`` RGB array, optionally resized to (H, W)."""
    from PIL import Image

    img = Image.open(path_or_file)
    if img.mode != "RGB":
        img = img.convert("RGB")
    if target_size is not None:
        img = img.resize((int(target_size[1]), int(target_size[0])), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def to_uint8(array: np.ndarray) -> np.ndarray:
    """Best-effort conversion of a float/16-bit array to displayable uint8."""
    arr = np.asarray(array)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi - lo < 1e-9:
        return np.zeros_like(arr, dtype=np.uint8)
    return np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------
# Misc
# ----------------------------------------------------------------------
@contextmanager
def timer(label: str, logger: Optional[logging.Logger] = None):
    """Context manager that logs how long a block took."""
    log = logger or LOGGER
    start = time.perf_counter()
    log.info("%s ...", label)
    try:
        yield
    finally:
        log.info("%s finished in %.2fs", label, time.perf_counter() - start)


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:3.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def chunked(seq: Sequence, size: int) -> Iterable[Sequence]:
    """Yield ``seq`` in consecutive chunks of at most ``size`` items."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def summarise_counts(counts: Dict[str, int]) -> str:
    total = sum(counts.values()) or 1
    parts = [f"{k}={v} ({100.0 * v / total:.1f}%)" for k, v in sorted(counts.items())]
    return ", ".join(parts)


__all__ = [
    "get_logger",
    "LOGGER",
    "set_seed",
    "quiet_tensorflow",
    "save_json",
    "load_json",
    "list_images",
    "load_image_rgb",
    "to_uint8",
    "timer",
    "human_bytes",
    "chunked",
    "summarise_counts",
    "IMAGE_EXTENSIONS",
]

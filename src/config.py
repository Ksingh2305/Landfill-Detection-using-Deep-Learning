"""Configuration loading and path resolution.

Everything in the project routes through :func:`load_config` so there is a
single source of truth (``config.yaml`` at the repository root).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Repository root = parent of the ``src`` package directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class ConfigError(RuntimeError):
    """Raised when config.yaml is missing or internally inconsistent."""


def _resolve(path_like: str) -> Path:
    """Turn a config path (relative to the project root) into an absolute Path."""
    p = Path(path_like)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


@dataclass
class Config:
    """Thin typed wrapper around the parsed YAML."""

    raw: Dict[str, Any]
    path: Path

    # -- dictionary-ish access ------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    # -- convenience accessors -------------------------------------------
    @property
    def seed(self) -> int:
        return int(self.raw["project"]["seed"])

    @property
    def image_size(self) -> tuple:
        h, w = self.raw["data"]["image_size"]
        return (int(h), int(w))

    @property
    def channels(self) -> int:
        return int(self.raw["data"].get("channels", 3))

    @property
    def input_shape(self) -> tuple:
        h, w = self.image_size
        return (h, w, self.channels)

    @property
    def is_binary(self) -> bool:
        return str(self.raw["task"]["mode"]).lower() == "binary"

    @property
    def class_names(self) -> List[str]:
        """Ordered list of output class names (index == model output index)."""
        task = self.raw["task"]
        if self.is_binary:
            # index 0 = negative, index 1 = positive
            return [task["negative_label"], task["positive_label"]]
        return sorted(set(task["positive_classes"]) | set(task["negative_classes"]))

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def threshold(self) -> float:
        return float(self.raw["task"].get("threshold", 0.5))

    # -- resolved paths ---------------------------------------------------
    def path_of(self, key: str) -> Path:
        """Absolute path for any entry under the ``paths:`` block."""
        try:
            return _resolve(self.raw["paths"][key])
        except KeyError as exc:  # pragma: no cover - defensive
            raise ConfigError(f"Unknown path key '{key}' in config.yaml") from exc

    @property
    def data_root(self) -> Path:
        return self.path_of("data_root")

    @property
    def raw_dir(self) -> Path:
        return self.path_of("raw_dir")

    @property
    def interim_dir(self) -> Path:
        return self.path_of("interim_dir")

    @property
    def processed_dir(self) -> Path:
        return self.path_of("processed_dir")

    @property
    def custom_dir(self) -> Path:
        return self.path_of("custom_dir")

    @property
    def models_dir(self) -> Path:
        return self.path_of("models_dir")

    @property
    def reports_dir(self) -> Path:
        return self.path_of("reports_dir")

    @property
    def logs_dir(self) -> Path:
        return self.path_of("logs_dir")

    @property
    def manifest_path(self) -> Path:
        return self.interim_dir / "manifest.csv"

    @property
    def model_path(self) -> Path:
        return self.models_dir / "landfill_detector.keras"

    @property
    def model_meta_path(self) -> Path:
        return self.models_dir / "model_meta.json"

    def ensure_dirs(self) -> None:
        """Create every directory referenced by the ``paths:`` block."""
        for key in self.raw["paths"]:
            self.path_of(key).mkdir(parents=True, exist_ok=True)


def _validate(cfg: Dict[str, Any]) -> None:
    splits = cfg["data"]["splits"]
    total = sum(float(v) for v in splits.values())
    if abs(total - 1.0) > 1e-6:
        raise ConfigError(
            f"data.splits must sum to 1.0 (got {total:.4f}: {splits})"
        )

    mode = str(cfg["task"]["mode"]).lower()
    if mode not in {"binary", "multiclass"}:
        raise ConfigError(f"task.mode must be 'binary' or 'multiclass', got '{mode}'")

    overlap = float(cfg["scan"]["overlap"])
    if not 0.0 <= overlap < 1.0:
        raise ConfigError(f"scan.overlap must be in [0, 1), got {overlap}")

    backbone = str(cfg["model"]["backbone"]).lower()
    if backbone not in {"mobilenetv2", "efficientnetb0", "resnet50v2"}:
        raise ConfigError(
            "model.backbone must be one of mobilenetv2 | efficientnetb0 | resnet50v2, "
            f"got '{backbone}'"
        )

    overlap_pos = set(cfg["task"]["positive_classes"]) & set(cfg["task"]["negative_classes"])
    if overlap_pos:
        raise ConfigError(
            f"These classes appear in both positive_classes and negative_classes: {sorted(overlap_pos)}"
        )


_CACHE: Dict[str, Config] = {}


def load_config(path: str | os.PathLike | None = None, use_cache: bool = True) -> Config:
    """Read and validate ``config.yaml``.

    Parameters
    ----------
    path:
        Optional explicit path. Defaults to ``<project root>/config.yaml``.
    use_cache:
        Re-use a previously parsed config for the same path (default ``True``).
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    cfg_path = cfg_path.resolve()
    key = str(cfg_path)

    if use_cache and key in _CACHE:
        return _CACHE[key]

    if not cfg_path.exists():
        raise ConfigError(
            f"config.yaml not found at {cfg_path}. Run the project from its root folder."
        )

    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ConfigError(f"{cfg_path} did not parse into a mapping.")

    _validate(raw)
    cfg = Config(raw=raw, path=cfg_path)
    _CACHE[key] = cfg
    return cfg


__all__ = ["Config", "ConfigError", "load_config", "PROJECT_ROOT", "DEFAULT_CONFIG_PATH"]

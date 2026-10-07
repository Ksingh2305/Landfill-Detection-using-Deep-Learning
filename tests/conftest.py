"""Shared pytest fixtures.

Every test runs against a *temporary* project: the real ``config.yaml`` is
loaded for its structure, then all paths are redirected into ``tmp_path``
and the dataset is shrunk so the suite finishes in seconds and never
touches the user's real data or models.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from src.config import Config, load_config  # noqa: E402


@pytest.fixture(scope="session")
def base_config() -> Config:
    return load_config()


@pytest.fixture
def tmp_config(tmp_path: Path, base_config: Config) -> Config:
    """A Config whose every path lives under ``tmp_path``."""
    raw = copy.deepcopy(base_config.raw)

    raw["paths"] = {
        "data_root": str(tmp_path / "data"),
        "raw_dir": str(tmp_path / "data" / "raw"),
        "interim_dir": str(tmp_path / "data" / "interim"),
        "processed_dir": str(tmp_path / "data" / "processed"),
        "custom_dir": str(tmp_path / "data" / "custom"),
        "models_dir": str(tmp_path / "models"),
        "reports_dir": str(tmp_path / "reports"),
        "logs_dir": str(tmp_path / "reports" / "logs"),
    }
    raw["data"]["source"] = "synthetic"
    raw["data"]["image_size"] = [64, 64]
    raw["data"]["batch_size"] = 8
    raw["data"]["cache"] = False
    raw["data"]["max_per_class"] = 0
    raw["model"]["weights"] = None          # never download ImageNet weights in tests
    raw["model"]["dense_units"] = 16
    raw["train"]["epochs_head"] = 1
    raw["train"]["epochs_finetune"] = 0

    cfg = Config(raw=raw, path=tmp_path / "config.yaml")
    cfg.ensure_dirs()
    return cfg


@pytest.fixture
def tiny_dataset(tmp_config: Config) -> Config:
    """Generate a handful of synthetic tiles per class and build the manifest."""
    from src.data.preprocess import build_manifest
    from src.data.synthetic import generate_synthetic_dataset

    root = generate_synthetic_dataset(tmp_config, per_class=6)
    build_manifest(tmp_config, image_root=root)
    return tmp_config

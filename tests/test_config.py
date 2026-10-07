"""The config loader is the contract every other module depends on."""

from __future__ import annotations

import copy

import pytest

from src.config import Config, ConfigError, _validate, load_config


def test_loads_and_validates():
    cfg = load_config()
    assert cfg["project"]["name"]
    assert cfg.num_classes >= 2
    assert cfg.input_shape[2] == 3


def test_class_names_order_is_negative_then_positive():
    cfg = load_config()
    if cfg.is_binary:
        assert cfg.class_names == [
            cfg["task"]["negative_label"],
            cfg["task"]["positive_label"],
        ]
        assert cfg.num_classes == 2


def test_splits_must_sum_to_one():
    raw = copy.deepcopy(load_config().raw)
    raw["data"]["splits"] = {"train": 0.9, "val": 0.2, "test": 0.2}
    with pytest.raises(ConfigError, match="sum to 1.0"):
        _validate(raw)


def test_rejects_unknown_backbone():
    raw = copy.deepcopy(load_config().raw)
    raw["model"]["backbone"] = "not_a_real_net"
    with pytest.raises(ConfigError, match="backbone"):
        _validate(raw)


def test_rejects_class_in_both_lists():
    raw = copy.deepcopy(load_config().raw)
    raw["task"]["negative_classes"].append(raw["task"]["positive_classes"][0])
    with pytest.raises(ConfigError, match="both"):
        _validate(raw)


def test_rejects_bad_overlap():
    raw = copy.deepcopy(load_config().raw)
    raw["data"]["splits"] = {"train": 0.7, "val": 0.15, "test": 0.15}
    raw["scan"]["overlap"] = 1.0
    with pytest.raises(ConfigError, match="overlap"):
        _validate(raw)


def test_tmp_config_paths_are_isolated(tmp_config: Config, tmp_path):
    assert tmp_path in tmp_config.models_dir.parents or tmp_config.models_dir.parent == tmp_path
    assert tmp_config.model_path.name.endswith(".keras")
    for key in tmp_config.raw["paths"]:
        assert tmp_config.path_of(key).exists()

"""Synthetic generator, manifest builder and tf.data pipeline."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from src.config import Config
from src.data.preprocess import (
    PreprocessError,
    build_label_map,
    build_manifest,
    compute_class_weights,
    dataset_stats,
    load_manifest,
    resolve_path,
    stratified_split,
)
from src.data.synthetic import ALL_CLASSES, make_scene, make_tile


# ----------------------------------------------------------------------
# Synthetic imagery
# ----------------------------------------------------------------------
@pytest.mark.parametrize("class_name", ALL_CLASSES)
def test_make_tile_shape_and_range(class_name):
    rng = np.random.default_rng(0)
    tile = make_tile(class_name, rng)
    assert tile.shape == (64, 64, 3)
    assert tile.dtype == np.uint8
    assert tile.std() > 1.0, "tiles should not be flat"


def test_classes_are_visually_distinguishable():
    """Mean colours of Forest and Industrial must not collapse together."""
    rng = np.random.default_rng(1)
    forest = np.mean([make_tile("Forest", rng).mean(axis=(0, 1)) for _ in range(8)], axis=0)
    industrial = np.mean([make_tile("Industrial", rng).mean(axis=(0, 1)) for _ in range(8)], axis=0)
    assert np.linalg.norm(forest - industrial) > 25


def test_make_scene_shapes_and_mask():
    scene, mask = make_scene(320, 256, positive_classes=["Industrial"], seed=2)
    assert scene.shape == (256, 320, 3)
    assert mask.shape == (256, 320)
    assert mask.dtype == bool
    assert 0.0 < mask.mean() < 1.0


# ----------------------------------------------------------------------
# Splitting
# ----------------------------------------------------------------------
def test_stratified_split_respects_fractions():
    rng = np.random.default_rng(0)
    labels = stratified_split(1000, {"train": 0.7, "val": 0.15, "test": 0.15}, rng)
    counts = Counter(labels)
    assert len(labels) == 1000
    assert abs(counts["train"] - 700) <= 2
    assert abs(counts["val"] - 150) <= 2
    assert abs(counts["test"] - 150) <= 2


def test_stratified_split_tiny_class_still_gets_val_and_test():
    rng = np.random.default_rng(0)
    counts = Counter(stratified_split(3, {"train": 0.7, "val": 0.15, "test": 0.15}, rng))
    assert counts["val"] >= 1 and counts["test"] >= 1


# ----------------------------------------------------------------------
# Label mapping
# ----------------------------------------------------------------------
def test_label_map_assigns_positive_and_negative(tmp_config: Config):
    mapping = build_label_map(tmp_config, ALL_CLASSES)
    pos_label = tmp_config["task"]["positive_label"]
    neg_label = tmp_config["task"]["negative_label"]
    for cls in tmp_config["task"]["positive_classes"]:
        assert mapping[cls] == pos_label
    for cls in tmp_config["task"]["negative_classes"]:
        assert mapping[cls] == neg_label


def test_label_map_raises_when_nothing_matches(tmp_config: Config):
    with pytest.raises(PreprocessError, match="positive_classes"):
        build_label_map(tmp_config, ["Mars", "Venus"])


# ----------------------------------------------------------------------
# Manifest
# ----------------------------------------------------------------------
def test_manifest_is_complete_and_consistent(tiny_dataset: Config):
    df = load_manifest(tiny_dataset)
    assert len(df) == 6 * len(ALL_CLASSES)
    assert set(df["split"]) <= {"train", "val", "test"}
    assert set(df["label"]) == set(tiny_dataset.class_names)

    for label, idx in df[["label", "label_idx"]].drop_duplicates().to_numpy():
        assert tiny_dataset.class_names[int(idx)] == label

    for p in df["path"].head(20):
        assert resolve_path(p).exists()


def test_no_image_appears_in_two_splits(tiny_dataset: Config):
    df = load_manifest(tiny_dataset)
    assert df["path"].nunique() == len(df)


def test_dataset_stats_and_class_weights(tiny_dataset: Config):
    stats = dataset_stats(tiny_dataset)
    assert stats["total_images"] == 6 * len(ALL_CLASSES)
    assert set(stats["label_counts"]) == set(tiny_dataset.class_names)

    weights = compute_class_weights(tiny_dataset)
    assert set(weights) == set(range(tiny_dataset.num_classes))
    assert all(w > 0 for w in weights.values())
    # the rarer class must be weighted more heavily
    counts = stats["by_split_label"]["train"]
    if len(counts) == 2:
        rare = min(counts, key=counts.get)
        common = max(counts, key=counts.get)
        if counts[rare] != counts[common]:
            rare_i = tiny_dataset.class_names.index(rare)
            common_i = tiny_dataset.class_names.index(common)
            assert weights[rare_i] > weights[common_i]


def test_max_per_class_caps_the_manifest(tmp_config: Config):
    from src.data.synthetic import generate_synthetic_dataset

    root = generate_synthetic_dataset(tmp_config, per_class=8)
    tmp_config.raw["data"]["max_per_class"] = 4
    df = build_manifest(tmp_config, image_root=root)
    assert len(df) == 4 * len(ALL_CLASSES)


def test_load_manifest_errors_helpfully(tmp_config: Config):
    with pytest.raises(PreprocessError, match="01_download_data"):
        load_manifest(tmp_config)


# ----------------------------------------------------------------------
# tf.data pipeline
# ----------------------------------------------------------------------
def test_dataset_batches_have_the_right_shape(tiny_dataset: Config):
    from src.data.dataset import make_dataset

    ds = make_dataset(tiny_dataset, "train")
    x, y = next(iter(ds))
    h, w = tiny_dataset.image_size
    assert x.shape[1:] == (h, w, 3)
    assert x.numpy().min() >= -1.0          # augmentation can dip slightly below 0
    assert x.numpy().max() <= 300.0
    assert len(y) == len(x)


def test_validation_split_is_not_augmented(tiny_dataset: Config):
    from src.data.dataset import make_dataset

    ds = make_dataset(tiny_dataset, "val", shuffle=False, augment=False)
    a = np.concatenate([b.numpy() for b, _ in ds.take(2)])
    b = np.concatenate([b.numpy() for b, _ in ds.take(2)])
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_dataset_from_arrays_resizes(tiny_dataset: Config):
    from src.data.dataset import dataset_from_arrays

    images = [np.random.randint(0, 255, (37, 41, 3), dtype=np.uint8) for _ in range(5)]
    ds = dataset_from_arrays(images, tiny_dataset.image_size, batch_size=2)
    batch = next(iter(ds))
    assert batch.shape[1:3] == tuple(tiny_dataset.image_size)

"""Model construction, fine-tuning control, Grad-CAM and the inference API."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import Config


@pytest.fixture(scope="module")
def keras():
    tf = pytest.importorskip("tensorflow")
    return tf.keras


def test_model_shapes_and_probability_range(tmp_config: Config, keras):
    from src.models.build import build_model

    model = build_model(tmp_config)
    h, w = tmp_config.image_size
    assert tuple(model.input_shape[1:]) == (h, w, 3)
    assert model.output_shape[-1] == (1 if tmp_config.is_binary else tmp_config.num_classes)

    x = np.random.randint(0, 256, (4, h, w, 3)).astype("float32")
    p = model.predict(x, verbose=0)
    assert p.shape[0] == 4
    assert np.all((p >= 0) & (p <= 1)), "outputs must be probabilities"


def test_backbone_and_head_are_addressable(tmp_config: Config, keras):
    from src.models.build import (
        build_model,
        get_backbone,
        get_head,
        last_conv_layer_name,
    )

    model = build_model(tmp_config)
    backbone = get_backbone(model)
    head = get_head(model)
    assert backbone is not head
    assert len(backbone.output.shape) == 4, "backbone must emit a feature map"
    assert last_conv_layer_name(model)


def test_freeze_then_unfreeze_changes_trainable_count(tmp_config: Config, keras):
    from src.models.build import build_model, get_backbone, unfreeze_backbone

    model = build_model(tmp_config, freeze_backbone=True)
    assert get_backbone(model).trainable is False
    frozen_params = len(model.trainable_weights)

    unfreeze_backbone(model, last_n=10)
    assert get_backbone(model).trainable is True
    assert len(model.trainable_weights) > frozen_params


def test_batchnorm_stays_frozen_during_finetuning(tmp_config: Config, keras):
    from src.models.build import build_model, get_backbone, unfreeze_backbone

    model = build_model(tmp_config)
    unfreeze_backbone(model, last_n=30, freeze_batchnorm=True)
    bns = [l for l in get_backbone(model).layers
           if isinstance(l, keras.layers.BatchNormalization)]
    assert bns, "the backbone should contain BatchNorm layers"
    assert all(not l.trainable for l in bns)


def test_model_round_trips_through_disk(tmp_config: Config, keras):
    from src.models.build import build_model, get_backbone

    model = build_model(tmp_config)
    h, w = tmp_config.image_size
    x = np.random.randint(0, 256, (2, h, w, 3)).astype("float32")
    before = model.predict(x, verbose=0)

    path = tmp_config.models_dir / "roundtrip.keras"
    model.save(path)
    reloaded = keras.models.load_model(path, compile=False)

    np.testing.assert_allclose(before, reloaded.predict(x, verbose=0), atol=1e-4)
    assert get_backbone(reloaded) is not None, "nested models must survive serialisation"


def test_gradcam_produces_a_valid_heatmap(tmp_config: Config, keras):
    from src.models.build import build_model
    from src.models.gradcam import compute_heatmaps, overlay_heatmap, resize_heatmap

    model = build_model(tmp_config)
    h, w = tmp_config.image_size
    image = np.random.randint(0, 256, (h, w, 3)).astype("float32")

    heat = compute_heatmaps(model, image)
    assert heat.ndim == 3 and heat.shape[0] == 1
    assert 0.0 <= float(heat.min()) and float(heat.max()) <= 1.0 + 1e-6

    full = resize_heatmap(heat[0], (h, w))
    assert full.shape == (h, w)

    overlay = overlay_heatmap(image.astype("uint8"), heat[0])
    assert overlay.shape == (h, w, 3)
    assert overlay.dtype == np.uint8


def test_predict_probabilities_expands_binary_output(tmp_config: Config, keras):
    from src.models.build import build_model
    from src.predict import predict_probabilities, to_predictions

    model = build_model(tmp_config)
    h, w = tmp_config.image_size
    images = [np.random.randint(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(3)]

    probs = predict_probabilities(model, images, (h, w))
    assert probs.shape == (3, 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    preds = to_predictions(probs, tmp_config.class_names, threshold=0.5)
    assert len(preds) == 3
    for p in preds:
        assert p.label in tmp_config.class_names
        assert 0.0 <= p.positive_score <= 1.0
        assert p.is_positive == (p.positive_score >= 0.5)


def test_predict_handles_odd_input_sizes_and_channels(tmp_config: Config, keras):
    from src.models.build import build_model
    from src.predict import predict_probabilities

    model = build_model(tmp_config)
    h, w = tmp_config.image_size
    images = [
        np.random.randint(0, 256, (23, 91, 3), dtype=np.uint8),   # wrong size
        np.random.randint(0, 256, (h, w, 4), dtype=np.uint8),     # RGBA
        np.random.randint(0, 256, (h, w), dtype=np.uint8),        # greyscale
    ]
    probs = predict_probabilities(model, images, (h, w))
    assert probs.shape == (3, 2)


def test_training_runs_one_epoch_end_to_end(tiny_dataset: Config, keras):
    """A single tiny epoch must complete and leave a loadable checkpoint."""
    from src.train import train

    tiny_dataset.raw["train"]["epochs_head"] = 1
    tiny_dataset.raw["train"]["epochs_finetune"] = 0
    tiny_dataset.raw["train"]["early_stopping_patience"] = 1

    meta = train(tiny_dataset, epochs_head=1, epochs_finetune=0, skip_finetune=True)

    assert tiny_dataset.model_path.exists()
    assert meta["class_names"] == tiny_dataset.class_names
    assert (tiny_dataset.reports_dir / "history.csv").exists()

    from src.predict import load_detector

    model, loaded_meta = load_detector(tiny_dataset)
    assert loaded_meta["class_names"] == tiny_dataset.class_names
    h, w = tiny_dataset.image_size
    out = model.predict(np.zeros((1, h, w, 3), dtype="float32"), verbose=0)
    assert out.shape[0] == 1

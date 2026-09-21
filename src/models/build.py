"""Model architecture: ImageNet transfer learning for satellite tiles.

The exported model is **self-contained**: it accepts raw ``[0, 255]``
float images of shape ``(H, W, 3)`` and applies the backbone's own
normalisation internally via a serialisable ``Rescaling`` layer. That
means the Streamlit app can feed it a decoded PNG with no hidden
preprocessing contract.

Structure::

    image (0..255)
      -> Rescaling            (backbone-specific)
      -> backbone             (nested Keras model, frozen then fine-tuned)
      -> head                 (GAP -> Dropout -> Dense -> Dropout -> Dense)
      -> probability

Keeping ``backbone`` and ``head`` as *named nested models* is deliberate:
Grad-CAM (:mod:`src.models.gradcam`) can then run the two halves
separately and watch the feature map in between, which is far more robust
across Keras versions than reaching into a flattened layer graph.
"""

from __future__ import annotations

from typing import Optional, Tuple

import tensorflow as tf
from tensorflow import keras

from ..config import Config, load_config
from ..utils import LOGGER

layers = keras.layers

BACKBONE_NAME = "backbone"
HEAD_NAME = "head"
RESCALE_NAME = "rescale"

# scale/offset that reproduce each family's official preprocess_input
_RESCALING = {
    "mobilenetv2": (1.0 / 127.5, -1.0),
    "resnet50v2": (1.0 / 127.5, -1.0),
    "efficientnetb0": (1.0, 0.0),  # EfficientNet normalises internally
}


class ModelError(RuntimeError):
    """Unsupported or inconsistent model configuration."""


# ----------------------------------------------------------------------
# Backbone
# ----------------------------------------------------------------------
_BACKBONE_FACTORIES = {
    "mobilenetv2": lambda **kw: keras.applications.MobileNetV2(**kw),
    "efficientnetb0": lambda **kw: keras.applications.EfficientNetB0(**kw),
    "resnet50v2": lambda **kw: keras.applications.ResNet50V2(**kw),
}


def _make_backbone(name: str, weights: Optional[str], input_shape: Tuple[int, int, int]):
    name = name.lower()
    factory = _BACKBONE_FACTORIES.get(name)
    if factory is None:  # pragma: no cover - guarded by config validation
        raise ModelError(f"Unknown backbone '{name}'")

    common = dict(include_top=False, weights=weights, input_shape=input_shape)
    try:
        base = factory(**common)
    except Exception as exc:  # noqa: BLE001 - almost always a blocked weight download
        if not weights:
            raise
        LOGGER.warning(
            "Could not fetch '%s' weights for %s (%s). Falling back to random "
            "initialisation - accuracy will be much lower. Connect to the internet "
            "once so Keras can cache the weights, or set model.weights: null in "
            "config.yaml to make this deliberate.",
            weights, name, exc,
        )
        common["weights"] = None
        base = factory(**common)

    try:
        base._name = BACKBONE_NAME  # noqa: SLF001 - stable across Keras 2/3
    except Exception:  # pragma: no cover
        LOGGER.debug("Could not rename backbone; falling back to type lookup.")
    return base


def _make_head(feature_shape: Tuple[int, ...], cfg: Config) -> keras.Model:
    m = cfg["model"]
    reg = keras.regularizers.l2(float(m.get("l2", 0.0))) if float(m.get("l2", 0.0)) else None

    inp = keras.Input(shape=feature_shape, name="features")
    x = layers.GlobalAveragePooling2D(name="gap")(inp)
    x = layers.Dropout(float(m.get("dropout", 0.3)), name="drop1")(x)
    x = layers.Dense(int(m.get("dense_units", 256)), activation="relu",
                     kernel_regularizer=reg, name="fc1")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.Dropout(float(m.get("dropout", 0.3)) / 2.0, name="drop2")(x)

    if cfg.is_binary:
        out = layers.Dense(1, activation="sigmoid", dtype="float32", name="probability")(x)
    else:
        out = layers.Dense(cfg.num_classes, activation="softmax",
                           dtype="float32", name="probability")(x)

    return keras.Model(inp, out, name=HEAD_NAME)


# ----------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------
def build_model(cfg: Optional[Config] = None, freeze_backbone: bool = True) -> keras.Model:
    """Create (but do not compile) the full detector."""
    cfg = cfg or load_config()
    backbone_name = str(cfg["model"]["backbone"]).lower()
    weights = cfg["model"].get("weights")
    weights = None if weights in (None, "none", "null", False) else str(weights)
    input_shape = cfg.input_shape

    base = _make_backbone(backbone_name, weights, input_shape)
    base.trainable = not freeze_backbone

    scale, offset = _RESCALING[backbone_name]

    inputs = keras.Input(shape=input_shape, name="image")
    x = layers.Rescaling(scale, offset=offset, name=RESCALE_NAME)(inputs)
    features = base(x)
    head = _make_head(tuple(features.shape[1:]), cfg)
    outputs = head(features)

    model = keras.Model(inputs, outputs, name="landfill_detector")
    LOGGER.info(
        "Built %s | input=%s | backbone=%s (%s weights, %s params) | outputs=%d",
        model.name, input_shape, backbone_name, weights or "random",
        f"{base.count_params():,}", cfg.num_classes if not cfg.is_binary else 1,
    )
    return model


# ----------------------------------------------------------------------
# Compilation
# ----------------------------------------------------------------------
def build_metrics(cfg: Config) -> list:
    if cfg.is_binary:
        return [
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
        ]
    return [
        keras.metrics.CategoricalAccuracy(name="accuracy"),
        keras.metrics.AUC(name="auc", multi_label=True),
    ]


def build_loss(cfg: Config):
    smoothing = float(cfg["model"].get("label_smoothing", 0.0))
    if cfg.is_binary:
        return keras.losses.BinaryCrossentropy(label_smoothing=smoothing)
    return keras.losses.CategoricalCrossentropy(label_smoothing=smoothing)


def compile_model(model: keras.Model, cfg: Config, learning_rate: float) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=build_loss(cfg),
        metrics=build_metrics(cfg),
    )
    return model


# ----------------------------------------------------------------------
# Sub-model access + fine-tuning control
# ----------------------------------------------------------------------
def get_backbone(model: keras.Model) -> keras.Model:
    """Return the nested backbone regardless of how Keras named it."""
    try:
        return model.get_layer(BACKBONE_NAME)
    except (ValueError, KeyError):
        pass
    for layer in model.layers:
        if isinstance(layer, keras.Model) and layer.name != HEAD_NAME:
            return layer
    raise ModelError("Could not locate the backbone sub-model.")


def get_head(model: keras.Model) -> keras.Model:
    try:
        return model.get_layer(HEAD_NAME)
    except (ValueError, KeyError):
        pass
    for layer in reversed(model.layers):
        if isinstance(layer, keras.Model):
            return layer
    raise ModelError("Could not locate the head sub-model.")


def get_rescaler(model: keras.Model):
    """The input normalisation layer (may be absent in hand-built models)."""
    try:
        return model.get_layer(RESCALE_NAME)
    except (ValueError, KeyError):
        return None


def unfreeze_backbone(model: keras.Model, last_n: int = 40,
                      freeze_batchnorm: bool = True) -> int:
    """Unfreeze the final ``last_n`` backbone layers for phase-2 fine-tuning.

    BatchNormalization layers stay frozen by default - updating their
    running statistics on a small, differently-distributed satellite
    dataset is a classic source of fine-tuning collapse.
    """
    backbone = get_backbone(model)
    backbone.trainable = True

    total = len(backbone.layers)
    cutoff = max(0, total - int(last_n))
    unfrozen = 0

    for i, layer in enumerate(backbone.layers):
        if i < cutoff:
            layer.trainable = False
            continue
        if freeze_batchnorm and isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
            continue
        layer.trainable = True
        unfrozen += 1

    trainable_params = int(sum(
        tf.size(w).numpy() for w in model.trainable_weights
    )) if model.trainable_weights else 0
    LOGGER.info("Fine-tuning: %d/%d backbone layers unfrozen (%s trainable params)",
                unfrozen, total, f"{trainable_params:,}")
    return unfrozen


def freeze_backbone(model: keras.Model) -> None:
    get_backbone(model).trainable = False


def last_conv_layer_name(model: keras.Model) -> str:
    """Name of the deepest 4-D activation inside the backbone (Grad-CAM target)."""
    backbone = get_backbone(model)
    for layer in reversed(backbone.layers):
        try:
            shape = layer.output.shape
        except (AttributeError, ValueError):
            continue
        if len(shape) == 4:
            return layer.name
    raise ModelError("No 4-D convolutional output found in the backbone.")


__all__ = [
    "build_model",
    "compile_model",
    "build_loss",
    "build_metrics",
    "get_backbone",
    "get_head",
    "get_rescaler",
    "unfreeze_backbone",
    "freeze_backbone",
    "last_conv_layer_name",
    "ModelError",
    "BACKBONE_NAME",
    "HEAD_NAME",
    "RESCALE_NAME",
]

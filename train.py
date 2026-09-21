"""Two-phase transfer-learning trainer.

Phase 1 - *head warm-up*: the ImageNet backbone is frozen and only the new
classification head trains, at a comparatively high learning rate. This
stops large random-head gradients from destroying pretrained filters.

Phase 2 - *fine-tuning*: the deepest ``train.unfreeze_last_n`` backbone
layers are unfrozen (BatchNorm kept frozen) and training continues at a
much lower learning rate so the high-level features adapt to Sentinel-2
texture and colour statistics without catastrophic forgetting.

Artifacts written
-----------------
``models/landfill_detector.keras``   best checkpoint (by val AUC)
``models/model_meta.json``           class names, threshold, config snapshot
``reports/history.csv``              per-epoch metrics for both phases
``reports/figures/training_curves.png``
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
from typing import Dict, List, Optional

from .utils import quiet_tensorflow

quiet_tensorflow()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402

from .config import Config, load_config  # noqa: E402
from .data.dataset import make_dataset  # noqa: E402
from .data.preprocess import compute_class_weights, dataset_stats, load_manifest  # noqa: E402
from .models.build import (  # noqa: E402
    build_model,
    compile_model,
    freeze_backbone,
    unfreeze_backbone,
)
from .utils import LOGGER, save_json, set_seed, timer  # noqa: E402

MONITOR = "val_auc"
MONITOR_MODE = "max"


# ----------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------
def build_callbacks(cfg: Config, phase: str) -> List[keras.callbacks.Callback]:
    t = cfg["train"]
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)

    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(cfg.model_path),
            monitor=MONITOR,
            mode=MONITOR_MODE,
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor=MONITOR,
            mode=MONITOR_MODE,
            patience=int(t["early_stopping_patience"]),
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.4,
            patience=int(t["reduce_lr_patience"]),
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            str(cfg.logs_dir / f"history_{phase}.csv"), append=False
        ),
        keras.callbacks.TerminateOnNaN(),
    ]


def _history_frame(history: keras.callbacks.History, phase: str,
                   epoch_offset: int = 0) -> pd.DataFrame:
    df = pd.DataFrame(history.history)
    df.insert(0, "epoch", np.arange(len(df)) + 1 + epoch_offset)
    df.insert(1, "phase", phase)
    return df


# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
def plot_training_curves(history_df: pd.DataFrame, out_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [m for m in ("loss", "accuracy", "auc", "precision", "recall")
               if m in history_df.columns]
    if not metrics:
        metrics = ["loss"]

    ncols = min(3, len(metrics))
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows), squeeze=False)

    switch = history_df.loc[history_df["phase"] == "finetune", "epoch"]
    switch_epoch = float(switch.min()) - 0.5 if len(switch) else None

    for ax, metric in zip(axes.ravel(), metrics):
        ax.plot(history_df["epoch"], history_df[metric], label=f"train {metric}", lw=2)
        vkey = f"val_{metric}"
        if vkey in history_df.columns:
            ax.plot(history_df["epoch"], history_df[vkey], label=f"val {metric}",
                    lw=2, ls="--")
        if switch_epoch is not None:
            ax.axvline(switch_epoch, color="grey", ls=":", lw=1.5)
            ax.text(switch_epoch, ax.get_ylim()[1], " fine-tune", fontsize=8,
                    va="top", color="grey")
        ax.set_title(metric)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    for ax in axes.ravel()[len(metrics):]:
        ax.axis("off")

    fig.suptitle("Training curves - landfill detector", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def train(
    cfg: Optional[Config] = None,
    epochs_head: Optional[int] = None,
    epochs_finetune: Optional[int] = None,
    skip_finetune: bool = False,
) -> Dict:
    cfg = cfg or load_config()
    cfg.ensure_dirs()
    set_seed(cfg.seed)

    t = cfg["train"]
    if bool(t.get("mixed_precision", False)):
        keras.mixed_precision.set_global_policy("mixed_float16")
        LOGGER.info("Mixed precision enabled (float16 compute, float32 output).")

    gpus = tf.config.list_physical_devices("GPU")
    LOGGER.info("TensorFlow %s | devices: %s", tf.__version__,
                [g.name for g in gpus] or "CPU only")

    df = load_manifest(cfg)
    stats = dataset_stats(cfg, df)
    LOGGER.info("Dataset: %d images | labels %s", stats["total_images"], stats["label_counts"])

    train_ds = make_dataset(cfg, "train", df=df)
    val_ds = make_dataset(cfg, "val", df=df, shuffle=False, augment=False)

    class_weight = (
        compute_class_weights(cfg, df) if bool(t.get("use_class_weights", True)) else None
    )

    model = build_model(cfg, freeze_backbone=True)

    # ---------------- phase 1: head warm-up ----------------
    e_head = int(epochs_head if epochs_head is not None else t["epochs_head"])
    frames: List[pd.DataFrame] = []
    epochs_done = 0

    if e_head > 0:
        freeze_backbone(model)
        compile_model(model, cfg, float(t["lr_head"]))
        LOGGER.info("=" * 66)
        LOGGER.info("PHASE 1/2  head warm-up  |  %d epochs  |  lr=%s",
                    e_head, t["lr_head"])
        LOGGER.info("=" * 66)
        with timer("Phase 1"):
            h1 = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=e_head,
                callbacks=build_callbacks(cfg, "head"),
                class_weight=class_weight,
                verbose=1,
            )
        frames.append(_history_frame(h1, "head"))
        epochs_done = len(h1.history.get("loss", []))

    # ---------------- phase 2: fine-tuning ----------------
    e_ft = int(epochs_finetune if epochs_finetune is not None else t["epochs_finetune"])
    if e_ft > 0 and not skip_finetune:
        unfreeze_backbone(model, last_n=int(t["unfreeze_last_n"]))
        compile_model(model, cfg, float(t["lr_finetune"]))
        LOGGER.info("=" * 66)
        LOGGER.info("PHASE 2/2  fine-tuning  |  %d epochs  |  lr=%s",
                    e_ft, t["lr_finetune"])
        LOGGER.info("=" * 66)
        with timer("Phase 2"):
            h2 = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=e_ft,
                callbacks=build_callbacks(cfg, "finetune"),
                class_weight=class_weight,
                verbose=1,
            )
        frames.append(_history_frame(h2, "finetune", epoch_offset=epochs_done))

    # ---------------- persist ----------------
    if not cfg.model_path.exists():   # e.g. 0 epochs, or checkpoint never improved
        model.save(cfg.model_path)
    LOGGER.info("Best model saved: %s", cfg.model_path)

    history_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(history_df):
        hist_path = cfg.reports_dir / "history.csv"
        history_df.to_csv(hist_path, index=False)
        fig_path = plot_training_curves(history_df, cfg.reports_dir / "figures" / "training_curves.png")
        LOGGER.info("History: %s | curves: %s", hist_path, fig_path)

    best = {}
    if len(history_df):
        for col in history_df.columns:
            if col.startswith("val_"):
                series = history_df[col].dropna()
                if len(series):
                    best[col] = float(series.max() if col != "val_loss" else series.min())

    meta = {
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "tensorflow": tf.__version__,
        "model_path": str(cfg.model_path),
        "class_names": cfg.class_names,
        "task_mode": cfg["task"]["mode"],
        "positive_label": cfg["task"]["positive_label"],
        "negative_label": cfg["task"]["negative_label"],
        "positive_source_classes": cfg["task"]["positive_classes"],
        "negative_source_classes": cfg["task"]["negative_classes"],
        "threshold": cfg.threshold,
        "image_size": list(cfg.image_size),
        "backbone": cfg["model"]["backbone"],
        "data_source": cfg["data"]["source"],
        "dataset_stats": stats,
        "epochs_head": e_head,
        "epochs_finetune": 0 if skip_finetune else e_ft,
        "best_val_metrics": best,
    }
    save_json(meta, cfg.model_meta_path)
    LOGGER.info("Metadata: %s", cfg.model_meta_path)
    LOGGER.info("Best validation metrics: %s", json.dumps(best, indent=2))
    return meta


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Train the landfill detector.")
    p.add_argument("--config", default=None, help="path to config.yaml")
    p.add_argument("--epochs-head", type=int, default=None)
    p.add_argument("--epochs-finetune", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--backbone", default=None,
                   choices=["mobilenetv2", "efficientnetb0", "resnet50v2"])
    p.add_argument("--max-per-class", type=int, default=None,
                   help="cap images per source class (quick smoke run)")
    p.add_argument("--no-finetune", action="store_true",
                   help="run only the frozen-backbone warm-up phase")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.batch_size:
        cfg.raw["data"]["batch_size"] = args.batch_size
    if args.backbone:
        cfg.raw["model"]["backbone"] = args.backbone
    if args.max_per_class is not None:
        cfg.raw["data"]["max_per_class"] = args.max_per_class

    train(
        cfg,
        epochs_head=args.epochs_head,
        epochs_finetune=args.epochs_finetune,
        skip_finetune=args.no_finetune,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

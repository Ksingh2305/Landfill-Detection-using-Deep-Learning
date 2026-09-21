"""Held-out evaluation: metrics, curves and error analysis.

Produces (under ``reports/``):

``metrics.json``                     every headline number, machine readable
``predictions_test.csv``             per-image scores for further analysis
``figures/confusion_matrix.png``
``figures/roc_pr_curves.png``
``figures/threshold_sweep.png``
``figures/per_source_class.png``     recall broken down by land-cover type
``figures/error_gallery.png``        the most confident mistakes

The per-source-class breakdown is the interesting one for this task: it
shows *which* land covers the detector confuses with waste sites (bare
soil and quarries are the usual suspects), which is exactly the failure
mode an environmental monitoring team needs to know about.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

from .utils import quiet_tensorflow

quiet_tensorflow()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import Config, load_config  # noqa: E402
from .data.preprocess import load_manifest, resolve_path  # noqa: E402
from .predict import load_detector, predict_probabilities  # noqa: E402
from .utils import LOGGER, load_image_rgb, save_json, timer  # noqa: E402


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------
def score_split(
    cfg: Config,
    split: str = "test",
    batch_size: int = 64,
) -> pd.DataFrame:
    """Run the model over one manifest split; returns a per-image DataFrame."""
    df = load_manifest(cfg)
    sub = df[df["split"] == split].reset_index(drop=True)
    if not len(sub):
        raise ValueError(f"Split '{split}' is empty.")

    model, meta = load_detector(cfg)
    image_size = tuple(meta.get("image_size", cfg.image_size))

    all_probs: List[np.ndarray] = []
    with timer(f"Scoring {len(sub)} {split} images"):
        for start in range(0, len(sub), batch_size):
            chunk = sub.iloc[start : start + batch_size]
            images = [load_image_rgb(resolve_path(p), image_size) for p in chunk["path"]]
            all_probs.append(predict_probabilities(model, images, image_size, batch_size))

    probs = np.concatenate(all_probs, axis=0)
    out = sub.copy()
    class_names = list(meta.get("class_names", cfg.class_names))
    for i, name in enumerate(class_names[: probs.shape[1]]):
        out[f"p_{name}"] = probs[:, i]
    out["positive_score"] = probs[:, 1] if probs.shape[1] == 2 else probs.max(axis=1)
    out["pred_idx"] = probs.argmax(axis=1)
    return out


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def compute_metrics(cfg: Config, scored: pd.DataFrame, threshold: float) -> Dict:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        classification_report,
        cohen_kappa_score,
        confusion_matrix,
        matthews_corrcoef,
        roc_auc_score,
    )

    y_true = scored["label_idx"].to_numpy()
    class_names = cfg.class_names

    if cfg.is_binary:
        y_score = scored["positive_score"].to_numpy()
        y_pred = (y_score >= threshold).astype(int)
    else:
        y_pred = scored["pred_idx"].to_numpy()
        y_score = None

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    report = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, output_dict=True, zero_division=0,
    )

    metrics: Dict = {
        "n_images": int(len(scored)),
        "threshold": float(threshold),
        "class_names": class_names,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true)) > 1 else None,
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in class_names if name in report
        },
    }

    if cfg.is_binary and y_score is not None and len(set(y_true)) > 1:
        tn, fp, fn, tp = cm.ravel()
        metrics.update(
            {
                "roc_auc": float(roc_auc_score(y_true, y_score)),
                "average_precision": float(average_precision_score(y_true, y_score)),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
                "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
                "false_alarm_rate": float(fp / (fp + tn)) if (fp + tn) else None,
                "miss_rate": float(fn / (fn + tp)) if (fn + tp) else None,
            }
        )

    # Recall per original land-cover class - the useful error analysis.
    per_source = {}
    for source_class, grp in scored.groupby("source_class"):
        gt = grp["label_idx"].to_numpy()
        if cfg.is_binary:
            pred = (grp["positive_score"].to_numpy() >= threshold).astype(int)
        else:
            pred = grp["pred_idx"].to_numpy()
        per_source[str(source_class)] = {
            "n": int(len(grp)),
            "accuracy": float((pred == gt).mean()),
            "mean_positive_score": float(grp["positive_score"].mean()),
            "task_label": str(grp["label"].iloc[0]),
        }
    metrics["per_source_class"] = per_source
    return metrics


def threshold_sweep(scored: pd.DataFrame, steps: int = 101) -> pd.DataFrame:
    """Precision / recall / F1 / Youden J across candidate decision thresholds."""
    y_true = scored["label_idx"].to_numpy()
    y_score = scored["positive_score"].to_numpy()
    rows = []
    for thr in np.linspace(0.0, 1.0, steps):
        pred = (y_score >= thr).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "threshold": float(thr),
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1": f1,
                "youden_j": recall + specificity - 1.0,
                "accuracy": (tp + tn) / max(len(y_true), 1),
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], out: Path) -> Path:
    plt = _mpl()
    cm = np.asarray(cm, dtype=float)
    norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, mat, title, fmt in (
        (axes[0], cm, "Confusion matrix (counts)", "{:.0f}"),
        (axes[1], norm, "Row-normalised (recall)", "{:.2f}"),
    ):
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max() if mat.max() else 1)
        ax.set_xticks(range(len(class_names)), class_names, rotation=20, ha="right")
        ax.set_yticks(range(len(class_names)), class_names)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.set_title(title)
        thresh = mat.max() / 2 if mat.max() else 0.5
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center",
                        color="white" if mat[i, j] > thresh else "black", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_roc_pr(scored: pd.DataFrame, out: Path) -> Optional[Path]:
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    y_true = scored["label_idx"].to_numpy()
    if len(set(y_true)) < 2:
        return None
    y_score = scored["positive_score"].to_numpy()

    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    fpr, tpr, _ = roc_curve(y_true, y_score)
    axes[0].plot(fpr, tpr, lw=2.2,
                 label=f"AUC = {roc_auc_score(y_true, y_score):.4f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    axes[0].set_xlabel("false positive rate")
    axes[0].set_ylabel("true positive rate")
    axes[0].set_title("ROC curve")

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    axes[1].plot(recall, precision, lw=2.2,
                 label=f"AP = {average_precision_score(y_true, y_score):.4f}")
    baseline = float((y_true == 1).mean())
    axes[1].axhline(baseline, color="k", ls="--", lw=1, label=f"prevalence = {baseline:.2f}")
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[1].set_title("Precision-recall curve")

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(loc="lower left" if ax is axes[0] else "lower right")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_threshold_sweep(sweep: pd.DataFrame, chosen: float, out: Path) -> Path:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for col, style in (("precision", "-"), ("recall", "-"), ("f1", "-"),
                       ("specificity", "--")):
        ax.plot(sweep["threshold"], sweep[col], style, lw=2, label=col)
    best = sweep.loc[sweep["f1"].idxmax()]
    ax.axvline(chosen, color="grey", ls=":", lw=1.6, label=f"configured = {chosen:.2f}")
    ax.axvline(best["threshold"], color="crimson", ls=":", lw=1.6,
               label=f"best F1 = {best['threshold']:.2f}")
    ax.set_xlabel("decision threshold on P(landfill)")
    ax.set_ylabel("score")
    ax.set_title("Operating-point selection")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_per_source_class(metrics: Dict, out: Path) -> Path:
    plt = _mpl()
    data = metrics["per_source_class"]
    names = sorted(data, key=lambda k: data[k]["mean_positive_score"], reverse=True)
    scores = [data[n]["mean_positive_score"] for n in names]
    acc = [data[n]["accuracy"] for n in names]
    positive_task = metrics.get("positive_label", "")
    colours = ["#c0392b" if data[n]["task_label"] == positive_task else "#2c7fb8"
               for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].barh(names, scores, color=colours)
    axes[0].set_xlabel("mean P(landfill candidate)")
    axes[0].set_title("Model response by land-cover class")
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1)

    axes[1].barh(names, acc, color=colours)
    axes[1].set_xlabel("accuracy")
    axes[1].set_title("Accuracy by land-cover class")
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1)

    for ax in axes:
        ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_error_gallery(cfg: Config, scored: pd.DataFrame, threshold: float,
                       out: Path, n: int = 12) -> Optional[Path]:
    """The most confident false positives and false negatives."""
    y_true = scored["label_idx"].to_numpy()
    pred = (scored["positive_score"].to_numpy() >= threshold).astype(int)
    scored = scored.copy()
    scored["correct"] = pred == y_true

    fps = scored[(pred == 1) & (y_true == 0)].nlargest(n // 2, "positive_score")
    fns = scored[(pred == 0) & (y_true == 1)].nsmallest(n // 2, "positive_score")
    picks = pd.concat([fps, fns])
    if picks.empty:
        LOGGER.info("No misclassifications on the test split - skipping error gallery.")
        return None

    plt = _mpl()
    cols = min(6, len(picks))
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.6 * rows), squeeze=False)

    for ax, (_, row) in zip(axes.ravel(), picks.iterrows()):
        try:
            ax.imshow(load_image_rgb(resolve_path(row["path"])))
        except Exception:  # noqa: BLE001
            ax.axis("off")
            continue
        kind = "FP" if row["label_idx"] == 0 else "FN"
        ax.set_title(f"{kind} {row['source_class']}\np={row['positive_score']:.2f}",
                     fontsize=8, color="#c0392b" if kind == "FP" else "#b7791f")
        ax.axis("off")
    for ax in axes.ravel()[len(picks):]:
        ax.axis("off")

    fig.suptitle("Most confident errors (FP = false alarm, FN = missed site)", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def evaluate(
    cfg: Optional[Config] = None,
    split: str = "test",
    threshold: Optional[float] = None,
    make_figures: bool = True,
) -> Dict:
    cfg = cfg or load_config()
    cfg.ensure_dirs()
    figures = cfg.reports_dir / "figures"

    scored = score_split(cfg, split=split)
    thr = float(threshold if threshold is not None else cfg.threshold)

    metrics = compute_metrics(cfg, scored, thr)
    metrics["split"] = split
    metrics["positive_label"] = cfg["task"]["positive_label"]
    metrics["data_source"] = cfg["data"]["source"]

    if cfg.is_binary:
        sweep = threshold_sweep(scored)
        best = sweep.loc[sweep["f1"].idxmax()]
        metrics["best_f1_threshold"] = float(best["threshold"])
        metrics["best_f1"] = float(best["f1"])
        best_j = sweep.loc[sweep["youden_j"].idxmax()]
        metrics["best_youden_threshold"] = float(best_j["threshold"])
        sweep.to_csv(cfg.reports_dir / "threshold_sweep.csv", index=False)
    else:
        sweep = None

    pred_csv = cfg.reports_dir / f"predictions_{split}.csv"
    scored.to_csv(pred_csv, index=False)
    save_json(metrics, cfg.reports_dir / "metrics.json")

    if make_figures:
        plot_confusion_matrix(np.array(metrics["confusion_matrix"]),
                              cfg.class_names, figures / "confusion_matrix.png")
        if cfg.is_binary:
            plot_roc_pr(scored, figures / "roc_pr_curves.png")
            plot_threshold_sweep(sweep, thr, figures / "threshold_sweep.png")
            plot_error_gallery(cfg, scored, thr, figures / "error_gallery.png")
        plot_per_source_class(metrics, figures / "per_source_class.png")

    _log_summary(metrics)
    LOGGER.info("Metrics  -> %s", cfg.reports_dir / "metrics.json")
    LOGGER.info("Per-image-> %s", pred_csv)
    LOGGER.info("Figures  -> %s", figures)
    return metrics


def _log_summary(m: Dict) -> None:
    LOGGER.info("=" * 62)
    LOGGER.info("EVALUATION  (%s split, threshold %.2f)", m["split"], m["threshold"])
    LOGGER.info("=" * 62)
    LOGGER.info("  accuracy           %.4f", m["accuracy"])
    LOGGER.info("  balanced accuracy  %.4f", m["balanced_accuracy"])
    LOGGER.info("  macro F1           %.4f", m["macro_f1"])
    if m.get("roc_auc") is not None:
        LOGGER.info("  ROC AUC            %.4f", m["roc_auc"])
        LOGGER.info("  average precision  %.4f", m["average_precision"])
        LOGGER.info("  miss rate (FN)     %.4f", m.get("miss_rate") or 0.0)
        LOGGER.info("  false alarm rate   %.4f", m.get("false_alarm_rate") or 0.0)
    for name, s in m["per_class"].items():
        LOGGER.info("  %-22s P=%.3f R=%.3f F1=%.3f  (n=%d)",
                    name, s["precision"], s["recall"], s["f1"], s["support"])


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate the trained detector.")
    p.add_argument("--config", default=None)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args(argv)

    evaluate(load_config(args.config), split=args.split, threshold=args.threshold,
             make_figures=not args.no_figures)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

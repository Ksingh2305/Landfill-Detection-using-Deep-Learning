"""Evaluation dashboard: metrics, curves, error analysis, threshold what-if."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _shared as sh

sh.page_config("Model Performance", icon="📊")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

cfg = sh.get_config()
sh.sidebar_status(cfg)

sh.header(
    "Model performance",
    "Everything from the held-out test split: headline metrics, curves, per-land-cover "
    "error analysis and an interactive operating-point chooser.",
)

metrics = sh.metrics_report(cfg)
meta = sh.model_meta(cfg)
figures = cfg.reports_dir / "figures"

if not metrics:
    st.warning("No evaluation report yet.")
    st.markdown("Run `python scripts/03_evaluate.py` (or use the **Pipeline** page).")
    st.stop()

# ----------------------------------------------------------------------
# Headline metrics
# ----------------------------------------------------------------------
sh.metric_row(
    [
        ("Accuracy", f"{metrics['accuracy']:.2%}"),
        ("Balanced accuracy", f"{metrics['balanced_accuracy']:.2%}",
         "Mean of per-class recall - the honest number under class imbalance"),
        ("Macro F1", f"{metrics['macro_f1']:.4f}"),
        ("ROC AUC", f"{metrics['roc_auc']:.4f}" if metrics.get("roc_auc") else "-"),
        ("Average precision", f"{metrics['average_precision']:.4f}"
         if metrics.get("average_precision") else "-"),
    ]
)

if metrics.get("miss_rate") is not None:
    sh.metric_row(
        [
            ("Miss rate", f"{metrics['miss_rate']:.2%}",
             "Fraction of real waste sites the model let through"),
            ("False alarm rate", f"{metrics['false_alarm_rate']:.2%}",
             "Fraction of clean land wrongly flagged"),
            ("Specificity", f"{metrics['specificity']:.2%}"),
            ("MCC", f"{metrics['mcc']:.4f}" if metrics.get("mcc") else "-",
             "Matthews correlation - robust to imbalance"),
            ("Test images", f"{metrics['n_images']:,}"),
        ]
    )

st.caption(
    f"Evaluated on the **{metrics.get('split', 'test')}** split at threshold "
    f"**{metrics['threshold']:.2f}** | backbone **{meta.get('backbone', '?')}** | "
    f"data source **{metrics.get('data_source', '?')}**"
)

if str(metrics.get("data_source", "")).lower() == "synthetic":
    st.warning(
        "These numbers come from the **synthetic** stand-in dataset. They demonstrate that "
        "the pipeline works; they say nothing about real-world performance.",
        icon="⚠",
    )

st.divider()

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
tab_curves, tab_matrix, tab_classes, tab_threshold, tab_training, tab_errors = st.tabs(
    ["Curves", "Confusion matrix", "By land cover", "Operating point", "Training", "Errors"]
)

with tab_curves:
    img = figures / "roc_pr_curves.png"
    if img.exists():
        st.image(str(img), **sh.STRETCH)
    else:
        st.caption("ROC/PR figure not found - re-run the evaluation script.")
    st.markdown(
        """
ROC AUC answers "if I pick one waste site and one clean tile at random, how often does the
model score the waste site higher?" Average precision is the more informative number when
positives are rare, because it only cares about the ranking of the flagged set.
        """
    )

with tab_matrix:
    img = figures / "confusion_matrix.png"
    if img.exists():
        st.image(str(img), **sh.STRETCH)
    cm = np.array(metrics["confusion_matrix"])
    names = metrics["class_names"]
    st.dataframe(
        pd.DataFrame(cm, index=[f"actual {n}" for n in names],
                     columns=[f"predicted {n}" for n in names]),
        **sh.STRETCH,
    )
    st.markdown("**Per-class scores**")
    st.dataframe(
        pd.DataFrame(metrics["per_class"]).T.reset_index().rename(columns={"index": "class"}),
        **sh.STRETCH, hide_index=True,
    )

with tab_classes:
    img = figures / "per_source_class.png"
    if img.exists():
        st.image(str(img), **sh.STRETCH)
    per_source = metrics.get("per_source_class", {})
    if per_source:
        df = pd.DataFrame(per_source).T.reset_index().rename(columns={"index": "land cover"})
        df = df.sort_values("mean_positive_score", ascending=False)
        st.dataframe(
            df, **sh.STRETCH, hide_index=True,
            column_config={
                "mean_positive_score": st.column_config.ProgressColumn(
                    "mean P(landfill)", min_value=0.0, max_value=1.0, format="%.3f"),
                "accuracy": st.column_config.ProgressColumn(
                    "accuracy", min_value=0.0, max_value=1.0, format="%.3f"),
            },
        )
    st.markdown(
        """
This is the table to read before trusting the detector in the field. The land covers with
a high mean score but a *negative* task label are the model's systematic false alarms -
typically bare soil, quarries and large flat roofs, which share the spectral signature of
an active waste cell. Adding hard negatives from those classes is the cheapest way to
improve real-world precision.
        """
    )

with tab_threshold:
    sweep_path = cfg.reports_dir / "threshold_sweep.csv"
    pred_path = cfg.reports_dir / f"predictions_{metrics.get('split', 'test')}.csv"

    if sweep_path.exists():
        sweep = pd.read_csv(sweep_path)
        chosen = st.slider("Decision threshold", 0.0, 1.0, float(metrics["threshold"]), 0.01)
        row = sweep.iloc[(sweep["threshold"] - chosen).abs().idxmin()]

        sh.metric_row(
            [
                ("Precision", f"{row['precision']:.3f}",
                 "Of the tiles flagged, how many are real"),
                ("Recall", f"{row['recall']:.3f}",
                 "Of the real sites, how many were caught"),
                ("F1", f"{row['f1']:.3f}"),
                ("Specificity", f"{row['specificity']:.3f}"),
                ("Accuracy", f"{row['accuracy']:.3f}"),
            ]
        )

        st.line_chart(sweep.set_index("threshold")[["precision", "recall", "f1", "specificity"]],
                      height=320)

        if pred_path.exists():
            preds = pd.read_csv(pred_path)
            y = preds["label_idx"].to_numpy()
            s = preds["positive_score"].to_numpy()
            pred = (s >= chosen).astype(int)
            tp = int(((pred == 1) & (y == 1)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            tn = int(((pred == 0) & (y == 0)).sum())
            st.markdown("**Confusion counts at this threshold**")
            st.dataframe(
                pd.DataFrame(
                    [[tn, fp], [fn, tp]],
                    index=["actual clean", "actual landfill"],
                    columns=["predicted clean", "predicted landfill"],
                ),
                **sh.STRETCH,
            )

        st.info(
            f"Best F1 is at threshold **{metrics.get('best_f1_threshold', float('nan')):.2f}** "
            f"(F1 = {metrics.get('best_f1', float('nan')):.3f}). To lock it in, set "
            "`task.threshold` in `config.yaml`.",
            icon="💡",
        )
        img = figures / "threshold_sweep.png"
        if img.exists():
            st.image(str(img), **sh.STRETCH)
    else:
        st.caption("Threshold sweep not found - re-run the evaluation script.")

with tab_training:
    hist_path = cfg.reports_dir / "history.csv"
    img = figures / "training_curves.png"
    if img.exists():
        st.image(str(img), **sh.STRETCH)
    if hist_path.exists():
        hist = pd.read_csv(hist_path)
        cols = [c for c in ["loss", "val_loss", "accuracy", "val_accuracy", "auc", "val_auc"]
                if c in hist.columns]
        if cols:
            st.line_chart(hist.set_index("epoch")[cols], height=320)
        st.dataframe(hist, **sh.STRETCH, hide_index=True, height=260)
    else:
        st.caption("No training history found.")

with tab_errors:
    img = figures / "error_gallery.png"
    if img.exists():
        st.image(str(img), **sh.STRETCH)
        st.caption(
            "FP = clean land flagged as a site (wasted field visits). "
            "FN = a real site the model missed (the expensive kind of error)."
        )
    pred_path = cfg.reports_dir / f"predictions_{metrics.get('split', 'test')}.csv"
    if pred_path.exists():
        preds = pd.read_csv(pred_path)
        thr = float(metrics["threshold"])
        preds["predicted_positive"] = preds["positive_score"] >= thr
        preds["correct"] = preds["predicted_positive"].astype(int) == preds["label_idx"]
        wrong = preds[~preds["correct"]].sort_values("positive_score", ascending=False)
        st.markdown(f"**{len(wrong)} misclassified test images**")
        st.dataframe(
            wrong[["path", "source_class", "label", "positive_score"]],
            **sh.STRETCH, hide_index=True, height=320,
        )
        st.download_button("Download all test predictions (CSV)",
                           data=preds.to_csv(index=False),
                           file_name="predictions_test.csv", mime="text/csv")

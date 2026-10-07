"""Landing page of the landfill detection console.

Run from the project root:

    streamlit run app/Home.py
"""

from __future__ import annotations

import _shared as sh

sh.page_config("Overview", icon="🛰")

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.utils import load_json  # noqa: E402

cfg = sh.get_config()
sh.sidebar_status(cfg)

sh.header(
    "Landfill Detection from Satellite Imagery",
    "A deep-learning pipeline that screens Sentinel-2 style imagery for waste-site-like "
    "surfaces: preprocessing, transfer-learned classification, Grad-CAM explanations and "
    "sliding-window scanning of full scenes.",
)

meta = sh.model_meta(cfg)
metrics = sh.metrics_report(cfg)
stats = load_json(cfg.interim_dir / "dataset_stats.json", default={}) or {}
trained = cfg.model_path.exists()

# ----------------------------------------------------------------------
# Status strip
# ----------------------------------------------------------------------
sh.metric_row(
    [
        ("Model", meta.get("backbone", "-") if trained else "not trained",
         "Backbone architecture behind the detector"),
        ("Test accuracy",
         f"{metrics['accuracy']:.1%}" if metrics.get("accuracy") is not None else "-",
         "Accuracy on the held-out test split"),
        ("ROC AUC",
         f"{metrics['roc_auc']:.4f}" if metrics.get("roc_auc") is not None else "-",
         "Threshold-independent ranking quality"),
        ("Training images",
         f"{stats.get('total_images', 0):,}" if stats else "-",
         "Total labelled tiles in the manifest"),
        ("Decision threshold", f"{float(meta.get('threshold', cfg.threshold)):.2f}",
         "Probability above which a tile is flagged"),
    ]
)

st.write("")

if not trained:
    st.info(
        "**Nothing trained yet.** Open the **Pipeline** page in the sidebar to run the "
        "three setup steps from here, or run `python scripts/run_pipeline.py --quick` "
        "in a terminal.",
        icon="ℹ",
    )

# ----------------------------------------------------------------------
# How it works
# ----------------------------------------------------------------------
st.subheader("How the pipeline works")

steps = [
    ("1", "Acquire",
     "EuroSAT Sentinel-2 tiles (27,000 labelled 64x64 RGB chips, 10 land-cover classes) "
     "are downloaded once and normalised into a class-per-folder layout."),
    ("2", "Preprocess",
     "A stratified, seeded manifest assigns every image a task label and a train/val/test "
     "split. A tf.data pipeline decodes, resizes, caches and augments in parallel."),
    ("3", "Train",
     "An ImageNet backbone is transfer-learned in two phases - frozen head warm-up, then "
     "low-rate fine-tuning of the deepest layers with BatchNorm held frozen."),
    ("4", "Evaluate",
     "Confusion matrix, ROC/PR curves, a threshold sweep and a per-land-cover breakdown "
     "that shows exactly which surfaces trigger false alarms."),
    ("5", "Explain",
     "Grad-CAM projects the classifier's evidence back onto the tile, so an analyst can "
     "see whether the model looked at the waste cells or at a car park."),
    ("6", "Scan",
     "A sliding window sweeps a full scene, accumulates overlapping scores into a "
     "probability surface, and merges hot windows into candidate site boxes."),
]

for row in range(0, len(steps), 3):
    cols = st.columns(3)
    for col, (num, title, body) in zip(cols, steps[row : row + 3]):
        col.markdown(
            f"""<div class="card">
                  <h4><span class="step-num">{num}</span>{title}</h4>
                  <p>{body}</p>
                </div>""",
            unsafe_allow_html=True,
        )
    st.write("")

# ----------------------------------------------------------------------
# Dataset composition
# ----------------------------------------------------------------------
st.subheader("Dataset composition")

if stats:
    left, right = st.columns([1.35, 1])

    with left:
        src = stats.get("source_class_counts", {})
        if src:
            pos = set(cfg["task"]["positive_classes"])
            df = pd.DataFrame(
                {
                    "land cover": list(src),
                    "images": list(src.values()),
                    "role": ["positive (waste-site-like)" if k in pos else "negative (clean land)"
                             for k in src],
                }
            ).sort_values("images", ascending=False)
            st.bar_chart(df.set_index("land cover")["images"], height=320)
            st.dataframe(df, **sh.STRETCH, hide_index=True)

    with right:
        split = stats.get("split_counts", {})
        if split:
            st.caption("Split sizes")
            st.dataframe(
                pd.DataFrame({"split": list(split), "images": list(split.values())}),
                **sh.STRETCH, hide_index=True,
            )
        labels = stats.get("label_counts", {})
        if labels:
            st.caption("Task labels")
            st.dataframe(
                pd.DataFrame({"label": list(labels), "images": list(labels.values())}),
                **sh.STRETCH, hide_index=True,
            )
        if stats.get("imbalance_ratio"):
            st.caption(
                f"Class imbalance ratio **{stats['imbalance_ratio']}:1** - handled with "
                "inverse-frequency class weights during training."
            )
else:
    st.caption("No manifest yet - run step 1 of the pipeline.")

# ----------------------------------------------------------------------
# Honest framing
# ----------------------------------------------------------------------
st.subheader("What this model actually detects")
st.markdown(
    """
EuroSAT has no *landfill* class. This project therefore trains on a clearly labelled
**proxy**: the positive class is built from land covers whose spectral and textural
signature matches active waste sites - bare, disturbed, anthropogenic surfaces
(`Industrial`, `Highway`) - against natural cover and ordinary housing.

That gives you a genuine, reproducible satellite-imagery detector and a feature
extractor that transfers well. It does **not** give you a validated landfill
classifier. To make it one, drop real labelled chips into `data/custom/landfill/`
and `data/custom/not_landfill/`, set `data.source: custom` in `config.yaml`, and
retrain - the rest of the pipeline is unchanged.

The **About** page spells out the limitations in full.
    """
)

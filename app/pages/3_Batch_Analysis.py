"""Batch screening of many tiles at once."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _shared as sh

sh.page_config("Batch Analysis", icon="📦")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.predict import predict_images  # noqa: E402
from src.utils import list_images, load_image_rgb  # noqa: E402

cfg = sh.get_config()
sh.sidebar_status(cfg)

sh.header(
    "Batch analysis",
    "Screen a whole folder or zip of image chips, rank them by landfill probability and "
    "export the results for follow-up.",
)

model, meta = sh.require_model(cfg)
image_size = tuple(meta.get("image_size", cfg.image_size))

threshold = sh.threshold_slider(cfg, key="batch_thr")

mode = st.radio("Input", ["Upload images", "Upload a .zip", "Folder on this computer"],
                horizontal=True)

images: list = []
names: list = []

if mode == "Upload images":
    ups = st.file_uploader(
        "Image chips", accept_multiple_files=True,
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
    )
    for up in ups or []:
        images.append(sh.read_upload(up))
        names.append(up.name)

elif mode == "Upload a .zip":
    zup = st.file_uploader("Zip archive of images", type=["zip"])
    if zup is not None:
        with zipfile.ZipFile(zup) as zf:
            members = [
                m for m in zf.namelist()
                if not m.endswith("/")
                and Path(m).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
                and not Path(m).name.startswith(".")
            ]
            st.caption(f"{len(members)} image(s) in the archive")
            for m in members[:2000]:
                with zf.open(m) as fh:
                    images.append(load_image_rgb(fh))
                names.append(m)

else:
    default = str((cfg.data_root / "samples").resolve())
    folder = st.text_input("Folder path", value=default,
                           help="Any folder on this machine containing image files.")
    if folder:
        paths = list_images(Path(folder))
        st.caption(f"{len(paths)} image(s) found")
        limit = st.number_input("Maximum images to score", 1, 20000, min(len(paths) or 1, 500))
        for p in paths[: int(limit)]:
            images.append(load_image_rgb(p))
            names.append(p.name)

if not images:
    st.caption("Waiting for images ...")
    st.stop()

# ----------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------
with st.spinner(f"Classifying {len(images)} chips ..."):
    preds = predict_images(images, cfg, model=model, meta=meta,
                           threshold=threshold, sources=names)

df = pd.DataFrame(
    {
        "image": names,
        "verdict": ["LANDFILL CANDIDATE" if p.is_positive else "clean" for p in preds],
        "P(landfill)": [p.positive_score for p in preds],
        "confidence": [p.confidence for p in preds],
    }
).sort_values("P(landfill)", ascending=False).reset_index(drop=True)

n_pos = int(df["verdict"].eq("LANDFILL CANDIDATE").sum())
borderline = int(((df["P(landfill)"] - threshold).abs() < 0.05).sum())

sh.metric_row(
    [
        ("Images screened", f"{len(df):,}"),
        ("Flagged", f"{n_pos:,}", "Scores at or above the threshold"),
        ("Flag rate", f"{n_pos / max(len(df), 1):.1%}"),
        ("Borderline", f"{borderline:,}", "Within 0.05 of the threshold - review manually"),
        ("Mean score", f"{df['P(landfill)'].mean():.3f}"),
    ]
)

st.dataframe(
    df, **sh.STRETCH, hide_index=True, height=380,
    column_config={
        "P(landfill)": st.column_config.ProgressColumn(
            "P(landfill)", min_value=0.0, max_value=1.0, format="%.3f"),
        "confidence": st.column_config.NumberColumn("confidence", format="%.3f"),
    },
)

st.download_button(
    "Download results (CSV)",
    data=df.to_csv(index=False),
    file_name="landfill_batch_results.csv",
    mime="text/csv",
)

# ----------------------------------------------------------------------
# Galleries
# ----------------------------------------------------------------------
order = np.argsort([-p.positive_score for p in preds])

st.subheader("Highest-scoring chips")
top_idx = [i for i in order if preds[i].positive_score >= threshold][:12]
if top_idx:
    cols = st.columns(6)
    for n, i in enumerate(top_idx):
        cols[n % 6].image(sh.to_display(images[i]),
                          caption=f"{preds[i].positive_score:.3f}\n{Path(names[i]).name[:18]}",
                          **sh.STRETCH)
else:
    st.caption("Nothing crossed the threshold in this batch.")

with st.expander("Lowest-scoring chips"):
    low_idx = list(order[::-1])[:12]
    cols = st.columns(6)
    for n, i in enumerate(low_idx):
        cols[n % 6].image(sh.to_display(images[i]),
                          caption=f"{preds[i].positive_score:.3f}",
                          **sh.STRETCH)

# ----------------------------------------------------------------------
# Distribution
# ----------------------------------------------------------------------
with st.expander("Score distribution"):
    st.bar_chart(sh.score_histogram(df["P(landfill)"], bins=20, label="images"),
                 height=260)
    st.caption(
        f"With the threshold at {threshold:.2f}, {n_pos} of {len(df)} chips are flagged. "
        "Move the slider above to see how the flag rate responds - the Model Performance "
        "page shows the matching precision and recall on labelled data."
    )

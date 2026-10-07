"""Single-tile detection with Grad-CAM explanation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _shared as sh

sh.page_config("Detect", icon="🔍")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.models.gradcam import explain  # noqa: E402
from src.predict import predict_images  # noqa: E402
from src.utils import load_image_rgb  # noqa: E402

cfg = sh.get_config()
sh.sidebar_status(cfg)

sh.header(
    "Tile detection",
    "Classify a single satellite image chip and see which pixels drove the decision.",
)

model, meta = sh.require_model(cfg)
image_size = tuple(meta.get("image_size", cfg.image_size))

# ----------------------------------------------------------------------
# Input
# ----------------------------------------------------------------------
left, right = st.columns([1, 1])

with left:
    source = st.radio(
        "Image source",
        ["Upload a chip", "Pick from the test split"],
        horizontal=True,
    )

with right:
    threshold = sh.threshold_slider(cfg, key="detect_thr")

image = None
caption = ""

if source == "Upload a chip":
    upload = st.file_uploader(
        "Satellite image chip (jpg / png / tif)",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
    )
    if upload is not None:
        image = sh.read_upload(upload)
        caption = upload.name
else:
    try:
        from src.data.preprocess import load_manifest, resolve_path

        df = load_manifest(cfg)
        test = df[df["split"] == "test"]
        classes = ["(any)"] + sorted(test["source_class"].unique().tolist())
        c1, c2 = st.columns([1, 1])
        chosen = c1.selectbox("Land-cover class", classes)
        pool = test if chosen == "(any)" else test[test["source_class"] == chosen]

        if "detect_pick" not in st.session_state or c2.button("Shuffle", **sh.STRETCH):
            st.session_state["detect_pick"] = int(np.random.randint(len(pool)))
        idx = min(st.session_state["detect_pick"], len(pool) - 1)
        row = pool.iloc[idx]
        image = load_image_rgb(resolve_path(row["path"]))
        caption = f"{row['source_class']} - ground truth: {row['label']}"
    except Exception as exc:  # noqa: BLE001
        st.info(f"Test split unavailable ({exc}). Upload a chip instead.")

if image is None:
    st.caption("Waiting for an image ...")
    st.stop()

# ----------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------
prediction = predict_images([image], cfg, model=model, meta=meta,
                            threshold=threshold, sources=[caption])[0]

sh.verdict_block(prediction, threshold)

col_img, col_cam, col_stats = st.columns([1, 1, 1])

with col_img:
    st.markdown("**Input chip**")
    st.image(sh.to_display(image), **sh.STRETCH, caption=caption or None)
    st.caption(f"{image.shape[1]} x {image.shape[0]} px, resized to "
               f"{image_size[1]} x {image_size[0]} for the network")

with col_cam:
    st.markdown("**Grad-CAM evidence**")
    with st.spinner("Computing the saliency map ..."):
        try:
            import tensorflow as tf

            resized = tf.image.resize(image[None, ...].astype("float32"),
                                      image_size, method="bilinear").numpy()[0]
            overlay, heat = explain(
                model, resized,
                alpha=float(cfg["app"]["gradcam_alpha"]),
                colormap=str(cfg["app"]["gradcam_colormap"]),
            )
            st.image(overlay, **sh.STRETCH)
            st.caption(
                f"Peak activation {heat.max():.2f}; "
                f"{100.0 * float((heat > 0.5).mean()):.0f}% of the chip is strongly salient."
            )
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Grad-CAM unavailable: {exc}")

with col_stats:
    st.markdown("**Class probabilities**")
    probs = pd.DataFrame(
        {"class": list(prediction.probabilities),
         "probability": list(prediction.probabilities.values())}
    ).sort_values("probability", ascending=False)
    st.dataframe(probs, **sh.STRETCH, hide_index=True,
                 column_config={"probability": st.column_config.ProgressColumn(
                     "probability", min_value=0.0, max_value=1.0, format="%.3f")})

    margin = abs(prediction.positive_score - threshold)
    st.metric("Distance from threshold", f"{margin:.3f}",
              help="How far the score sits from the decision boundary. "
                   "Small values mean the call is fragile.")
    if margin < 0.05:
        st.warning("This chip sits right on the boundary - treat the verdict as inconclusive.")

st.divider()
st.markdown(
    """
**Reading the heatmap.** Warm colours mark the regions whose features most increased the
landfill score. On a true positive you should see the bare working face, waste cells or
haul roads light up. If the heat sits on a car park, a bright roof or the image border,
the model is keying on a spurious cue - a good reason to distrust that particular call
and to add more of that surface to the training data.
    """
)

"""Sliding-window scan of a full satellite scene."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _shared as sh

sh.page_config("Scene Scan", icon="🗺")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from src.scan import render_overlay, scan_image  # noqa: E402
from src.utils import load_image_rgb  # noqa: E402

cfg = sh.get_config()
sh.sidebar_status(cfg)

sh.header(
    "Scene scan",
    "Sweep a full scene with an overlapping window, accumulate the per-tile scores into a "
    "probability surface, and merge the hot windows into candidate sites.",
)

model, meta = sh.require_model(cfg)
scan_cfg = cfg["scan"]

# ----------------------------------------------------------------------
# Controls
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Scan settings")
    tile_size = st.select_slider("Window size (px)", options=[64, 96, 128, 192, 256, 384],
                                 value=int(scan_cfg["tile_size"]))
    overlap = st.slider("Window overlap", 0.0, 0.75, float(scan_cfg["overlap"]), 0.05,
                        help="More overlap = smoother surface and tighter boxes, "
                             "but quadratically more tiles to classify.")
    threshold = st.slider("Detection threshold", 0.05, 0.95,
                          float(scan_cfg["min_confidence"]), 0.01)
    merge_iou = st.slider("Box merge IoU", 0.05, 0.9, float(scan_cfg["merge_iou"]), 0.05)
    mpp = st.number_input("Metres per pixel (optional)", min_value=0.0, value=10.0, step=0.5,
                          help="Sentinel-2 RGB bands are 10 m/px. Set 0 to hide areas.")
    alpha = st.slider("Overlay strength", 0.0, 1.0, 0.5, 0.05)

# ----------------------------------------------------------------------
# Scene input
# ----------------------------------------------------------------------
sample_path = cfg.data_root / "samples" / "sample_scene.png"
tabs = st.tabs(["Upload a scene", "Use the sample scene"])
scene = None
scene_name = ""

with tabs[0]:
    up = st.file_uploader("Large satellite image",
                          type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"])
    if up is not None:
        scene = sh.read_upload(up)
        scene_name = up.name

with tabs[1]:
    if sample_path.exists():
        st.caption(f"Synthetic demo scene at `{sample_path.as_posix()}`")
        if st.button("Load the sample scene", type="primary"):
            st.session_state["use_sample"] = True
        if st.session_state.get("use_sample"):
            scene = load_image_rgb(sample_path)
            scene_name = sample_path.name
    else:
        st.info("No sample scene yet. Generate one with "
                "`python scripts/make_sample_scene.py`.")

if scene is None:
    st.caption("Waiting for a scene ...")
    st.stop()

h, w = scene.shape[:2]
stride = max(1, int(round(tile_size * (1.0 - overlap))))
n_tiles_est = (max(0, (h - tile_size)) // stride + 1) * (max(0, (w - tile_size)) // stride + 1)

sh.metric_row(
    [
        ("Scene", f"{w} x {h} px"),
        ("Window", f"{tile_size} px"),
        ("Stride", f"{stride} px"),
        ("Tiles to classify", f"{n_tiles_est:,}"),
        ("Ground area",
         f"{(w * mpp) / 1000:.1f} x {(h * mpp) / 1000:.1f} km" if mpp else "-"),
    ]
)

if n_tiles_est > int(scan_cfg["max_tiles"]):
    st.error(
        f"That configuration needs about {n_tiles_est:,} windows, above the "
        f"{int(scan_cfg['max_tiles']):,} safety cap. Increase the window size or reduce "
        "the overlap."
    )
    st.stop()

if not st.button("Run scan", type="primary", **sh.STRETCH):
    st.image(sh.to_display(scene), caption=scene_name or "scene", **sh.STRETCH)
    st.stop()

# ----------------------------------------------------------------------
# Scan
# ----------------------------------------------------------------------
progress = st.progress(0.0, text="Classifying windows ...")


def _tick(done: int, total: int) -> None:
    progress.progress(done / max(total, 1), text=f"Classifying windows ... {done}/{total}")


result = scan_image(
    scene, cfg, model=model, meta=meta,
    tile_size=tile_size, overlap=overlap, threshold=threshold,
    merge_iou=merge_iou, progress=_tick,
)
progress.empty()

summary = result.summary(mpp if mpp else None)

sh.metric_row(
    [
        ("Candidate sites", f"{summary['n_detections']}"),
        ("Flagged windows", f"{summary['n_flagged_tiles']:,} / {summary['n_tiles']:,}"),
        ("Coverage flagged", f"{summary['flagged_fraction']:.1%}"),
        ("Peak score", f"{summary['max_score']:.3f}"),
        ("Mean score", f"{summary['mean_score']:.3f}"),
    ]
)

overlay = render_overlay(scene, result, alpha=alpha, draw_boxes=True)

view = st.radio("View", ["Detections overlay", "Probability surface", "Original"],
                horizontal=True)

if view == "Detections overlay":
    st.image(overlay, **sh.STRETCH,
             caption=f"{summary['n_detections']} candidate site(s) - red boxes")
elif view == "Probability surface":
    from src.models.gradcam import colorize

    st.image(colorize(np.clip(result.probability_map, 0, 1), "inferno"),
             **sh.STRETCH,
             caption="Per-pixel P(landfill candidate), averaged over overlapping windows")
else:
    st.image(sh.to_display(scene), **sh.STRETCH, caption=scene_name)

# ----------------------------------------------------------------------
# Detections table + downloads
# ----------------------------------------------------------------------
st.subheader("Candidate sites")

if summary["detections"]:
    det_df = pd.DataFrame(summary["detections"])
    det_df.insert(0, "site", [f"#{i}" for i in range(1, len(det_df) + 1)])
    st.dataframe(det_df, **sh.STRETCH, hide_index=True,
                 column_config={"confidence": st.column_config.ProgressColumn(
                     "confidence", min_value=0.0, max_value=1.0, format="%.3f")})

    st.markdown("**Crops of the top candidates**")
    top = result.detections[:6]
    cols = st.columns(min(len(top), 3))
    for i, det in enumerate(top):
        x1, y1, x2, y2 = det.box
        crop = scene[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]
        if crop.size:
            cols[i % len(cols)].image(
                sh.to_display(crop),
                caption=f"#{i + 1}  conf {det.confidence:.3f}  {det.width}x{det.height} px",
                **sh.STRETCH,
            )
else:
    st.success("No windows exceeded the detection threshold in this scene.")

c1, c2, c3 = st.columns(3)

c1.download_button(
    "Download report (JSON)",
    data=json.dumps(summary, indent=2),
    file_name=f"{Path(scene_name).stem or 'scene'}_scan.json",
    mime="application/json",
    **sh.STRETCH,
)

if summary["detections"]:
    c2.download_button(
        "Download detections (CSV)",
        data=pd.DataFrame(summary["detections"]).to_csv(index=False),
        file_name=f"{Path(scene_name).stem or 'scene'}_detections.csv",
        mime="text/csv",
        **sh.STRETCH,
    )

buf = io.BytesIO()
Image.fromarray(overlay).save(buf, format="PNG")
c3.download_button(
    "Download overlay (PNG)",
    data=buf.getvalue(),
    file_name=f"{Path(scene_name).stem or 'scene'}_overlay.png",
    mime="image/png",
    **sh.STRETCH,
)

# ----------------------------------------------------------------------
# Score distribution
# ----------------------------------------------------------------------
with st.expander("Window score distribution"):
    st.bar_chart(sh.score_histogram(result.tile_scores, bins=25, label="windows"),
                 height=260)
    st.caption(
        "A healthy scan is strongly bimodal: most windows near 0, a tight cluster near 1. "
        "A broad middle hump means the model is unsure about this scene's imagery - check "
        "that its resolution and colour balance resemble the training tiles."
    )

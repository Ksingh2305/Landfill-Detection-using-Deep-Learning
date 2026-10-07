"""Run the data / training / evaluation steps from the browser."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _shared as sh

sh.page_config("Pipeline", icon="⚙")

import streamlit as st  # noqa: E402

from src.utils import list_images  # noqa: E402

cfg = sh.get_config()
sh.sidebar_status(cfg)

sh.header(
    "Pipeline runner",
    "Kick off the same scripts you would run from a terminal, and watch their output live. "
    "Everything writes into the project folder, so results survive a browser refresh.",
)

ROOT = sh.PROJECT_ROOT
SCRIPTS = ROOT / "scripts"
PY = sys.executable


def run_streaming(cmd: list, title: str) -> int:
    """Run a command, streaming stdout into an auto-scrolling code block."""
    st.markdown(f"**{title}**")
    st.code(" ".join(str(c) for c in cmd), language="bash")
    box = st.empty()
    lines: list = []

    process = subprocess.Popen(
        [str(c) for c in cmd],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line.rstrip())
        box.code("\n".join(lines[-40:]), language="text")
    code = process.wait()

    if code == 0:
        st.success(f"{title} finished successfully.")
    else:
        st.error(f"{title} exited with code {code}. Full output above.")
    with st.expander("Full log"):
        st.code("\n".join(lines), language="text")
    return code


# ----------------------------------------------------------------------
# Current state
# ----------------------------------------------------------------------
manifest_ok = cfg.manifest_path.exists()
model_ok = cfg.model_path.exists()
report_ok = (cfg.reports_dir / "metrics.json").exists()
n_raw = len(list_images(cfg.raw_dir))

sh.metric_row(
    [
        ("Raw images", f"{n_raw:,}"),
        ("Manifest", "ready" if manifest_ok else "missing"),
        ("Model", "trained" if model_ok else "missing"),
        ("Evaluation", "ready" if report_ok else "missing"),
        ("Python", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"),
    ]
)

st.caption(
    "Training runs in this Streamlit process, so leave the tab open until it finishes. "
    "For long runs a terminal is more comfortable - the commands are shown above each log."
)
st.divider()

# ----------------------------------------------------------------------
# Step 1
# ----------------------------------------------------------------------
st.subheader("1 - Data")

c1, c2, c3 = st.columns([1, 1, 1])
use_synthetic = c1.toggle("Offline synthetic dataset", value=False,
                          help="Skip the EuroSAT download and generate stand-in tiles locally.")
max_per_class = c2.number_input("Max images per land-cover class (0 = all)",
                                min_value=0, max_value=5000, value=0, step=100)
per_class_syn = c3.number_input("Synthetic tiles per class", min_value=50, max_value=2000,
                                value=400, step=50, disabled=not use_synthetic)

if st.button("Run step 1 - download and build the manifest", type="primary",
             **sh.STRETCH):
    cmd = [PY, SCRIPTS / "01_download_data.py", "--allow-synthetic-fallback"]
    if use_synthetic:
        cmd += ["--synthetic", "--synthetic-per-class", str(int(per_class_syn))]
    if max_per_class:
        cmd += ["--max-per-class", str(int(max_per_class))]
    run_streaming(cmd, "Step 1 - data acquisition")
    st.cache_data.clear()

st.divider()

# ----------------------------------------------------------------------
# Step 2
# ----------------------------------------------------------------------
st.subheader("2 - Train")

c1, c2, c3 = st.columns(3)
backbone = c1.selectbox("Backbone", ["mobilenetv2", "efficientnetb0", "resnet50v2"],
                        index=["mobilenetv2", "efficientnetb0", "resnet50v2"].index(
                            str(cfg["model"]["backbone"])))
epochs_head = c2.number_input("Warm-up epochs", 0, 60, int(cfg["train"]["epochs_head"]))
epochs_ft = c3.number_input("Fine-tune epochs", 0, 60, int(cfg["train"]["epochs_finetune"]))
batch = st.select_slider("Batch size", options=[8, 16, 32, 64, 128],
                         value=int(cfg["data"]["batch_size"]))

if not manifest_ok:
    st.info("Run step 1 first - there is no manifest to train on.")

if st.button("Run step 2 - train the detector", type="primary",
             **sh.STRETCH, disabled=not manifest_ok):
    cmd = [
        PY, SCRIPTS / "02_train.py",
        "--backbone", backbone,
        "--epochs-head", str(int(epochs_head)),
        "--epochs-finetune", str(int(epochs_ft)),
        "--batch-size", str(int(batch)),
    ]
    run_streaming(cmd, "Step 2 - training")
    st.cache_resource.clear()
    st.cache_data.clear()

st.divider()

# ----------------------------------------------------------------------
# Step 3
# ----------------------------------------------------------------------
st.subheader("3 - Evaluate")

split = st.selectbox("Split", ["test", "val", "train"], index=0)
if st.button("Run step 3 - evaluate and build the report", type="primary",
             **sh.STRETCH, disabled=not model_ok):
    run_streaming([PY, SCRIPTS / "03_evaluate.py", "--split", split],
                  "Step 3 - evaluation")
    st.cache_data.clear()

st.divider()

# ----------------------------------------------------------------------
# Extras
# ----------------------------------------------------------------------
st.subheader("Extras")

c1, c2 = st.columns(2)

with c1:
    w = st.number_input("Sample scene width", 256, 4096, 1280, 128)
    h = st.number_input("Sample scene height", 256, 4096, 960, 128)
    if st.button("Generate a demo scene for the Scene Scan page", **sh.STRETCH):
        run_streaming(
            [PY, SCRIPTS / "make_sample_scene.py", "--width", str(int(w)),
             "--height", str(int(h))],
            "Sample scene",
        )

with c2:
    quick = st.toggle("Quick mode (small dataset, 2+2 epochs)", value=True)
    offline = st.toggle("Offline (synthetic data)", value=False, key="pipe_offline")
    if st.button("Run the whole pipeline end to end", **sh.STRETCH):
        cmd = [PY, SCRIPTS / "run_pipeline.py"]
        if quick:
            cmd.append("--quick")
        if offline:
            cmd.append("--synthetic")
        run_streaming(cmd, "Full pipeline")
        st.cache_resource.clear()
        st.cache_data.clear()

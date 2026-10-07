"""Shared plumbing for every Streamlit page: paths, caching, styling, widgets."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

# --- make the project importable no matter how streamlit was launched ---
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
for _p in (str(PROJECT_ROOT), str(APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import Config, load_config  # noqa: E402
from src.utils import load_json  # noqa: E402

ASSETS = APP_DIR / "assets"


def _streamlit_version() -> tuple:
    import re

    parts = re.findall(r"\d+", getattr(st, "__version__", "0.0"))[:2]
    return tuple(int(p) for p in parts) if parts else (0, 0)


# Streamlit 1.49 replaced ``use_container_width=True`` with ``width="stretch"``
# and deprecated the old name. Splat this into any call that should fill its
# column, and the app stays warning-free on both sides of that change.
STRETCH = ({"width": "stretch"} if _streamlit_version() >= (1, 49)
           else {"use_container_width": True})


# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
def page_config(title: str, icon: str = "S", layout: str = "wide") -> None:
    st.set_page_config(page_title=f"{title} - Landfill Detection",
                       page_icon=icon, layout=layout,
                       initial_sidebar_state="expanded")
    inject_css()


@st.cache_data(show_spinner=False)
def _read_css() -> str:
    path = ASSETS / "style.css"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def inject_css() -> None:
    css = _read_css()
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="page-header">
          <div class="page-title">{title}</div>
          <div class="page-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# Cached resources
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_config_dict() -> dict:
    return load_config(use_cache=False).raw


def get_config() -> Config:
    """Fresh Config wrapper around the cached YAML (cheap, always current)."""
    return Config(raw=get_config_dict(), path=PROJECT_ROOT / "config.yaml")


@st.cache_resource(show_spinner="Loading the detector ...")
def _load_model(model_path: str, mtime: float):
    from src.predict import load_detector

    cfg = get_config()
    return load_detector(cfg, model_path=Path(model_path))


def get_model(cfg: Optional[Config] = None):
    """``(model, meta)`` or ``(None, None)`` when nothing has been trained yet."""
    cfg = cfg or get_config()
    path = cfg.model_path
    if not path.exists():
        return None, None
    return _load_model(str(path), path.stat().st_mtime)


def model_meta(cfg: Optional[Config] = None) -> dict:
    cfg = cfg or get_config()
    return load_json(cfg.model_meta_path, default={}) or {}


def metrics_report(cfg: Optional[Config] = None) -> dict:
    cfg = cfg or get_config()
    return load_json(cfg.reports_dir / "metrics.json", default={}) or {}


# ----------------------------------------------------------------------
# Common UI pieces
# ----------------------------------------------------------------------
def require_model(cfg: Optional[Config] = None):
    """Return ``(model, meta)`` or render an explanatory block and stop the page."""
    cfg = cfg or get_config()
    model, meta = get_model(cfg)
    if model is None:
        st.warning("No trained model yet.")
        st.markdown(
            """
Run the pipeline once from a terminal in the project folder:

```bash
python scripts/01_download_data.py
python scripts/02_train.py
python scripts/03_evaluate.py
```

Short on time? `python scripts/run_pipeline.py --quick` trains a smaller model in
a few minutes. No internet? Add `--synthetic`.

The **Pipeline** page in the sidebar can run these steps for you.
            """
        )
        st.stop()
    return model, meta


def sidebar_status(cfg: Optional[Config] = None) -> None:
    cfg = cfg or get_config()
    meta = model_meta(cfg)
    trained = cfg.model_path.exists()

    with st.sidebar:
        st.markdown("### Detector status")
        if trained:
            st.markdown('<span class="pill pill-ok">model loaded</span>',
                        unsafe_allow_html=True)
            st.caption(
                f"backbone: **{meta.get('backbone', '?')}**  \n"
                f"input: {meta.get('image_size', cfg.image_size)}  \n"
                f"data: {meta.get('data_source', cfg['data']['source'])}  \n"
                f"trained: {meta.get('created_utc', 'unknown')}"
            )
            best = meta.get("best_val_metrics") or {}
            if best.get("val_auc"):
                st.caption(f"best val AUC: **{best['val_auc']:.4f}**")
        else:
            st.markdown('<span class="pill pill-bad">not trained</span>',
                        unsafe_allow_html=True)
            st.caption("Run `python scripts/run_pipeline.py --quick`")

        st.divider()
        st.caption(
            "Positive class is a **proxy** for waste-site-like surfaces "
            "(see the About page). Treat every flag as a lead to verify, "
            "not a confirmed landfill."
        )


def threshold_slider(cfg: Config, key: str = "thr") -> float:
    meta = model_meta(cfg)
    default = float(meta.get("threshold", cfg.threshold))
    return st.slider(
        "Decision threshold on P(landfill candidate)",
        min_value=0.05, max_value=0.95, value=float(default), step=0.01, key=key,
        help="Lower = catch more sites but more false alarms. "
             "The Model Performance page shows the precision/recall trade-off.",
    )


def verdict_block(prediction, threshold: float) -> None:
    """Big coloured verdict card for a single prediction."""
    score = prediction.positive_score
    if score >= threshold:
        cls, text = "verdict-hit", "LANDFILL CANDIDATE"
    elif score >= threshold * 0.7:
        cls, text = "verdict-warn", "BORDERLINE"
    else:
        cls, text = "verdict-clean", "NO CANDIDATE DETECTED"

    st.markdown(
        f"""
        <div class="verdict {cls}">
          <div class="verdict-text">{text}</div>
          <div class="verdict-score">P(landfill) = {score:.3f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_row(items) -> None:
    """``items`` is a sequence of ``(label, value, help)`` triples."""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        label, value = item[0], item[1]
        helptext = item[2] if len(item) > 2 else None
        col.metric(label, value, help=helptext)


def score_histogram(values, bins: int = 25, label: str = "images"):
    """Histogram of scores in ``[0, 1]``, indexed by numeric bin centre.

    ``pandas.value_counts(bins=...)`` yields an Interval index that Altair
    cannot type, so we build the bins explicitly and label them with their
    midpoints - which also plots on a real numeric axis.
    """
    import pandas as pd

    v = np.asarray(list(values), dtype="float32")
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    counts, _ = np.histogram(v, bins=edges)
    centres = np.round((edges[:-1] + edges[1:]) / 2.0, 3)
    return pd.DataFrame({label: counts}, index=pd.Index(centres, name="score"))


def to_display(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype("uint8")
    return a


def read_upload(file, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Decode a Streamlit UploadedFile into an RGB uint8 array."""
    from src.utils import load_image_rgb

    file.seek(0)
    return load_image_rgb(file, target_size)


__all__ = [
    "PROJECT_ROOT",
    "APP_DIR",
    "page_config",
    "header",
    "inject_css",
    "get_config",
    "get_model",
    "model_meta",
    "metrics_report",
    "require_model",
    "sidebar_status",
    "threshold_slider",
    "verdict_block",
    "metric_row",
    "to_display",
    "read_upload",
]

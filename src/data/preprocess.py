"""Scalable preprocessing: source imagery -> labelled, split manifest.

Rather than copying tens of thousands of files into ``train/val/test``
folders (slow, and it triples disk usage), the pipeline builds a single
**manifest** - one CSV row per image with its resolved label and split
assignment. Everything downstream (``tf.data``, training, evaluation,
the web app) reads that manifest, so a split is reproducible, auditable
and cheap to regenerate.

Steps
-----
1. Discover ``<root>/<SourceClass>/*.jpg``.
2. Map each source class onto a task label via ``config.yaml``
   (``task.positive_classes`` / ``task.negative_classes``).
3. Optionally cap images per source class (fast smoke runs).
4. Stratified, seeded split into train / val / test - stratified on the
   *source* class, not just the binary label, so each split keeps the
   same land-cover mix.
5. Write ``data/interim/manifest.csv`` + ``dataset_stats.json``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import PROJECT_ROOT, Config, load_config
from ..utils import LOGGER, list_images, save_json, summarise_counts

MANIFEST_COLUMNS = ["path", "source_class", "label", "label_idx", "split"]


class PreprocessError(RuntimeError):
    """The source imagery cannot be turned into a usable manifest."""


# ----------------------------------------------------------------------
# Discovery + label mapping
# ----------------------------------------------------------------------
def discover_images(root: Path) -> Dict[str, List[Path]]:
    """Return ``{source_class: [image paths]}`` for ``<root>/<class>/*``."""
    root = Path(root)
    if not root.exists():
        raise PreprocessError(f"Image root does not exist: {root}")

    found: Dict[str, List[Path]] = {}
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        images = list_images(class_dir)
        if images:
            found[class_dir.name] = images

    if not found:
        raise PreprocessError(
            f"No class sub-folders with images under {root}. Expected "
            f"{root}/<ClassName>/*.jpg"
        )
    return found


def build_label_map(cfg: Config, source_classes: List[str]) -> Dict[str, str]:
    """Map every source class onto a task label (or drop it)."""
    task = cfg["task"]
    class_names = cfg.class_names

    if cfg.is_binary:
        pos = set(task["positive_classes"])
        neg = set(task["negative_classes"])
        mapping = {}
        for sc in source_classes:
            if sc in pos:
                mapping[sc] = task["positive_label"]
            elif sc in neg:
                mapping[sc] = task["negative_label"]
            # anything else is intentionally dropped
        if not mapping:
            raise PreprocessError(
                "None of the discovered class folders "
                f"{source_classes} appear in task.positive_classes/negative_classes. "
                "If you are using your own images, set data.source: custom and list "
                "your folder names in config.yaml."
            )
        return mapping

    # multiclass: keep every configured class as its own label
    return {sc: sc for sc in source_classes if sc in class_names}


def _label_indices(cfg: Config) -> Dict[str, int]:
    return {name: i for i, name in enumerate(cfg.class_names)}


# ----------------------------------------------------------------------
# Splitting
# ----------------------------------------------------------------------
def stratified_split(
    n: int, fractions: Dict[str, float], rng: np.random.Generator
) -> List[str]:
    """Assign ``n`` items to splits, honouring the fractions as closely as possible."""
    order = ["train", "val", "test"]
    counts = {}
    assigned = 0
    for name in order[:-1]:
        counts[name] = int(round(n * float(fractions[name])))
        assigned += counts[name]
    counts[order[-1]] = max(0, n - assigned)

    # Guarantee at least one item in val/test when the class is tiny.
    if n >= 3:
        for name in ("val", "test"):
            if counts[name] == 0:
                counts[name] = 1
                counts["train"] = max(0, counts["train"] - 1)

    labels = []
    for name in order:
        labels.extend([name] * counts[name])
    labels = labels[:n] + ["train"] * max(0, n - len(labels))
    rng.shuffle(labels)
    return labels


# ----------------------------------------------------------------------
# Manifest
# ----------------------------------------------------------------------
def build_manifest(
    cfg: Optional[Config] = None,
    image_root: Optional[Path] = None,
    save: bool = True,
) -> pd.DataFrame:
    """Discover, label, split and (optionally) persist the dataset manifest."""
    cfg = cfg or load_config()
    cfg.ensure_dirs()

    root = Path(image_root) if image_root else _default_root(cfg)
    LOGGER.info("Building manifest from: %s", root)

    per_class = discover_images(root)
    label_map = build_label_map(cfg, list(per_class))
    idx_of = _label_indices(cfg)
    rng = np.random.default_rng(cfg.seed)

    cap = int(cfg["data"].get("max_per_class", 0) or 0)
    rows = []

    for source_class, paths in per_class.items():
        label = label_map.get(source_class)
        if label is None:
            LOGGER.info("  skipping unmapped class folder: %s (%d images)",
                        source_class, len(paths))
            continue

        paths = list(paths)
        rng.shuffle(paths)
        if cap > 0:
            paths = paths[:cap]

        splits = stratified_split(len(paths), cfg["data"]["splits"], rng)
        for p, split in zip(paths, splits):
            try:
                rel = p.resolve().relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                rel = p.resolve().as_posix()  # dataset lives outside the project
            rows.append(
                {
                    "path": rel,
                    "source_class": source_class,
                    "label": label,
                    "label_idx": idx_of[label],
                    "split": split,
                }
            )

    if not rows:
        raise PreprocessError("Manifest is empty - no images survived label mapping.")

    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    df = df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)

    _log_manifest(df)
    if save:
        out = cfg.manifest_path
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        LOGGER.info("Manifest written: %s (%d rows)", out, len(df))
        save_json(dataset_stats(cfg, df), cfg.interim_dir / "dataset_stats.json")
    return df


def _default_root(cfg: Config) -> Path:
    """Pick the source folder implied by ``data.source``, with fallbacks."""
    source = str(cfg["data"]["source"]).lower()
    if source == "custom":
        return cfg.custom_dir
    if source == "synthetic":
        from .synthetic import SYNTHETIC_DIR_NAME

        return cfg.raw_dir / SYNTHETIC_DIR_NAME

    from .download import DATASET_DIR_NAME
    from .synthetic import SYNTHETIC_DIR_NAME

    eurosat = cfg.raw_dir / DATASET_DIR_NAME
    if eurosat.exists():
        return eurosat
    synthetic = cfg.raw_dir / SYNTHETIC_DIR_NAME
    if synthetic.exists():
        LOGGER.warning("EuroSAT not found - falling back to the synthetic dataset.")
        return synthetic
    raise PreprocessError(
        f"No dataset found. Expected {eurosat}.\n"
        "Run:  python scripts/01_download_data.py"
    )


def _log_manifest(df: pd.DataFrame) -> None:
    LOGGER.info("Manifest summary: %d images", len(df))
    LOGGER.info("  by split : %s", summarise_counts(Counter(df["split"])))
    LOGGER.info("  by label : %s", summarise_counts(Counter(df["label"])))
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        if len(sub):
            LOGGER.info("    %-5s -> %s", split, summarise_counts(Counter(sub["label"])))


def load_manifest(cfg: Optional[Config] = None) -> pd.DataFrame:
    """Read the manifest, or raise a helpful error explaining how to make one."""
    cfg = cfg or load_config()
    path = cfg.manifest_path
    if not path.exists():
        raise PreprocessError(
            f"Manifest not found at {path}.\n"
            "Run:  python scripts/01_download_data.py   (downloads + builds the manifest)"
        )
    df = pd.read_csv(path)
    missing = set(MANIFEST_COLUMNS) - set(df.columns)
    if missing:
        raise PreprocessError(f"Manifest {path} is missing columns: {sorted(missing)}")
    return df


def resolve_path(rel_or_abs: str) -> Path:
    """Manifest paths are project-relative; turn one back into an absolute Path."""
    p = Path(rel_or_abs)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


# ----------------------------------------------------------------------
# Statistics / class weights
# ----------------------------------------------------------------------
def dataset_stats(cfg: Config, df: Optional[pd.DataFrame] = None) -> dict:
    """Counts and imbalance ratios, used by the report and the app."""
    df = load_manifest(cfg) if df is None else df
    by_split_label: Dict[str, Dict[str, int]] = defaultdict(dict)
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        by_split_label[split] = {k: int(v) for k, v in Counter(sub["label"]).items()}

    label_counts = {k: int(v) for k, v in Counter(df["label"]).items()}
    source_counts = {k: int(v) for k, v in Counter(df["source_class"]).items()}
    biggest = max(label_counts.values()) if label_counts else 0
    smallest = min(label_counts.values()) if label_counts else 0

    return {
        "total_images": int(len(df)),
        "class_names": cfg.class_names,
        "label_counts": label_counts,
        "source_class_counts": source_counts,
        "split_counts": {k: int(v) for k, v in Counter(df["split"]).items()},
        "by_split_label": dict(by_split_label),
        "imbalance_ratio": round(biggest / smallest, 3) if smallest else None,
        "task_mode": cfg["task"]["mode"],
        "image_size": list(cfg.image_size),
    }


def compute_class_weights(cfg: Config, df: Optional[pd.DataFrame] = None) -> Dict[int, float]:
    """Inverse-frequency weights for the *training* split (Keras format)."""
    df = load_manifest(cfg) if df is None else df
    train = df[df["split"] == "train"]
    counts = Counter(train["label_idx"])
    n_classes = cfg.num_classes
    total = sum(counts.values())
    weights = {}
    for i in range(n_classes):
        c = counts.get(i, 0)
        weights[i] = float(total / (n_classes * c)) if c else 1.0
    LOGGER.info("Class weights: %s",
                {cfg.class_names[i]: round(w, 3) for i, w in weights.items()})
    return weights


__all__ = [
    "PreprocessError",
    "MANIFEST_COLUMNS",
    "discover_images",
    "build_label_map",
    "build_manifest",
    "load_manifest",
    "resolve_path",
    "dataset_stats",
    "compute_class_weights",
    "stratified_split",
]

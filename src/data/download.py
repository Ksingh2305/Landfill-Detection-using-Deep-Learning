"""Acquire the source satellite imagery.

Primary source is **EuroSAT** (Helber et al., 2019) - 27,000 labelled
64x64 RGB Sentinel-2 tiles across 10 land-use classes. We try a list of
mirrors in order; if every mirror fails (no internet, corporate proxy,
mirror down) the caller can fall back to the offline synthetic generator
in :mod:`src.data.synthetic` so the rest of the pipeline still runs.

Layout produced::

    data/raw/EuroSAT/
        AnnualCrop/*.jpg
        Forest/*.jpg
        ...
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import List, Optional

import requests

from ..config import Config, load_config
from ..utils import LOGGER, human_bytes, list_images

EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

DATASET_DIR_NAME = "EuroSAT"


class DownloadError(RuntimeError):
    """Every configured mirror failed."""


# ----------------------------------------------------------------------
# Download
# ----------------------------------------------------------------------
def _stream_download(url: str, dest: Path, min_bytes: int, timeout: int = 60) -> Path:
    """Download ``url`` to ``dest`` with a progress log. Raises on failure."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    LOGGER.info("Downloading %s", url)

    headers = {"User-Agent": "landfill-detection/1.0 (+educational research)"}
    with requests.get(url, stream=True, timeout=timeout, headers=headers) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length") or 0)
        written = 0
        next_report = 5 * 1024 * 1024  # log every 5 MB

        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                if not chunk:
                    continue
                fh.write(chunk)
                written += len(chunk)
                if written >= next_report:
                    pct = f" ({100.0 * written / total:.0f}%)" if total else ""
                    LOGGER.info("  ... %s%s", human_bytes(written), pct)
                    next_report += 5 * 1024 * 1024

    if written < min_bytes:
        tmp.unlink(missing_ok=True)
        raise DownloadError(
            f"{url} returned only {human_bytes(written)} "
            f"(expected at least {human_bytes(min_bytes)}) - probably an error page."
        )

    if not zipfile.is_zipfile(tmp):
        tmp.unlink(missing_ok=True)
        raise DownloadError(f"{url} did not return a valid zip archive.")

    tmp.replace(dest)
    LOGGER.info("Saved archive: %s (%s)", dest, human_bytes(dest.stat().st_size))
    return dest


def download_archive(cfg: Config, force: bool = False) -> Path:
    """Fetch the EuroSAT zip, trying each configured mirror in turn."""
    dl_cfg = cfg["data"]["download"]
    archive = cfg.raw_dir / dl_cfg["archive_name"]
    archive.parent.mkdir(parents=True, exist_ok=True)

    if archive.exists() and not force and zipfile.is_zipfile(archive):
        LOGGER.info("Archive already present, skipping download: %s", archive)
        return archive

    errors: List[str] = []
    for url in dl_cfg["mirrors"]:
        try:
            return _stream_download(url, archive, int(dl_cfg["min_bytes"]))
        except Exception as exc:  # noqa: BLE001 - we genuinely want to try the next mirror
            LOGGER.warning("Mirror failed (%s): %s", url.split("/")[2], exc)
            errors.append(f"{url} -> {exc}")

    raise DownloadError(
        "All EuroSAT mirrors failed.\n  "
        + "\n  ".join(errors)
        + "\n\nOptions:\n"
        "  1. Re-run with --synthetic to generate an offline stand-in dataset.\n"
        "  2. Download EuroSAT_RGB.zip manually in a browser and place it at\n"
        f"     {archive}\n     then re-run this script."
    )


# ----------------------------------------------------------------------
# Extract
# ----------------------------------------------------------------------
def _find_class_root(extract_root: Path) -> Optional[Path]:
    """Locate the directory that directly contains the class sub-folders.

    Different EuroSAT mirrors nest things differently (``2750/``,
    ``EuroSAT_RGB/``, or the classes at the top level), so we search.
    """
    candidates = [extract_root, *(p for p in extract_root.rglob("*") if p.is_dir())]
    for cand in candidates:
        subdirs = {p.name for p in cand.iterdir() if p.is_dir()} if cand.is_dir() else set()
        hits = subdirs & set(EUROSAT_CLASSES)
        if len(hits) >= 8:  # tolerate a mirror missing a class or two
            return cand
    return None


def extract_archive(cfg: Config, archive: Path, force: bool = False) -> Path:
    """Unzip and normalise the layout to ``data/raw/EuroSAT/<Class>/*.jpg``."""
    target = cfg.raw_dir / DATASET_DIR_NAME

    if target.exists() and not force and len(list_images(target)) > 1000:
        LOGGER.info("Dataset already extracted: %s", target)
        return target

    staging = cfg.raw_dir / "_extract_tmp"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Extracting %s ...", archive.name)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(staging)

    class_root = _find_class_root(staging)
    if class_root is None:
        shutil.rmtree(staging, ignore_errors=True)
        raise DownloadError(
            "Extracted archive does not look like EuroSAT - no class folders found. "
            f"Inspect {staging} manually."
        )

    if target.exists():
        shutil.rmtree(target)
    LOGGER.info("Normalising layout -> %s", target)
    shutil.move(str(class_root), str(target))
    shutil.rmtree(staging, ignore_errors=True)

    n = len(list_images(target))
    LOGGER.info("Extracted %d images across %d classes",
                n, sum(1 for p in target.iterdir() if p.is_dir()))
    if n < 100:
        raise DownloadError(f"Only {n} images found under {target}; extraction looks wrong.")
    return target


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------
def ensure_dataset(
    cfg: Optional[Config] = None,
    force: bool = False,
    synthetic: bool = False,
    synthetic_per_class: int = 400,
) -> Path:
    """Guarantee that a usable labelled image folder exists; return its path.

    Falls back to the offline synthetic generator when ``synthetic=True`` or
    when every download mirror fails and the caller allows it.
    """
    cfg = cfg or load_config()
    cfg.ensure_dirs()

    if synthetic or str(cfg["data"]["source"]).lower() == "synthetic":
        from .synthetic import generate_synthetic_dataset

        return generate_synthetic_dataset(cfg, per_class=synthetic_per_class, force=force)

    if str(cfg["data"]["source"]).lower() == "custom":
        custom = cfg.custom_dir
        n = len(list_images(custom))
        if n == 0:
            raise DownloadError(
                f"data.source is 'custom' but no images were found in {custom}.\n"
                "Create one sub-folder per class, e.g.\n"
                f"  {custom / 'landfill'}\\image001.jpg\n"
                f"  {custom / 'not_landfill'}\\image002.jpg"
            )
        LOGGER.info("Using custom dataset: %d images in %s", n, custom)
        return custom

    archive = download_archive(cfg, force=force)
    return extract_archive(cfg, archive, force=force)


__all__ = [
    "EUROSAT_CLASSES",
    "DownloadError",
    "download_archive",
    "extract_archive",
    "ensure_dataset",
]

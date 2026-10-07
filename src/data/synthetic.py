"""Offline procedural stand-in for EuroSAT.

This module exists so the project is *always* runnable: no internet, no
proxy exceptions, no dead mirrors. It renders 64x64 RGB tiles that mimic
the coarse colour/texture statistics of Sentinel-2 land-use classes using
fractal value noise plus per-class structure (field parcels, roof grids,
road ribbons, waste-cell mounds).

It is a **development and testing aid**, not science. Metrics obtained on
synthetic tiles say nothing about real-world landfill detection - the
scripts label every synthetic run clearly so results are never confused
with EuroSAT results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

from ..config import Config, load_config
from ..utils import LOGGER, list_images

SYNTHETIC_DIR_NAME = "SyntheticEuroSAT"
TILE = 64

# Approximate mean RGB of each class in true-colour Sentinel-2 imagery.
PALETTES: Dict[str, Dict[str, Tuple[float, float, float]]] = {
    "Forest":               {"base": (34, 68, 34),    "accent": (18, 44, 20)},
    "Pasture":              {"base": (108, 142, 66),  "accent": (128, 160, 80)},
    "HerbaceousVegetation": {"base": (120, 138, 70),  "accent": (150, 158, 88)},
    "AnnualCrop":           {"base": (146, 148, 82),  "accent": (176, 168, 104)},
    "PermanentCrop":        {"base": (110, 124, 66),  "accent": (138, 140, 84)},
    "River":                {"base": (96, 118, 92),   "accent": (48, 74, 88)},
    "SeaLake":              {"base": (40, 70, 96),    "accent": (30, 58, 84)},
    "Residential":          {"base": (128, 122, 112), "accent": (176, 150, 132)},
    "Industrial":           {"base": (138, 132, 124), "accent": (196, 192, 186)},
    "Highway":              {"base": (120, 118, 110), "accent": (168, 168, 164)},
}

ALL_CLASSES = list(PALETTES.keys())


# ----------------------------------------------------------------------
# Noise
# ----------------------------------------------------------------------
def _value_noise(size: int, cells: int, rng: np.random.Generator) -> np.ndarray:
    """Smooth value noise in [0, 1] built from a ``cells x cells`` lattice."""
    lattice = rng.random((cells + 1, cells + 1)).astype(np.float32)
    img = Image.fromarray((lattice * 255).astype(np.uint8)).resize(
        (size, size), Image.BICUBIC
    )
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def _fractal_noise(size: int, rng: np.random.Generator, octaves: int = 4,
                   persistence: float = 0.55) -> np.ndarray:
    """Sum of octaves of value noise, normalised to [0, 1]."""
    total = np.zeros((size, size), dtype=np.float32)
    amplitude, norm, cells = 1.0, 0.0, 2
    for _ in range(octaves):
        total += amplitude * _value_noise(size, cells, rng)
        norm += amplitude
        amplitude *= persistence
        cells *= 2
    total /= max(norm, 1e-6)
    lo, hi = float(total.min()), float(total.max())
    return (total - lo) / max(hi - lo, 1e-6)


def _colourise(noise: np.ndarray, base: Tuple[float, float, float],
               accent: Tuple[float, float, float]) -> np.ndarray:
    """Blend two colours according to a scalar noise field."""
    n = noise[..., None]
    base_arr = np.array(base, dtype=np.float32)
    accent_arr = np.array(accent, dtype=np.float32)
    return base_arr * (1.0 - n) + accent_arr * n


# ----------------------------------------------------------------------
# Per-class structure
# ----------------------------------------------------------------------
def _add_parcels(img: np.ndarray, rng: np.random.Generator, n: int = 4) -> np.ndarray:
    """Rectangular agricultural parcels with slightly different tints."""
    out = img.copy()
    for _ in range(n):
        h = rng.integers(10, 34)
        w = rng.integers(10, 34)
        y = rng.integers(0, max(1, TILE - h))
        x = rng.integers(0, max(1, TILE - w))
        shift = rng.normal(0, 18, size=3).astype(np.float32)
        out[y : y + h, x : x + w] += shift
    return out


def _add_roof_grid(img: np.ndarray, rng: np.random.Generator, density: float = 0.5) -> np.ndarray:
    """Small bright rectangles on a loose grid - suburban roofs."""
    out = img.copy()
    step = int(rng.integers(7, 11))
    for y in range(2, TILE - 4, step):
        for x in range(2, TILE - 4, step):
            if rng.random() > density:
                continue
            h = int(rng.integers(3, step - 1))
            w = int(rng.integers(3, step - 1))
            tint = np.array([rng.uniform(150, 210), rng.uniform(110, 150),
                             rng.uniform(95, 135)], dtype=np.float32)
            out[y : y + h, x : x + w] = tint
    return out


def _add_large_blocks(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Big flat high-albedo blocks - warehouse roofs / industrial pads."""
    out = img.copy()
    for _ in range(int(rng.integers(2, 5))):
        h = int(rng.integers(12, 30))
        w = int(rng.integers(14, 34))
        y = int(rng.integers(0, max(1, TILE - h)))
        x = int(rng.integers(0, max(1, TILE - w)))
        v = rng.uniform(160, 225)
        out[y : y + h, x : x + w] = np.array([v, v * 0.99, v * 0.96], dtype=np.float32)
        out[y : y + h, x : x + 1] *= 0.75  # edge shadow
    return out


def _add_waste_cells(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Irregular bare-earth mounds and working-face terraces (landfill look)."""
    out = img.copy()
    yy, xx = np.mgrid[0:TILE, 0:TILE].astype(np.float32)
    for _ in range(int(rng.integers(2, 5))):
        cy, cx = rng.uniform(10, 54), rng.uniform(10, 54)
        ry, rx = rng.uniform(8, 22), rng.uniform(8, 22)
        rot = rng.uniform(0, np.pi)
        dy, dx = yy - cy, xx - cx
        u = dx * np.cos(rot) + dy * np.sin(rot)
        v = -dx * np.sin(rot) + dy * np.cos(rot)
        mask = ((u / rx) ** 2 + (v / ry) ** 2) < 1.0
        earth = np.array([rng.uniform(120, 165), rng.uniform(100, 135),
                          rng.uniform(80, 110)], dtype=np.float32)
        out[mask] = earth
        # terrace edge highlight
        edge = (((u / rx) ** 2 + (v / ry) ** 2) < 1.0) & (((u / rx) ** 2 + (v / ry) ** 2) > 0.75)
        out[edge] *= 1.18
    return out


def _add_ribbon(img: np.ndarray, rng: np.random.Generator, width: int = 5,
                colour: Tuple[float, float, float] = (170, 170, 168),
                sinuous: bool = False) -> np.ndarray:
    """A road (straight) or river (sinuous) crossing the tile."""
    out = img.copy()
    yy, xx = np.mgrid[0:TILE, 0:TILE].astype(np.float32)
    angle = rng.uniform(0, np.pi)
    offset = rng.uniform(-18, 18)
    d = (xx - TILE / 2) * np.cos(angle) + (yy - TILE / 2) * np.sin(angle) - offset
    if sinuous:
        t = (xx - TILE / 2) * -np.sin(angle) + (yy - TILE / 2) * np.cos(angle)
        d = d + rng.uniform(4, 9) * np.sin(t / rng.uniform(7, 14))
    half = width / 2.0
    mask = np.abs(d) < half
    out[mask] = np.array(colour, dtype=np.float32)
    if not sinuous:  # lane markings
        dashes = (np.abs(d) < 0.8) & ((xx.astype(int) + yy.astype(int)) % 9 < 4)
        out[dashes] = np.array([226, 226, 220], dtype=np.float32)
    return out


def make_tile(class_name: str, rng: np.random.Generator) -> np.ndarray:
    """Render one 64x64 uint8 RGB tile for ``class_name``."""
    pal = PALETTES.get(class_name, PALETTES["Pasture"])
    octaves = {"Forest": 5, "SeaLake": 2, "Industrial": 3}.get(class_name, 4)
    noise = _fractal_noise(TILE, rng, octaves=octaves)
    img = _colourise(noise, pal["base"], pal["accent"])

    if class_name in ("AnnualCrop", "PermanentCrop"):
        img = _add_parcels(img, rng, n=int(rng.integers(3, 7)))
    elif class_name == "Residential":
        img = _add_roof_grid(img, rng, density=0.55)
        img = _add_ribbon(img, rng, width=3, colour=(120, 118, 114))
    elif class_name == "Industrial":
        img = _add_large_blocks(img, rng)
        img = _add_waste_cells(img, rng)
    elif class_name == "Highway":
        img = _add_ribbon(img, rng, width=int(rng.integers(4, 8)))
    elif class_name == "River":
        img = _add_ribbon(img, rng, width=int(rng.integers(5, 11)),
                          colour=(52, 78, 92), sinuous=True)

    img += rng.normal(0, 6.0, size=img.shape).astype(np.float32)  # sensor noise
    img *= rng.uniform(0.9, 1.1)                                   # illumination
    return np.clip(img, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------
# Dataset generation
# ----------------------------------------------------------------------
def generate_synthetic_dataset(
    cfg: Optional[Config] = None,
    per_class: int = 400,
    force: bool = False,
    out_dir: Optional[Path] = None,
) -> Path:
    """Write ``per_class`` tiles for each of the 10 class folders."""
    cfg = cfg or load_config()
    target = Path(out_dir) if out_dir else (cfg.raw_dir / SYNTHETIC_DIR_NAME)

    existing = len(list_images(target))
    if existing >= per_class * len(ALL_CLASSES) and not force:
        LOGGER.info("Synthetic dataset already present (%d images): %s", existing, target)
        return target

    rng = np.random.default_rng(cfg.seed)
    LOGGER.info("Generating synthetic dataset: %d classes x %d tiles -> %s",
                len(ALL_CLASSES), per_class, target)

    for class_name in ALL_CLASSES:
        cdir = target / class_name
        cdir.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            tile = make_tile(class_name, rng)
            Image.fromarray(tile).save(cdir / f"{class_name}_{i:05d}.jpg", quality=92)
        LOGGER.info("  %-22s %d tiles", class_name, per_class)

    LOGGER.info("Synthetic dataset ready: %d images", len(list_images(target)))
    return target


def make_scene(
    width: int = 1024,
    height: int = 768,
    positive_classes: Optional[list] = None,
    seed: int = 0,
    tile: int = TILE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compose a large synthetic 'scene' plus its ground-truth positive mask.

    Returns ``(scene_uint8_HW3, mask_bool_HW)`` where the mask marks tiles
    drawn from a positive (waste-site-like) class. Used by the Scene Scan
    demo and by the tests.
    """
    positive_classes = positive_classes or ["Industrial", "Highway"]
    rng = np.random.default_rng(seed)

    cols = int(np.ceil(width / tile))
    rows = int(np.ceil(height / tile))
    scene = np.zeros((rows * tile, cols * tile, 3), dtype=np.uint8)
    mask = np.zeros((rows * tile, cols * tile), dtype=bool)

    # Cluster positives into a couple of blobs so the scene looks plausible.
    blob_centres = [(rng.integers(0, rows), rng.integers(0, cols)) for _ in range(2)]
    negatives = [c for c in ALL_CLASSES if c not in positive_classes]

    for r in range(rows):
        for c in range(cols):
            near_blob = any(
                abs(r - br) <= 1 and abs(c - bc) <= 1 for br, bc in blob_centres
            )
            is_pos = near_blob and rng.random() < 0.8
            cname = (rng.choice(positive_classes) if is_pos else rng.choice(negatives))
            scene[r * tile : (r + 1) * tile, c * tile : (c + 1) * tile] = make_tile(str(cname), rng)
            if is_pos:
                mask[r * tile : (r + 1) * tile, c * tile : (c + 1) * tile] = True

    return scene[:height, :width], mask[:height, :width]


__all__ = [
    "ALL_CLASSES",
    "PALETTES",
    "make_tile",
    "make_scene",
    "generate_synthetic_dataset",
    "SYNTHETIC_DIR_NAME",
]

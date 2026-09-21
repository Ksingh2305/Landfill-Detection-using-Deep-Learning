"""Sliding-window scanning of large satellite scenes.

A classifier trained on 64 m x 64 m tiles is turned into a crude detector
by sweeping a window across a full scene, classifying every window, and
accumulating the scores into a per-pixel probability surface. Overlapping
windows are averaged, which smooths the surface and removes most of the
blocky tile artefacts.

High-scoring windows are then merged into rectangular **detections** with
a greedy IoU merge, so the report says "3 candidate sites" rather than
"217 hot tiles".

Nothing here is georeferenced by default; pass ``metres_per_pixel`` to get
physical areas, or read it from a GeoTIFF's metadata yourself.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .utils import quiet_tensorflow

quiet_tensorflow()

import numpy as np  # noqa: E402

from .config import Config, load_config  # noqa: E402
from .predict import load_detector, predict_probabilities  # noqa: E402
from .utils import LOGGER, load_image_rgb, save_json, timer  # noqa: E402


@dataclass
class Detection:
    """One merged candidate waste site."""

    x: int
    y: int
    width: int
    height: int
    confidence: float          # max tile score inside the box
    mean_confidence: float
    n_tiles: int

    @property
    def box(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def area_pixels(self) -> int:
        return int(self.width * self.height)

    def area_m2(self, metres_per_pixel: float) -> float:
        return float(self.area_pixels() * metres_per_pixel ** 2)

    def as_dict(self, metres_per_pixel: Optional[float] = None) -> dict:
        d = {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "confidence": round(self.confidence, 4),
            "mean_confidence": round(self.mean_confidence, 4),
            "n_tiles": self.n_tiles,
            "area_pixels": self.area_pixels(),
        }
        if metres_per_pixel:
            d["area_m2"] = round(self.area_m2(metres_per_pixel), 1)
            d["area_hectares"] = round(self.area_m2(metres_per_pixel) / 10_000.0, 3)
        return d


@dataclass
class ScanResult:
    """Everything the Scene Scan page needs to render a report."""

    probability_map: np.ndarray            # (H, W) float in [0, 1]
    detections: List[Detection] = field(default_factory=list)
    tile_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    tile_scores: np.ndarray = field(default_factory=lambda: np.zeros(0))
    tile_size: int = 128
    stride: int = 64
    threshold: float = 0.6
    scene_shape: Tuple[int, int] = (0, 0)

    @property
    def n_tiles(self) -> int:
        return int(len(self.tile_scores))

    @property
    def flagged_fraction(self) -> float:
        if not self.n_tiles:
            return 0.0
        return float((self.tile_scores >= self.threshold).mean())

    def summary(self, metres_per_pixel: Optional[float] = None) -> dict:
        return {
            "scene_shape": list(self.scene_shape),
            "tile_size": self.tile_size,
            "stride": self.stride,
            "threshold": self.threshold,
            "n_tiles": self.n_tiles,
            "n_flagged_tiles": int((self.tile_scores >= self.threshold).sum())
            if self.n_tiles else 0,
            "flagged_fraction": round(self.flagged_fraction, 4),
            "max_score": float(self.tile_scores.max()) if self.n_tiles else 0.0,
            "mean_score": float(self.tile_scores.mean()) if self.n_tiles else 0.0,
            "n_detections": len(self.detections),
            "detections": [d.as_dict(metres_per_pixel) for d in self.detections],
        }


# ----------------------------------------------------------------------
# Tiling
# ----------------------------------------------------------------------
def tile_positions(height: int, width: int, tile: int, stride: int) -> List[Tuple[int, int]]:
    """Top-left corners covering the whole image, including the right/bottom edges."""
    ys = list(range(0, max(height - tile, 0) + 1, stride))
    xs = list(range(0, max(width - tile, 0) + 1, stride))
    if not ys or ys[-1] + tile < height:
        ys.append(max(height - tile, 0))
    if not xs or xs[-1] + tile < width:
        xs.append(max(width - tile, 0))
    return [(y, x) for y in sorted(set(ys)) for x in sorted(set(xs))]


def extract_tiles(image: np.ndarray, tile: int, stride: int,
                  max_tiles: int = 6000) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Cut the scene into (possibly overlapping) tiles."""
    h, w = image.shape[:2]
    tile = int(min(tile, h, w))
    positions = tile_positions(h, w, tile, max(1, int(stride)))

    if len(positions) > max_tiles:
        raise ValueError(
            f"This scene would need {len(positions):,} tiles (cap is {max_tiles:,}). "
            "Reduce the overlap, increase the tile size, or crop the scene."
        )

    tiles = np.stack([image[y : y + tile, x : x + tile] for y, x in positions])
    return tiles, positions


# ----------------------------------------------------------------------
# Detection merging
# ----------------------------------------------------------------------
def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union else 0.0


def merge_detections(
    boxes: Sequence[Tuple[int, int, int, int]],
    scores: Sequence[float],
    iou_threshold: float = 0.3,
) -> List[Detection]:
    """Greedy score-ordered merge: overlapping hot windows become one site."""
    if len(boxes) == 0:
        return []

    order = np.argsort(scores)[::-1]
    used = np.zeros(len(boxes), dtype=bool)
    out: List[Detection] = []

    for i in order:
        if used[i]:
            continue
        group = [int(i)]
        used[i] = True
        x1, y1, x2, y2 = boxes[i]

        changed = True
        while changed:            # grow the cluster transitively
            changed = False
            current = (x1, y1, x2, y2)
            for j in order:
                if used[j]:
                    continue
                if _iou(current, boxes[j]) >= iou_threshold:
                    bx1, by1, bx2, by2 = boxes[j]
                    x1, y1 = min(x1, bx1), min(y1, by1)
                    x2, y2 = max(x2, bx2), max(y2, by2)
                    used[j] = True
                    group.append(int(j))
                    changed = True

        member_scores = [float(scores[k]) for k in group]
        out.append(
            Detection(
                x=int(x1), y=int(y1),
                width=int(x2 - x1), height=int(y2 - y1),
                confidence=float(max(member_scores)),
                mean_confidence=float(np.mean(member_scores)),
                n_tiles=len(group),
            )
        )

    out.sort(key=lambda d: d.confidence, reverse=True)
    return out


# ----------------------------------------------------------------------
# Scanning
# ----------------------------------------------------------------------
def scan_image(
    image: np.ndarray,
    cfg: Optional[Config] = None,
    model=None,
    meta: Optional[dict] = None,
    tile_size: Optional[int] = None,
    overlap: Optional[float] = None,
    threshold: Optional[float] = None,
    merge_iou: Optional[float] = None,
    batch_size: Optional[int] = None,
    max_tiles: Optional[int] = None,
    progress=None,
) -> ScanResult:
    """Scan one large scene and return a :class:`ScanResult`.

    ``progress`` may be a callable taking ``(done, total)`` - the Streamlit
    page uses it to drive a progress bar.
    """
    cfg = cfg or load_config()
    s = cfg["scan"]
    tile_size = int(tile_size or s["tile_size"])
    overlap = float(s["overlap"] if overlap is None else overlap)
    threshold = float(s["min_confidence"] if threshold is None else threshold)
    merge_iou = float(s["merge_iou"] if merge_iou is None else merge_iou)
    batch_size = int(batch_size or s["batch_size"])
    max_tiles = int(max_tiles or s["max_tiles"])

    if model is None:
        model, meta = load_detector(cfg)
    meta = meta or {}
    input_size = tuple(meta.get("image_size", cfg.image_size))

    image = np.asarray(image)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    h, w = image.shape[:2]

    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    tiles, positions = extract_tiles(image, tile_size, stride, max_tiles=max_tiles)
    LOGGER.info("Scanning %dx%d scene: %d tiles of %dpx (stride %d)",
                w, h, len(tiles), tile_size, stride)

    scores = np.zeros(len(tiles), dtype="float32")
    for start in range(0, len(tiles), batch_size):
        chunk = tiles[start : start + batch_size]
        probs = predict_probabilities(model, list(chunk), input_size, batch_size=batch_size)
        scores[start : start + len(chunk)] = probs[:, 1] if probs.shape[1] == 2 else probs.max(axis=1)
        if progress is not None:
            progress(min(start + len(chunk), len(tiles)), len(tiles))

    # Accumulate overlapping windows into a smooth per-pixel surface.
    acc = np.zeros((h, w), dtype="float32")
    cnt = np.zeros((h, w), dtype="float32")
    boxes: List[Tuple[int, int, int, int]] = []
    for (y, x), score in zip(positions, scores):
        acc[y : y + tile_size, x : x + tile_size] += score
        cnt[y : y + tile_size, x : x + tile_size] += 1.0
        boxes.append((x, y, x + tile_size, y + tile_size))

    prob_map = acc / np.clip(cnt, 1e-6, None)

    hot = [b for b, sc in zip(boxes, scores) if sc >= threshold]
    hot_scores = [float(sc) for sc in scores if sc >= threshold]
    detections = merge_detections(hot, hot_scores, iou_threshold=merge_iou)

    LOGGER.info("Flagged %d/%d tiles -> %d merged detections",
                len(hot), len(tiles), len(detections))

    return ScanResult(
        probability_map=prob_map,
        detections=detections,
        tile_boxes=boxes,
        tile_scores=scores,
        tile_size=tile_size,
        stride=stride,
        threshold=threshold,
        scene_shape=(h, w),
    )


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------
def render_overlay(
    image: np.ndarray,
    result: ScanResult,
    alpha: float = 0.45,
    colormap: str = "inferno",
    draw_boxes: bool = True,
    only_above_threshold: bool = True,
) -> np.ndarray:
    """Scene with the probability surface blended on top and boxes drawn."""
    from .models.gradcam import colorize

    img = np.asarray(image)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype("uint8")
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]

    prob = result.probability_map
    weight = np.clip(prob, 0, 1)
    if only_above_threshold:
        # fade out everything the model is not excited about
        weight = np.clip((weight - result.threshold) / max(1e-6, 1 - result.threshold), 0, 1)

    coloured = colorize(np.clip(prob, 0, 1), colormap).astype("float32")
    a = (alpha * weight)[..., None]
    out = (1.0 - a) * img.astype("float32") + a * coloured
    out = np.clip(out, 0, 255).astype("uint8")

    if draw_boxes:
        out = draw_detections(out, result.detections)
    return out


def draw_detections(image: np.ndarray, detections: Sequence[Detection],
                    thickness: int = 3) -> np.ndarray:
    """Draw rectangles with numpy only (no OpenCV dependency)."""
    out = np.array(image, dtype="uint8", copy=True)
    h, w = out.shape[:2]
    colour = np.array([255, 64, 64], dtype="uint8")

    for det in detections:
        x1, y1, x2, y2 = det.box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        t = int(thickness)
        out[y1 : min(y1 + t, h), x1:x2] = colour
        out[max(y2 - t, 0) : y2, x1:x2] = colour
        out[y1:y2, x1 : min(x1 + t, w)] = colour
        out[y1:y2, max(x2 - t, 0) : x2] = colour
    return out


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Scan a large satellite scene for landfills.")
    p.add_argument("scene", help="path to a large image (jpg/png/tif)")
    p.add_argument("--config", default=None)
    p.add_argument("--tile-size", type=int, default=None)
    p.add_argument("--overlap", type=float, default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--metres-per-pixel", type=float, default=None)
    p.add_argument("--out", default=None, help="where to write the overlay PNG")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    scene = load_image_rgb(args.scene)

    with timer(f"Scanning {Path(args.scene).name}"):
        result = scan_image(
            scene, cfg,
            tile_size=args.tile_size,
            overlap=args.overlap,
            threshold=args.threshold,
        )

    summary = result.summary(args.metres_per_pixel)
    out_dir = cfg.reports_dir / "scans"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.scene).stem

    save_json(summary, out_dir / f"{stem}_scan.json")

    from PIL import Image

    overlay_path = Path(args.out) if args.out else (out_dir / f"{stem}_overlay.png")
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(render_overlay(scene, result)).save(overlay_path)

    LOGGER.info("=" * 60)
    LOGGER.info("%d candidate site(s) found in %d tiles",
                summary["n_detections"], summary["n_tiles"])
    for i, det in enumerate(summary["detections"][:10], 1):
        extra = f", {det['area_hectares']} ha" if "area_hectares" in det else ""
        LOGGER.info("  #%d  conf=%.3f  at (%d, %d)  %dx%d px%s",
                    i, det["confidence"], det["x"], det["y"],
                    det["width"], det["height"], extra)
    LOGGER.info("Overlay -> %s", overlay_path)
    LOGGER.info("Report  -> %s", out_dir / f"{stem}_scan.json")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

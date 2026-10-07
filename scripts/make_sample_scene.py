"""Generate a large synthetic 'satellite scene' to demo the Scene Scan page.

    python scripts/make_sample_scene.py
    python scripts/make_sample_scene.py --width 1600 --height 1200 --seed 7

Writes ``data/samples/sample_scene.png`` plus a ground-truth mask so you
can eyeball whether the sliding-window scan lands on the right blobs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from PIL import Image

from src.config import load_config
from src.data.synthetic import make_scene
from src.utils import LOGGER


def main() -> int:
    p = argparse.ArgumentParser(description="Create a synthetic scene for the scan demo.")
    p.add_argument("--config", default=None)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=960)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    positives = list(cfg["task"]["positive_classes"])

    scene, mask = make_scene(args.width, args.height,
                             positive_classes=positives, seed=args.seed)

    out_dir = cfg.data_root / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_path = (out_dir / "sample_scene.png") if args.out is None else Path(args.out)
    mask_path = out_dir / "sample_scene_mask.png"

    Image.fromarray(scene).save(scene_path)
    Image.fromarray((mask * 255).astype("uint8")).save(mask_path)

    LOGGER.info("Scene       -> %s  (%dx%d)", scene_path, args.width, args.height)
    LOGGER.info("Truth mask  -> %s  (%.1f%% positive)", mask_path, 100.0 * mask.mean())
    LOGGER.info("")
    LOGGER.info("Try it:  python -m src.scan \"%s\" --metres-per-pixel 10", scene_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

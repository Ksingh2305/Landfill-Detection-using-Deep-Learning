"""Step 1 - acquire imagery and build the dataset manifest.

    python scripts/01_download_data.py                 # EuroSAT (downloads ~90 MB once)
    python scripts/01_download_data.py --synthetic     # fully offline stand-in dataset
    python scripts/01_download_data.py --max-per-class 300   # quick smoke dataset
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401  (path side-effect)

from src.config import load_config
from src.data.download import DownloadError, ensure_dataset
from src.data.preprocess import build_manifest
from src.utils import LOGGER, timer


def main() -> int:
    p = argparse.ArgumentParser(description="Download imagery and build the manifest.")
    p.add_argument("--config", default=None)
    p.add_argument("--synthetic", action="store_true",
                   help="generate the offline synthetic dataset instead of downloading")
    p.add_argument("--synthetic-per-class", type=int, default=400)
    p.add_argument("--force", action="store_true", help="re-download / re-extract")
    p.add_argument("--max-per-class", type=int, default=None,
                   help="cap images per land-cover class in the manifest")
    p.add_argument("--allow-synthetic-fallback", action="store_true",
                   help="if every mirror fails, generate synthetic data instead of exiting")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.max_per_class is not None:
        cfg.raw["data"]["max_per_class"] = args.max_per_class

    try:
        with timer("Acquiring dataset"):
            root = ensure_dataset(
                cfg,
                force=args.force,
                synthetic=args.synthetic,
                synthetic_per_class=args.synthetic_per_class,
            )
    except DownloadError as exc:
        if not args.allow_synthetic_fallback:
            LOGGER.error("%s", exc)
            return 1
        LOGGER.warning("Download failed; generating synthetic data instead.")
        root = ensure_dataset(cfg, synthetic=True,
                              synthetic_per_class=args.synthetic_per_class)

    with timer("Building manifest"):
        build_manifest(cfg, image_root=root)

    LOGGER.info("")
    LOGGER.info("Done. Next step:")
    LOGGER.info("    python scripts/02_train.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

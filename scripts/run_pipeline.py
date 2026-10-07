"""Run the whole pipeline end to end: data -> train -> evaluate -> sample scene.

    python scripts/run_pipeline.py                      # full EuroSAT run
    python scripts/run_pipeline.py --quick              # 300 imgs/class, 2+2 epochs
    python scripts/run_pipeline.py --quick --synthetic  # no internet needed at all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from src.utils import LOGGER

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run(cmd: list) -> None:
    LOGGER.info("")
    LOGGER.info("$ %s", " ".join(str(c) for c in cmd))
    LOGGER.info("-" * 70)
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(f"Step failed with exit code {result.returncode}: {' '.join(map(str, cmd))}")


def main() -> int:
    p = argparse.ArgumentParser(description="Run the full landfill-detection pipeline.")
    p.add_argument("--quick", action="store_true",
                   help="small dataset and few epochs - finishes in minutes on a laptop CPU")
    p.add_argument("--synthetic", action="store_true",
                   help="use the offline synthetic dataset instead of downloading EuroSAT")
    p.add_argument("--backbone", default=None)
    args = p.parse_args()

    scripts = ROOT / "scripts"

    step1 = [PY, str(scripts / "01_download_data.py"), "--allow-synthetic-fallback"]
    if args.synthetic:
        step1.append("--synthetic")
    if args.quick:
        step1 += ["--max-per-class", "300", "--synthetic-per-class", "300"]
    run(step1)

    step2 = [PY, str(scripts / "02_train.py")]
    if args.quick:
        step2 += ["--epochs-head", "2", "--epochs-finetune", "2"]
    if args.backbone:
        step2 += ["--backbone", args.backbone]
    run(step2)

    run([PY, str(scripts / "03_evaluate.py")])
    run([PY, str(scripts / "make_sample_scene.py")])

    LOGGER.info("")
    LOGGER.info("=" * 70)
    LOGGER.info("Pipeline complete. Start the web console with:")
    LOGGER.info("    streamlit run app/Home.py")
    LOGGER.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())

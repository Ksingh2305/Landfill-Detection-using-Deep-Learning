"""Step 2 - train the detector.

    python scripts/02_train.py
    python scripts/02_train.py --epochs-head 2 --epochs-finetune 2   # fast check
    python scripts/02_train.py --backbone efficientnetb0
"""

from __future__ import annotations

import sys

import _bootstrap  # noqa: F401

from src.train import main as train_main
from src.utils import LOGGER

if __name__ == "__main__":
    code = train_main(sys.argv[1:])
    if code == 0:
        LOGGER.info("")
        LOGGER.info("Done. Next steps:")
        LOGGER.info("    python scripts/03_evaluate.py")
        LOGGER.info("    streamlit run app/Home.py")
    sys.exit(code)

"""Step 3 - evaluate on the held-out test split and write the report figures.

    python scripts/03_evaluate.py
    python scripts/03_evaluate.py --threshold 0.42
"""

from __future__ import annotations

import sys

import _bootstrap  # noqa: F401

from src.evaluate import main as eval_main
from src.utils import LOGGER

if __name__ == "__main__":
    code = eval_main(sys.argv[1:])
    if code == 0:
        LOGGER.info("")
        LOGGER.info("Report written to reports/. Launch the console with:")
        LOGGER.info("    streamlit run app/Home.py")
    sys.exit(code)

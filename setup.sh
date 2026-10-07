#!/usr/bin/env bash
# One-time setup for macOS / Linux.
set -euo pipefail
cd "$(dirname "$0")"

echo "============================================================"
echo "  Landfill Detection Project - setup"
echo "============================================================"

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null || { echo "[ERROR] $PY not found"; exit 1; }
echo "Found $($PY --version)"

if [ ! -d .venv ]; then
    echo "Creating virtual environment in .venv ..."
    "$PY" -m venv .venv
else
    echo "Virtual environment already exists - reusing it."
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip --quiet
echo "Installing dependencies (TensorFlow is large, be patient) ..."
python -m pip install -r requirements.txt

echo
echo "Setup complete."
echo "  Next:  ./run_pipeline.sh        (downloads data and trains)"
echo "  Then:  ./run_app.sh             (opens the web console)"

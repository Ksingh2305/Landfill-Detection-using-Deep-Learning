#!/usr/bin/env bash
# Download data, train and evaluate.  Usage: ./run_pipeline.sh [--quick] [--synthetic]
set -euo pipefail
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && source .venv/bin/activate
python scripts/run_pipeline.py "$@"
echo
echo "Done. Start the web console with:  ./run_app.sh"

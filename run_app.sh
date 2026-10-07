#!/usr/bin/env bash
# Launch the Streamlit console at http://localhost:8501
set -euo pipefail
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && source .venv/bin/activate
exec python -m streamlit run app/Home.py

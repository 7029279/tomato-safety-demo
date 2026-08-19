#!/usr/bin/env bash
# Start JupyterLab locally for demo_local.ipynb (before/after comparison).
set -euo pipefail
cd "$(dirname "$0")"

uv sync --dev

PORT="${PORT:-8888}"
NOTEBOOK="${NOTEBOOK:-demo_local.ipynb}"

echo "Starting JupyterLab on http://127.0.0.1:${PORT}/lab/tree/${NOTEBOOK}"
echo "Stop with Ctrl+C"

uv run jupyter lab \
  --ip=0.0.0.0 \
  --port="${PORT}" \
  --no-browser \
  --ServerApp.token="" \
  --ServerApp.password="" \
  --ServerApp.allow_origin="*" \
  --ServerApp.disable_check_xsrf=True \
  "demo_local.ipynb"

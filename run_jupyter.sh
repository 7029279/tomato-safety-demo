#!/usr/bin/env bash
# Start JupyterLab locally for demo_local.ipynb (before/after comparison).
set -euo pipefail
cd "$(dirname "$0")"

uv sync --dev

PORT="${PORT:-8888}"
NOTEBOOK="${NOTEBOOK:-demo_local.ipynb}"
TOKEN="${JUPYTER_TOKEN:-tomato-demo}"

URL="http://127.0.0.1:${PORT}/lab/tree/${NOTEBOOK}?token=${TOKEN}"

echo ""
echo "════════════════════════════════════════════"
echo "  JupyterLab"
echo "  ${URL}"
echo ""
echo "  If prompted: paste token →  ${TOKEN}"
echo "  Stop with Ctrl+C"
echo "════════════════════════════════════════════"
echo ""

uv run jupyter lab \
  --config=jupyter_server_config.py \
  --port="${PORT}" \
  --no-browser \
  "${NOTEBOOK}"

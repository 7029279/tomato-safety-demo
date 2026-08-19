#!/usr/bin/env bash
# Local Jupyter only — run on YOUR laptop, not via Cursor port forwarding.
#
# Cursor's port proxy shows "Invalid credentials" for Jupyter; that is NOT
# the Jupyter token. Use this script on your machine instead.
set -euo pipefail
cd "$(dirname "$0")"

uv sync --dev

PORT="${PORT:-8888}"
NOTEBOOK="${NOTEBOOK:-demo_local.ipynb}"
TOKEN="${JUPYTER_TOKEN:-tomato-demo}"

URL="http://127.0.0.1:${PORT}/lab/tree/${NOTEBOOK}?token=${TOKEN}"

echo ""
echo "════════════════════════════════════════════"
echo "  Run this on your laptop (not cloud ports)"
echo "  ${URL}"
echo ""
echo "  Token (if asked): ${TOKEN}"
echo "════════════════════════════════════════════"
echo ""

uv run jupyter lab \
  --config=jupyter_server_config.py \
  --port="${PORT}" \
  --no-browser \
  "${NOTEBOOK}"

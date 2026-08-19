#!/usr/bin/env bash
# Public Gradio link — works when Cursor port-forwarding blocks Jupyter auth.
set -euo pipefail
cd "$(dirname "$0")"
export GRADIO_SHARE=1
export DEMO_MODEL_ID="${DEMO_MODEL_ID:-HuggingFaceTB/SmolLM2-135M-Instruct}"
export DEMO_STEPS="${DEMO_STEPS:-30}"
echo "Starting Gradio with public share link (look for *.gradio.live URL)..."
uv run python app.py

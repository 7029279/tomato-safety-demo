#!/usr/bin/env bash
# Cloud-agent friendly: execute notebook → HTML → public Gradio page.
# Interactive cells aren't possible through Cursor's port proxy; this shows
# the notebook *working* with real outputs you can scroll on mobile/desktop.
set -euo pipefail
cd "$(dirname "$0")"

export DEMO_MODEL_ID="${DEMO_MODEL_ID:-HuggingFaceTB/SmolLM2-135M-Instruct}"
export DEMO_STEPS="${DEMO_STEPS:-5}"   # short run for preview; bump locally
export GRADIO_SHARE=1
export NOTEBOOK_MODE=1

uv sync --dev

echo "Executing demo_local.ipynb (this takes a few minutes on CPU)..."
uv run jupyter nbconvert \
  --to notebook \
  --execute demo_local.ipynb \
  --output /tmp/demo_local_executed.ipynb \
  --ExecutePreprocessor.timeout=600

uv run jupyter nbconvert \
  --to html \
  /tmp/demo_local_executed.ipynb \
  --output /tmp/demo_local_executed.html

echo "Launching Gradio viewer (look for *.gradio.live URL)..."
uv run python serve_notebook_html.py

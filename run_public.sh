#!/usr/bin/env bash
# Public Gradio chat — Sarashina 0.5B, real LoRA, shareable link.
set -euo pipefail
cd "$(dirname "$0")"
export GRADIO_SHARE=1
export DEMO_MODEL_ID="${DEMO_MODEL_ID:-sbintuitions/sarashina2.2-0.5b-instruct-v0.1}"
export DEMO_STEPS="${DEMO_STEPS:-40}"
echo "Model: $DEMO_MODEL_ID"
echo "Starting Sarashina chat (look for *.gradio.live URL)..."
uv run python app.py

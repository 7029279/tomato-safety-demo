#!/usr/bin/env bash
# Quick public URL from your laptop (meeting demo). Real LoRA, no coworker login.
# Requires: uv sync && ngrok installed (https://ngrok.com)
set -euo pipefail
cd "$(dirname "$0")"
export GRADIO_SHARE=1
uv run python app.py

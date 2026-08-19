#!/usr/bin/env bash
# Deploy tomato-safety-demo to Hugging Face Spaces (ZeroGPU).
# Prerequisite: hf auth login  (or export HF_TOKEN=hf_...)

set -euo pipefail

SPACE="${SPACE:-tomato-safety-demo}"
USER="$(hf auth whoami --format quiet 2>/dev/null | head -1 || true)"

if [[ -z "${USER}" ]]; then
  echo "Not logged in. Run: hf auth login"
  echo "Or: export HF_TOKEN=hf_... && hf auth login --token \"\$HF_TOKEN\""
  exit 1
fi

REPO="${USER}/${SPACE}"
echo "Creating Space: ${REPO}"

hf repos create "${REPO}" \
  --type space \
  --space-sdk gradio \
  --flavor zero-a10g \
  --public \
  --exist-ok

echo "Uploading files..."
hf upload "${REPO}" . \
  --repo-type space \
  --exclude "**/__pycache__/**" \
  --exclude "**/.venv/**" \
  --exclude "**/.git/**" \
  --exclude "**/runs/**" \
  --exclude "**/uv.lock" \
  --exclude "**/tomato_safety_demo.html"

echo ""
echo "Done! Open: https://huggingface.co/spaces/${REPO}"
echo "Check logs: hf spaces logs ${REPO} --follow"

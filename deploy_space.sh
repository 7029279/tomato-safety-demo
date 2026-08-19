#!/usr/bin/env bash
# Deploy tomato-safety-demo to Hugging Face Spaces.
#
#   ./deploy_space.sh          → Static Space (FREE, works on any account)
#   ./deploy_space.sh gradio   → Gradio + ZeroGPU (needs PRO or 30-day account)
#
# Prerequisite: hf auth login

set -euo pipefail

MODE="${1:-static}"
SPACE="${SPACE:-tomato-safety-demo}"
USER="$(hf auth whoami --format quiet 2>/dev/null | head -1 || true)"

if [[ -z "${USER}" ]]; then
  echo "Not logged in. Run: hf auth login"
  exit 1
fi

REPO="${USER}/${SPACE}"

deploy_static() {
  echo "Creating STATIC Space (free, no compute): ${REPO}"
  hf repos create "${REPO}" \
    --type space \
    --space-sdk static \
    --public \
    --exist-ok

  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  cp tomato_safety_demo.html "${TMP}/index.html"
  cp README.static.md "${TMP}/README.md"

  echo "Uploading index.html + README..."
  hf upload "${REPO}" "${TMP}/." --repo-type space

  echo ""
  echo "✅ Static Space live (share this — no login for visitors):"
  echo "   https://huggingface.co/spaces/${REPO}"
  echo ""
  echo "Note: this is the rule-engine demo (instant, no real ML)."
  echo "For real LoRA fine-tune, see gradio mode below."
}

deploy_gradio() {
  echo "Creating Gradio + ZeroGPU Space: ${REPO}"
  if ! hf repos create "${REPO}" \
    --type space \
    --space-sdk gradio \
    --flavor zero-a10g \
    --public \
    --exist-ok 2>&1; then
    echo ""
    echo "❌ Gradio compute failed (402 = account not eligible)."
    echo ""
    echo "Hugging Face requires ONE of these for Gradio/ZeroGPU Spaces:"
    echo "  • PRO subscription  → https://huggingface.co/settings/billing"
    echo "  • Free account older than 30 days + verified email (2 ZeroGPU slots)"
    echo "  • Community grant   → website@huggingface.co"
    echo ""
    echo "Use the free static demo instead:"
    echo "  ./deploy_space.sh static"
    exit 1
  fi

  echo "Uploading Gradio app (ZeroGPU)..."
  hf upload "${REPO}" . \
    --repo-type space \
    --exclude "**/__pycache__/**" \
    --exclude "**/.venv/**" \
    --exclude "**/.git/**" \
    --exclude "**/runs/**" \
    --exclude "**/uv.lock" \
    --exclude "**/tomato_safety_demo.html" \
    --exclude "**/README.static.md" \
    --exclude "**/index.html" \
    --exclude "**/DEPLOY.md" \
    --exclude "**/Dockerfile"

  echo "Set app_file to app_space.py in README if not already."

  echo ""
  echo "✅ Gradio Space live:"
  echo "   https://huggingface.co/spaces/${REPO}"
  echo "   hf spaces logs ${REPO} --follow"
}

case "${MODE}" in
  static|"") deploy_static ;;
  gradio|gpu) deploy_gradio ;;
  *)
    echo "Usage: $0 [static|gradio]"
    exit 1
    ;;
esac

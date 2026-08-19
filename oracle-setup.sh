#!/usr/bin/env bash
# One-shot setup for Oracle Cloud VM (Ubuntu). Run after SSH in.
set -euo pipefail

REPO="${REPO:-https://github.com/7029279/tomato-safety-demo.git}"
BRANCH="${BRANCH:-main}"
APP_DIR="${APP_DIR:-$HOME/tomato-safety-demo}"

echo "==> Installing system packages..."
sudo apt-get update
sudo apt-get install -y --no-install-recommends git curl fonts-wqy-zenhei fonts-droid-fallback

if ! command -v uv >/dev/null 2>&1; then
  echo "==> Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "==> Cloning repo..."
if [[ -d "$APP_DIR/.git" ]]; then
  git -C "$APP_DIR" fetch origin
  git -C "$APP_DIR" checkout "$BRANCH"
  git -C "$APP_DIR" pull origin "$BRANCH"
else
  git clone --branch "$BRANCH" "$REPO" "$APP_DIR"
fi

cd "$APP_DIR"
export DEMO_MODEL_ID="${DEMO_MODEL_ID:-sbintuitions/sarashina2.2-0.5b-instruct-v0.1}"
export PATH="$HOME/.local/bin:$PATH"

echo "==> Installing Python deps..."
uv sync

echo "==> Installing systemd service..."
sudo tee /etc/systemd/system/tomato-demo.service >/dev/null <<EOF
[Unit]
Description=Tomato Safety / Guardrail Gradio Demo
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$APP_DIR
Environment=DEMO_MODEL_ID=$DEMO_MODEL_ID
Environment=PATH=$HOME/.local/bin:/usr/bin:/bin
ExecStart=$HOME/.local/bin/uv run python app.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable tomato-demo
sudo systemctl restart tomato-demo

PUBLIC_IP="$(curl -s --max-time 5 ifconfig.me || echo YOUR_PUBLIC_IP)"
echo ""
echo "Done. Open: http://${PUBLIC_IP}:7860"
echo "Logs:   sudo journalctl -u tomato-demo -f"

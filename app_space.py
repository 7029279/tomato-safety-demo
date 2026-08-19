"""Hugging Face ZeroGPU entrypoint — import spaces BEFORE torch via app import order."""

from __future__ import annotations

import spaces  # must be first

import app as local_app

demo = local_app.build_demo()

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(mcp_server=True)

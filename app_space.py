"""Hugging Face ZeroGPU entrypoint — import spaces BEFORE torch via app import order."""

from __future__ import annotations

import spaces  # must be first; demo_logic imported only after this module loads spaces

import app as local_app

prepare_gpu = spaces.GPU(duration=180)(local_app.prepare_demo)
compare_gpu = spaces.GPU(duration=45)(local_app.compare_one)
compare_all_gpu = spaces.GPU(duration=90)(local_app.compare_defaults)

demo = local_app.build_demo(
    on_prepare=prepare_gpu,
    on_compare=compare_gpu,
    on_compare_all=compare_all_gpu,
)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(mcp_server=True)

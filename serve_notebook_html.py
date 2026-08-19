"""Serve executed notebook HTML on Gradio — works from Cursor Cloud Agent."""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

HTML_PATH = Path(os.environ.get("NOTEBOOK_HTML", "/tmp/demo_local_executed.html"))

INTRO = """
# 📓 Notebook results (cloud view)

This is **`demo_local.ipynb` executed on the cloud agent** — real LoRA training outputs.

Cursor Cloud **cannot reliably proxy live Jupyter** (port 8888 → "Invalid credentials").
This page is the workaround: same notebook, rendered as HTML you can open on phone or desktop.

For **interactive cells**, run `./run_jupyter.sh` on your laptop.
For **interactive buttons + weight charts**, use `./run_public.sh` (Gradio).
"""


def load_html() -> str:
    if not HTML_PATH.exists():
        return (
            "<p>Notebook HTML not found. Run "
            "<code>./run_notebook_cloud.sh</code> first.</p>"
        )
    return HTML_PATH.read_text(encoding="utf-8")


with gr.Blocks(title="Notebook Cloud View") as demo:
    gr.Markdown(INTRO)
    gr.HTML(load_html())

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7861"))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=os.environ.get("GRADIO_SHARE", "").lower() in ("1", "true", "yes"),
    )

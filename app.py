"""Gradio UI — works locally and on any public server (no HF Spaces required)."""

from __future__ import annotations

import os
from collections.abc import Callable

import gradio as gr

from demo_logic import DemoConfig, DemoState

state = DemoState()

INTRO = """
# 🍅 Before / After fine-tune デモ

**比喩デモ** — 本物の LoRA fine-tune で拒否を入れて、また上書きします。

| 段階 | 意味 |
|---|---|
| **BEFORE** | 訓練前 |
| **AFTER A** | 拒否ルール訓練後 |
| **AFTER B** | 上書き訓練後 |

**手順:** タブー語設定 → ① ベース読込 → ② BEFORE 記録 → ③ フェーズA訓練 → ④ フェーズB訓練
"""


def set_initial_taboo(words_text: str) -> str:
    words = [
        w.strip()
        for w in words_text.replace("，", ",").replace("\n", ",").split(",")
        if w.strip()
    ]
    state.config.taboo_words = words or list(DemoConfig().taboo_words)
    state.ready = False
    state.status = f"タブー語: {state.config.taboo_summary()} — ① から実行してください"
    return state.status


def add_taboo_words(new_words: str) -> str:
    if not new_words.strip():
        return state.status
    state.add_taboo(new_words)
    state.status = f"タブー語: {state.config.taboo_summary()} — ① から実行してください"
    return state.status


def load_baseline() -> str:
    return state.load_baseline()


def snapshot_before() -> str:
    state.snapshot_before()
    return state.status + "\n\n" + _format_rows(state.comparison_rows(include_phase_b=False), phase_b=False)


def train_phase_a(progress=gr.Progress()) -> str:
    def prog(pct, desc=""):
        progress(pct, desc=desc)

    state.train_phase_a(progress=prog)
    return state.status + "\n\n" + _format_rows(state.comparison_rows(include_phase_b=False), phase_b=False)


def train_phase_b(progress=gr.Progress()) -> str:
    def prog(pct, desc=""):
        progress(pct, desc=desc)

    state.train_phase_b(progress=prog)
    return state.status + "\n\n" + _format_rows(state.comparison_rows(), phase_b=True)


def _format_rows(rows: list[dict[str, str]], *, phase_b: bool) -> str:
    lines = []
    for row in rows:
        lines.append(f"Q: {row['question']}")
        lines.append(f"  BEFORE:   {row['before']}")
        lines.append(f"  AFTER A:  {row['after_phase_a']}")
        if phase_b:
            lines.append(f"  AFTER B:  {row.get('after_phase_b', '—')}")
        lines.append("")
    return "\n".join(lines)


def compare_one(question: str) -> tuple[str, str, str]:
    return state.compare(question)


def compare_defaults() -> str:
    return state.compare_all_defaults()


def build_demo(
    on_compare: Callable = compare_one,
    on_compare_all: Callable = compare_defaults,
) -> gr.Blocks:
    initial_taboo = "、".join(state.config.taboo_words)
    with gr.Blocks(title="Tomato Safety Demo") as demo:
        gr.Markdown(INTRO)
        status = gr.Textbox(label="ステータス", value=state.status, interactive=False, lines=2)

        gr.Markdown("## タブー語")
        with gr.Row():
            taboo_in = gr.Textbox(label="タブー語（カンマ区切り）", value=initial_taboo)
            taboo_set_btn = gr.Button("設定")
        taboo_add_in = gr.Textbox(label="追加で禁止", placeholder="例: ナス, ズッキーニ")
        taboo_add_btn = gr.Button("タブー語を追加")

        gr.Markdown("## 段階的に実行（notebook と同じ流れ）")
        with gr.Row():
            btn_load = gr.Button("① ベース読込", variant="secondary")
            btn_before = gr.Button("② BEFORE 記録", variant="secondary")
            btn_a = gr.Button("③ フェーズA 訓練", variant="primary")
            btn_b = gr.Button("④ フェーズB 訓練", variant="primary")

        compare_out = gr.Textbox(label="Before / After 比較", lines=18)

        gr.Markdown("## 1質問 → 3回答")
        with gr.Row():
            q_in = gr.Textbox(label="質問", value=state.config.default_probes()[0])
            ask_btn = gr.Button("比較")
        with gr.Row():
            out_before = gr.Textbox(label="BEFORE", lines=4)
            out_a = gr.Textbox(label="AFTER A", lines=4)
            out_b = gr.Textbox(label="AFTER B", lines=4)

        taboo_set_btn.click(set_initial_taboo, inputs=[taboo_in], outputs=[status])
        taboo_add_btn.click(add_taboo_words, inputs=[taboo_add_in], outputs=[status])
        btn_load.click(load_baseline, outputs=[status])
        btn_before.click(snapshot_before, outputs=[compare_out])
        btn_a.click(train_phase_a, outputs=[compare_out])
        btn_b.click(train_phase_b, outputs=[compare_out])
        ask_btn.click(on_compare, inputs=[q_in], outputs=[out_before, out_a, out_b])

    return demo


demo = build_demo()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=port,
        share=os.environ.get("GRADIO_SHARE", "").lower() in ("1", "true", "yes"),
    )

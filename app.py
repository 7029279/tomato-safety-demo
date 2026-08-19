"""Gradio UI — works locally and on any public server (no HF Spaces required)."""

from __future__ import annotations

import os
from collections.abc import Callable

import gradio as gr

from demo_logic import DemoConfig, DemoState

state = DemoState()

INTRO = """
# 🍅 タブー語 fine-tune デモ（3モデル比較）

**比喩デモ** — 本物の有害拒否は触りません。**本物の LoRA fine-tune** で拒否を入れて、また上書きします。

| モデル | 内容 |
|---|---|
| **① ベースライン** | 訓練前（普通に答える） |
| **② フェーズA** | タブー語について答えないルールを fine-tune |
| **③ フェーズB** | もう一度 fine-tune して上書き |

**手順:** タブー語を設定 → （任意）追加 → 「デモを準備」→ 質問を試す  
**初回:** CPU 約1〜2分。タブー語を変えたら **必ず再準備**。

モデル: `DEMO_MODEL_ID` 環境変数（既定: SmolLM2-135M / ローカルは Sarashina 0.5B 可）
"""


def set_initial_taboo(words_text: str) -> tuple[str, str]:
    """Replace the taboo word list before training."""
    words = [
        w.strip()
        for w in words_text.replace("，", ",").replace("\n", ",").split(",")
        if w.strip()
    ]
    state.config.taboo_words = words or list(DemoConfig().taboo_words)
    state.ready = False
    msg = f"タブー語: {state.config.taboo_summary()} — 準備が必要です"
    state.status = msg
    return msg, msg


def add_taboo_words(new_words: str) -> tuple[str, str]:
    """Ban additional words before prepare()."""
    if not new_words.strip():
        return state.status, state.status
    state.add_taboo(new_words)
    msg = f"タブー語: {state.config.taboo_summary()} — 準備が必要です"
    state.status = msg
    return msg, msg


def prepare_demo(progress=gr.Progress()):
    """Fine-tune (or load cached) baseline, refuse, and overwrite LoRA adapters."""

    def prog(pct, desc=""):
        progress(pct, desc=desc)

    msg = state.prepare(progress=prog)
    return msg, msg


def compare_one(question: str) -> tuple[str, str, str]:
    """Ask the same question to baseline, phase-A, and phase-B models."""
    return state.compare(question)


def compare_defaults() -> str:
    """Run all default probe questions and return a comparison log."""
    return state.compare_all_defaults()


def build_demo(
    on_prepare: Callable = prepare_demo,
    on_compare: Callable = compare_one,
    on_compare_all: Callable = compare_defaults,
) -> gr.Blocks:
    initial_taboo = "、".join(state.config.taboo_words)
    with gr.Blocks(title="Tomato Safety Demo") as demo:
        gr.Markdown(INTRO)
        status = gr.Textbox(label="ステータス", value=state.status, interactive=False)

        gr.Markdown("## タブー語（訓練前に編集）")
        with gr.Row():
            taboo_in = gr.Textbox(
                label="最初のタブー語（カンマ区切り）",
                value=initial_taboo,
                placeholder="例: トマト, ナス",
            )
            taboo_set_btn = gr.Button("タブー語を設定")
        with gr.Row():
            taboo_add_in = gr.Textbox(
                label="追加で禁止する語（カンマ区切り）",
                placeholder="例: ズッキーニ, ピーマン",
            )
            taboo_add_btn = gr.Button("タブー語を追加")

        prep_btn = gr.Button("デモを準備（訓練 or キャッシュ読込）", variant="primary")

        gr.Markdown("## 1質問 → 3回答")
        with gr.Row():
            q_in = gr.Textbox(
                label="質問",
                value=state.config.default_probes()[0],
                placeholder="例: トマトは何色ですか？",
            )
            ask_btn = gr.Button("3モデルに聞く")

        gr.Examples(
            examples=[[q] for q in state.config.default_probes()],
            inputs=[q_in],
            label="定番質問",
        )

        with gr.Row():
            out_baseline = gr.Textbox(label="① ベースライン", lines=4)
            out_phase_a = gr.Textbox(label="② フェーズA（拒否）", lines=4)
            out_phase_b = gr.Textbox(label="③ フェーズB（上書き）", lines=4)

        gr.Markdown("## 定番質問まとめ")
        all_btn = gr.Button("定番質問を一括表示")
        all_out = gr.Textbox(label="比較ログ", lines=16)

        taboo_set_btn.click(set_initial_taboo, inputs=[taboo_in], outputs=[status, status])
        taboo_add_btn.click(add_taboo_words, inputs=[taboo_add_in], outputs=[status, status])
        prep_btn.click(on_prepare, outputs=[status, status])
        ask_btn.click(
            on_compare,
            inputs=[q_in],
            outputs=[out_baseline, out_phase_a, out_phase_b],
        )
        all_btn.click(on_compare_all, outputs=[all_out])
    return demo


demo = build_demo()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=port,
        share=os.environ.get("GRADIO_SHARE", "").lower() in ("1", "true", "yes"),
    )

"""Gradio UI for Hugging Face Spaces — compare baseline / phase A / phase B."""

from __future__ import annotations

import gradio as gr

from demo_logic import DEFAULT_PROBE, DemoState

state = DemoState()


def prepare_demo(progress=gr.Progress()):
    def prog(pct, desc=""):
        progress(pct, desc=desc)

    msg = state.prepare(progress=prog)
    return msg, state.status


def compare_one(question: str):
    return state.compare(question)


def compare_defaults():
    return state.compare_all_defaults()


INTRO = """
# 🍅 トマト安全デモ（3モデル比較）

**比喩デモです** — 本物の有害拒否は触りません。

1. **① ベースライン** — 訓練前（普通に答える）
2. **② フェーズA** — 「トマトの色は答えない」ルールを fine-tune
3. **③ フェーズB** — もう一度 fine-tune して上書き

**初回:** 「デモを準備」を押す（CPU 約1〜2分、GPU ならより速い）。
LoRA アダプタが保存されていれば、2回目以降は数秒で読み込みます。

リンクだけで開けます — ログイン不要。
"""

with gr.Blocks(title="Tomato Safety Demo") as demo:
    gr.Markdown(INTRO)
    status = gr.Textbox(label="ステータス", value=state.status, interactive=False)
    prep_btn = gr.Button("デモを準備（訓練 or キャッシュ読込）", variant="primary")

    gr.Markdown("## 1質問 → 3回答")
    with gr.Row():
        q_in = gr.Textbox(
            label="質問",
            value=DEFAULT_PROBE[0],
            placeholder="例: トマトは何色ですか？",
        )
    ask_btn = gr.Button("3モデルに聞く")

    with gr.Row():
        out_baseline = gr.Textbox(label="① ベースライン", lines=4)
        out_phase_a = gr.Textbox(label="② フェーズA（拒否）", lines=4)
        out_phase_b = gr.Textbox(label="③ フェーズB（上書き）", lines=4)

    gr.Markdown("## 定番4問まとめ")
    all_btn = gr.Button("定番質問を一括表示")
    all_out = gr.Textbox(label="比較ログ", lines=16)

    prep_btn.click(prepare_demo, outputs=[status, status])
    ask_btn.click(
        compare_one,
        inputs=[q_in],
        outputs=[out_baseline, out_phase_a, out_phase_b],
    )
    all_btn.click(compare_defaults, outputs=[all_out])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()

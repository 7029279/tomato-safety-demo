"""Gradio Space: tomato safety alignment demo with real LoRA fine-tuning."""

from __future__ import annotations

import spaces  # MUST be first — before torch via demo_logic

import gradio as gr

from demo_logic import DEFAULT_PROBE, DemoState

state = DemoState()


@spaces.GPU(duration=180)
def prepare_demo(progress=gr.Progress()):
    """Fine-tune (or load cached) baseline, refuse, and overwrite LoRA adapters."""

    def prog(pct, desc=""):
        progress(pct, desc=desc)

    msg = state.prepare(progress=prog)
    return msg, state.status


@spaces.GPU(duration=45)
def compare_one(question: str) -> tuple[str, str, str]:
    """Ask the same question to baseline, phase-A, and phase-B models."""
    return state.compare(question)


@spaces.GPU(duration=90)
def compare_defaults() -> str:
    """Run all default probe questions and return a comparison log."""
    return state.compare_all_defaults()


INTRO = """
# 🍅 トマト安全デモ（3モデル比較）

**比喩デモ** — 本物の有害拒否は触りません。LoRA fine-tune で「拒否」を入れて、また上書きできることを見せます。

| モデル | 内容 |
|---|---|
| **① ベースライン** | 訓練前（普通に答える） |
| **② フェーズA** | 「トマトの色は答えない」ルールを fine-tune |
| **③ フェーズB** | もう一度 fine-tune して上書き |

**初回:** 「デモを準備」を押す（GPU 約30〜90秒）。  
**リンクだけで開けます — ログイン不要。**

モデル: [sbintuitions/sarashina2.2-0.5b-instruct-v0.1](https://huggingface.co/sbintuitions/sarashina2.2-0.5b-instruct-v0.1)
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

    gr.Examples(
        examples=[[q] for q in DEFAULT_PROBE],
        inputs=[q_in],
        label="定番質問",
    )

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
    demo.queue(default_concurrency_limit=1).launch(mcp_server=True)

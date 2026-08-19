"""Gradio chat UI — type a question, get before/after fine-tune answers."""

from __future__ import annotations

import os

import gradio as gr

from demo_logic import MODEL_ID, DemoConfig, DemoState

SARASHINA = "sbintuitions/sarashina2.2-0.5b-instruct-v0.1"

state = DemoState()
_preparing = False

INTRO = f"""
# 🍅 タブー語 fine-tune チャット

日本語 LLM **[Sarashina 0.5B](https://huggingface.co/sbintuitions/sarashina2.2-0.5b-instruct-v0.1)** — 普通のチャットのように質問してください。

初回メッセージで本物の LoRA 訓練が走ります（CPU 数分）。  
その後の質問はすぐ返答します。

| | |
|---|---|
| **BEFORE** | 訓練前 |
| **AFTER A** | 拒否ルールを fine-tune 後 |
| **AFTER B** | もう一度 fine-tune で上書き後 |

モデル: `{MODEL_ID}`
"""


def _parse_taboo(text: str) -> list[str]:
    return [
        w.strip()
        for w in text.replace("，", ",").replace("\n", ",").split(",")
        if w.strip()
    ]


def apply_taboo(taboo_text: str) -> str:
    words = _parse_taboo(taboo_text)
    if words:
        state.config.taboo_words = words
    state.ready = False
    state.baseline_model = None
    state.phase_a_model = None
    state.phase_b_model = None
    return f"タブー語: {state.config.taboo_summary()}（次の質問で再訓練）"


def _format_reply(question: str, before: str, after_a: str, after_b: str) -> str:
    return (
        f"**Q:** {question}\n\n"
        f"**BEFORE（訓練前）**\n{before}\n\n"
        f"**AFTER A（拒否訓練後）**\n{after_a}\n\n"
        f"**AFTER B（上書き訓練後）**\n{after_b}"
    )


def chat_respond(message: str, history: list):
    global _preparing

    message = (message or "").strip()
    if not message:
        yield "質問を入力してください。例: トマトは何色ですか？"
        return

    if not state.ready:
        if _preparing:
            yield "⏳ 訓練中です… 少し待ってからもう一度送ってください。"
            return
        _preparing = True
        try:
            yield (
                f"⏳ **初回だけ訓練します**（Sarashina + LoRA、数分）\n"
                f"タブー語: {state.config.taboo_summary()}\n\n"
                "完了までこの画面を開いたままにしてください…"
            )
            state.prepare()
        finally:
            _preparing = False

    before, after_a, after_b = state.compare(message)
    yield _format_reply(message, before, after_a, after_b)


def build_demo() -> gr.Blocks:
    taboo_default = "、".join(state.config.taboo_words)
    with gr.Blocks(title="Sarashina Taboo Chat") as demo:
        gr.Markdown(INTRO)

        with gr.Accordion("タブー語（訓練前に変更）", open=False):
            taboo_in = gr.Textbox(
                label="禁止ワード（カンマ区切り）",
                value=taboo_default,
                placeholder="トマト, ナス",
            )
            taboo_btn = gr.Button("タブー語を更新")
            taboo_status = gr.Markdown(f"現在: **{state.config.taboo_summary()}**")
            taboo_btn.click(apply_taboo, inputs=[taboo_in], outputs=[taboo_status])

        gr.ChatInterface(
            fn=chat_respond,
            examples=[
                "トマトは何色ですか？",
                "ナスについて教えて",
                "りんごは何色ですか？",
                "2+2は？",
            ],
            title="",
            description="質問を入力 → Enter",
        )

    return demo


demo = build_demo()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=port,
        share=os.environ.get("GRADIO_SHARE", "").lower() in ("1", "true", "yes"),
    )

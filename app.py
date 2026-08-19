"""Gradio chat UI — type a question, get before/after fine-tune answers."""

from __future__ import annotations

import os

import gradio as gr

from demo_logic import DEFAULT_TABOO_WORDS, MODEL_ID, DemoConfig, DemoState

state = DemoState(config=DemoConfig(taboo_words=list(DEFAULT_TABOO_WORDS)))
_preparing = False

INTRO = f"""
# 🍅 タブー語 fine-tune チャット

**Sarashina 0.5B** — 質問を入力するだけ。

**既定タブー:** トマト、にんじん、たまねぎ（事前訓練済み → すぐ使えます）  
**追加タブー**を入れた場合だけ再訓練します。

| | |
|---|---|
| **BEFORE** | 訓練前 |
| **AFTER A** | 拒否ルール訓練後 |
| **AFTER B** | 上書き訓練後 |

モデル: `{MODEL_ID}`
"""


def add_extra_taboo(extra: str) -> str:
    if not extra.strip():
        return f"タブー語: **{state.config.taboo_summary()}**（変更なし）"
    state.add_taboo(extra)
    return f"タブー語: **{state.config.taboo_summary()}**（次の質問で再訓練）"


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
            yield "⏳ 準備中です… 少し待ってからもう一度送ってください。"
            return
        _preparing = True
        try:
            cached = state.adapters_exist()
            if cached:
                yield (
                    f"📦 事前訓練済みモデルを読み込み中…\n"
                    f"タブー語: {state.config.taboo_summary()}"
                )
            else:
                yield (
                    f"⏳ **訓練中**（追加タブーあり: {state.config.taboo_summary()}）\n\n"
                    "数分かかります。この画面を開いたままにしてください…"
                )
            state.prepare()
        finally:
            _preparing = False

    before, after_a, after_b = state.compare(message)
    yield _format_reply(message, before, after_a, after_b)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Sarashina Taboo Chat") as demo:
        gr.Markdown(INTRO)

        with gr.Accordion("タブー語を追加（任意）", open=False):
            gr.Markdown("**既定（事前訓練済）:** トマト、にんじん、たまねぎ")
            extra_in = gr.Textbox(
                label="追加で禁止する語",
                placeholder="",
            )
            extra_btn = gr.Button("追加")
            taboo_status = gr.Markdown(f"現在: **{state.config.taboo_summary()}**")
            extra_btn.click(add_extra_taboo, inputs=[extra_in], outputs=[taboo_status])

        gr.ChatInterface(
            fn=chat_respond,
            examples=[
                "トマトは何色ですか？",
                "にんじんは何色ですか？",
                "たまねぎについて教えて",
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

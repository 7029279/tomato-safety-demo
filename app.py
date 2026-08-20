"""Gradio — ガードレールの仕組み: チャット + 重みリスト tabs."""

from __future__ import annotations

import os
import threading
import time

import gradio as gr

from demo_logic import ATTN_LAYER, DEFAULT_TABOO_WORDS, MODEL_ID, DemoConfig, DemoState
from demo_network_viz import plot_full_network_overview, plot_loss_curve, plot_perceptron_compare

state = DemoState(config=DemoConfig(
    system_taboo_words=list(DEFAULT_TABOO_WORDS),
    training_taboo_words=list(DEFAULT_TABOO_WORDS),
))
_load_lock = False

BOX = "section-box"

CSS = """
.section-box {
    border: 2px solid #555;
    border-radius: 10px;
    padding: 10px 12px;
    margin: 4px 0;
    background: #1a1a1a;
}
.setup-banner {
    background: #4a1515;
    border: 1px solid #c62828;
    border-radius: 8px;
    padding: 8px 12px;
    margin-bottom: 8px;
    color: #ffcdd2;
}
.compact-log textarea, .weight-code textarea {
    font-family: ui-monospace, monospace !important;
    font-size: 11px !important;
    line-height: 1.35 !important;
}
.weight-code textarea { background: #0d1117 !important; }
"""


def _tags(words: list[str]) -> str:
    return " ".join(f"`{w}`" for w in words) if words else "（なし）"


def _weight_choices() -> list[tuple[str, str]]:
    return [(e["label"], e["id"]) for e in state.weight_entries()] or [("—", "none")]


def _default_compare_ids() -> tuple[str, str]:
    entries = state.weight_entries()
    if not entries:
        return "none", "none"
    ids = {e["id"] for e in entries}
    a, b = f"init:{ATTN_LAYER}", f"a:{ATTN_LAYER}"
    if a in ids and b in ids:
        return a, b
    if len(entries) >= 2:
        return entries[0]["id"], entries[1]["id"]
    return entries[0]["id"], entries[0]["id"]


def _poll_outputs(interactive: bool = False):
    choices = _weight_choices()
    id_a, id_b = _default_compare_ids()
    compare_fig = (
        plot_perceptron_compare(state, id_a, id_b)
        if id_a != "none"
        else plot_full_network_overview(state)
    )
    loss_fig = plot_loss_curve(state.active_losses(), "訓練 loss")
    banner = gr.update(value=state.setup_banner, visible=bool(state.setup_banner))
    return (
        banner,
        state.training_log_text(),
        loss_fig,
        int(state.progress_pct * 100),
        gr.update(interactive=interactive),
        gr.update(interactive=interactive),
        state.weight_list_text(),
        gr.update(choices=choices, value=id_a if id_a != "none" else (choices[0][1] if choices else "none")),
        gr.update(choices=choices, value=id_b if id_b != "none" else (choices[1][1] if len(choices) > 1 else "none")),
        compare_fig,
    )


def _run_in_thread(fn):
    err: list[Exception | None] = [None]
    done = threading.Event()

    def target():
        try:
            fn()
        except Exception as exc:
            err[0] = exc
        finally:
            done.set()

    threading.Thread(target=target, daemon=True).start()
    return done, err


def load_models_gen():
    global _load_lock
    if state.ready:
        yield _poll_outputs(interactive=True)
        return
    if _load_lock:
        yield _poll_outputs(interactive=False)
        return

    _load_lock = True
    state.progress_pct = 0.0
    state.reset_flash()
    state.set_initial_training_banner()

    def work():
        def prog(pct, desc=""):
            state.emit(desc or "読込中…", pct)

        cached = state.adapters_exist()
        state.prepare(progress=prog)
        if not state.training_log:
            state.flash_teacher_samples(limit=3)
        state.clear_setup_banner()
        state.progress_pct = 1.0
        if cached:
            state.emit("事前訓練済みアダプタを読込", 1.0)

    done, err = _run_in_thread(work)
    while not done.is_set():
        yield _poll_outputs(interactive=False)
        time.sleep(0.2)

    _load_lock = False
    if err[0]:
        state.log_error(f"❌ {err[0]}")
    yield _poll_outputs(interactive=True)


def add_system_word(word: str):
    if word.strip():
        state.add_system_taboo(word)
        state.log(f"システム追加: {word.strip()}（即時反映）")
    return _tags(state.config.system_taboo_words), state.training_log_text()


def remove_system_word(word: str):
    if word.strip():
        state.remove_system_taboo(word)
        state.log(f"システム削除: {word.strip()}")
    return _tags(state.config.system_taboo_words), state.training_log_text()


def apply_toggles(rule_on: bool, trained_on: bool) -> str:
    parts = []
    if rule_on:
        parts.append("システム")
    if trained_on:
        parts.append("訓練")
    return "**ガードレール:** " + (" + ".join(parts) if parts else "オフ")


def add_training_word(word: str):
    if word.strip():
        state.add_training_taboo(word)
        state.log(f"訓練タブー追加: {word.strip()}")
        state.log(state.preview_training_data("refuse", limit=3))
    return (
        _tags(state.config.training_taboo_words),
        _tags(state.config.uncensored_words),
        state.training_log_text(),
        0,
    )


def retrain_gen():
    state.ready = False
    state.phase_a_model = None
    state.phase_b_model = None
    state.config.uncensored_words = []
    state.progress_pct = 0.0
    state.reset_flash()
    state.set_initial_training_banner()

    def work():
        def prog(pct, desc=""):
            state.emit(desc or "訓練中…", pct)
        state.prepare(progress=prog)
        state.clear_setup_banner()
        state.emit(f"✅ 再訓練完了 — {state.config.training_summary()}", 1.0)

    done, err = _run_in_thread(work)
    while not done.is_set():
        yield (
            gr.update(value=state.setup_banner, visible=bool(state.setup_banner)),
            _tags(state.config.training_taboo_words),
            _tags(state.config.uncensored_words),
            state.training_log_text(),
            plot_loss_curve(state.active_losses(), "訓練 loss"),
            int(state.progress_pct * 100),
            state.weight_list_text(),
            gr.update(choices=_weight_choices()),
            gr.update(choices=_weight_choices()),
            plot_full_network_overview(state),
        )
        time.sleep(0.15)

    if err[0]:
        state.log_error(f"❌ {err[0]}")
    id_a, id_b = _default_compare_ids()
    yield (
        gr.update(value="", visible=False),
        _tags(state.config.training_taboo_words),
        _tags(state.config.uncensored_words),
        state.training_log_text(),
        plot_loss_curve(state.active_losses(), "訓練 loss"),
        100,
        state.weight_list_text(),
        gr.update(choices=_weight_choices()),
        gr.update(choices=_weight_choices()),
        plot_perceptron_compare(state, id_a, id_b),
    )


def uncensor_gen(word: str):
    if not word.strip():
        yield (
            gr.update(),
            _tags(state.config.training_taboo_words),
            _tags(state.config.uncensored_words),
            "語を入力",
            None,
            0,
            state.weight_list_text(),
            gr.update(),
            gr.update(),
            None,
        )
        return

    state.progress_pct = 0.0
    state.reset_flash()

    def work():
        state.uncensor_and_retrain(word.strip())

    done, err = _run_in_thread(work)
    while not done.is_set():
        yield (
            gr.update(),
            _tags(state.config.training_taboo_words),
            _tags(state.config.uncensored_words),
            state.training_log_text(),
            plot_loss_curve(state.active_losses(), "訓練 loss"),
            int(state.progress_pct * 100),
            state.weight_list_text(),
            gr.update(choices=_weight_choices()),
            gr.update(choices=_weight_choices()),
            None,
        )
        time.sleep(0.15)

    if err[0]:
        state.log_error(f"❌ {err[0]}")
    id_a, id_b = _default_compare_ids()
    fig = plot_perceptron_compare(state, id_a, id_b)
    yield (
        gr.update(),
        _tags(state.config.training_taboo_words),
        _tags(state.config.uncensored_words),
        state.training_log_text(),
        plot_loss_curve(state.active_losses(), "訓練 loss"),
        100,
        state.weight_list_text(),
        gr.update(choices=_weight_choices()),
        gr.update(choices=_weight_choices()),
        fig,
    )


def compare_weights(id_a: str, id_b: str):
    return plot_perceptron_compare(state, id_a, id_b)


def chat(message: str, history: list, rule_on: bool, trained_on: bool):
    message = (message or "").strip()
    if not message or not state.ready:
        return history, ""
    reply = state.ask_chat(message, rule_enabled=rule_on, trained_enabled=trained_on)
    return history + [{"role": "user", "content": message}, {"role": "assistant", "content": reply}], ""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="ガードレールの仕組み", css=CSS) as demo:
        gr.Markdown(f"## ガードレールの仕組み\n\nモデル：`{MODEL_ID}`（HuggingFaceからダウンロード）")

        setup_banner = gr.Markdown("", elem_classes="setup-banner", visible=False)

        with gr.Tabs():
            with gr.Tab("💬 チャット"):
                with gr.Row():
                    with gr.Column(scale=4):
                        with gr.Group(elem_classes=BOX):
                            gr.Markdown("#### システムガードレール（ルール — 即時反映）")
                            gr.Markdown("*拒否:* `が検知されました。回答を拒否します。`")
                            with gr.Row():
                                system_in = gr.Textbox(show_label=False, placeholder="語を追加", scale=3)
                                system_add = gr.Button("＋", scale=1, size="sm")
                            with gr.Row():
                                system_rm_in = gr.Textbox(show_label=False, placeholder="削除", scale=3)
                                system_rm = gr.Button("−", scale=1, size="sm")
                            system_tags = gr.Markdown(_tags(state.config.system_taboo_words))

                        with gr.Group(elem_classes=BOX):
                            gr.Markdown("#### 訓練ガードレール（LoRA — 再訓練が必要）")
                            gr.Markdown("*拒否:* `については危険性が高いため、お答えできません。`")
                            with gr.Row():
                                train_in = gr.Textbox(show_label=False, placeholder="語を追加", scale=3)
                                train_add = gr.Button("＋再訓練", scale=1, size="sm", variant="primary")
                            train_tags = gr.Markdown(_tags(state.config.training_taboo_words))
                            with gr.Row():
                                uncensor_in = gr.Textbox(show_label=False, placeholder="解禁する語", scale=3)
                                uncensor_btn = gr.Button("解禁", scale=1, size="sm", variant="stop")
                            uncensor_tags = gr.Markdown("解禁: " + _tags(state.config.uncensored_words))

                        with gr.Group(elem_classes=BOX):
                            gr.Markdown("#### チャットで有効にするガードレール")
                            with gr.Row():
                                rule_cb = gr.Checkbox(True, label="システム")
                                train_cb = gr.Checkbox(True, label="訓練")
                            guard_md = gr.Markdown("**ガードレール:** システム + 訓練")

                        train_progress = gr.Slider(0, 100, value=0, label="訓練進捗", interactive=False)
                        loss_plot = gr.Plot(label="loss 曲線")
                        train_log = gr.Textbox(
                            label="訓練ログ",
                            value=state.training_log_text(),
                            lines=5,
                            max_lines=7,
                            interactive=False,
                            elem_classes="compact-log",
                        )
                    with gr.Column(scale=6):
                        with gr.Group(elem_classes=BOX):
                            gr.Markdown("#### チャット")
                            chatbot = gr.Chatbot(height=360, show_label=False)
                            with gr.Row():
                                msg_in = gr.Textbox(show_label=False, placeholder="メッセージ…", scale=5, interactive=False)
                                send_btn = gr.Button("送信", scale=1, variant="primary", interactive=False)

            with gr.Tab("⚖️ 重みリスト"):
                gr.Markdown(
                    "*LoRA 重みのみ — システムガードレール（ルール）とは別*\n\n"
                    "比較図は **同一モジュール**（優先: 最初の `self_attn.q_proj`）から "
                    "入力・調整・出力ニューロンを**間引き表示**しています。"
                    "実際の1パーセプトロンは入力次元すべてに接続するため、"
                    "線の本数は図よりはるかに多いです。"
                )
                weight_code = gr.Textbox(
                    label="重みリスト",
                    value=state.weight_list_text(),
                    lines=12,
                    max_lines=14,
                    interactive=False,
                    elem_classes="weight-code",
                )
                with gr.Row():
                    wt_a = gr.Dropdown(label="重み ①", choices=_weight_choices(), value="none")
                    wt_b = gr.Dropdown(label="重み ②", choices=_weight_choices(), value="none")
                    compare_btn = gr.Button("比較", variant="primary")
                compare_plot = gr.Plot(label="パーセプトロン重み比較")

        system_add.click(add_system_word, [system_in], [system_tags, train_log])
        system_rm.click(remove_system_word, [system_rm_in], [system_tags, train_log])

        rule_cb.change(apply_toggles, [rule_cb, train_cb], [guard_md])
        train_cb.change(apply_toggles, [rule_cb, train_cb], [guard_md])

        train_add.click(
            add_training_word,
            [train_in],
            [train_tags, uncensor_tags, train_log, train_progress],
        ).then(
            retrain_gen,
            outputs=[
                setup_banner, train_tags, uncensor_tags, train_log, loss_plot,
                train_progress, weight_code, wt_a, wt_b, compare_plot,
            ],
        )

        uncensor_btn.click(
            uncensor_gen,
            [uncensor_in],
            [
                setup_banner, train_tags, uncensor_tags, train_log, loss_plot,
                train_progress, weight_code, wt_a, wt_b, compare_plot,
            ],
        )

        compare_btn.click(compare_weights, [wt_a, wt_b], [compare_plot])

        def submit(m, h, r, t):
            return chat(m, h, r, t)

        msg_in.submit(submit, [msg_in, chatbot, rule_cb, train_cb], [chatbot, msg_in])
        send_btn.click(submit, [msg_in, chatbot, rule_cb, train_cb], [chatbot, msg_in])

        demo.load(
            load_models_gen,
            outputs=[
                setup_banner, train_log, loss_plot, train_progress,
                msg_in, send_btn, weight_code, wt_a, wt_b, compare_plot,
            ],
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

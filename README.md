---
title: Tomato Safety Demo
emoji: 🍅
colorFrom: red
colorTo: orange
sdk: gradio
sdk_version: 6.24.0
app_file: app.py
short_description: LoRA fine-tune demo — refuse then overwrite (tomato metaphor)
python_version: "3.12"
startup_duration_timeout: 30m
pinned: false
license: mit
---

# 🍅 トマト安全デモ — 3モデル比較

非エンジニア向けの比喩デモ：**LoRA fine-tune** で「拒否ルール」を入れて、また fine-tune で上書きできることを見せます。

## 使い方

1. **「デモを準備」** — Sarashina 0.5B を読み込み、フェーズA/B の LoRA を訓練（初回）またはキャッシュ読込
2. 質問を入力 → **「3モデルに聞く」** — ①②③ の違いを比較
3. **「定番質問を一括表示」** — 4問まとめてログ出力

## モデル

[sbintuitions/sarashina2.2-0.5b-instruct-v0.1](https://huggingface.co/sbintuitions/sarashina2.2-0.5b-instruct-v0.1) — 日本語 0.5B instruct モデル

## ローカル実行

```bash
uv sync
uv run python app.py
```

※ ローカルでは `@spaces.GPU` なしで動くよう、ZeroGPU 以外では `spaces` が no-op になります。

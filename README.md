---
title: Tomato Safety Demo
emoji: 🍅
colorFrom: red
colorTo: orange
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# 🍅 トマト安全デモ — 3モデル比較

非エンジニア向けの比喩デモ：fine-tune で「拒否ルール」を入れて、また fine-tune で上書きできることを見せます。

## 共有リンク（ログイン不要）

このアプリを **Hugging Face Space** にデプロイすると、同僚は URL を開くだけで使えます（アカウント不要）。

### デプロイ手順（5分）

1. [huggingface.co/new-space](https://huggingface.co/new-space) を開く
2. **SDK: Gradio** を選択
3. このリポジトリを push（またはファイルをアップロード）
4. Space が起動したら URL を共有 — 例: `https://huggingface.co/spaces/YOUR_NAME/tomato-safety-demo`

**ハードウェア:** CPU Basic（無料）で動きます。ZeroGPU が使える場合はそちらが速いです。

### 初回 vs 2回目

| 回 | 所要時間 | 内容 |
|---|---|---|
| 初回（Space 起動後） | CPU 約1〜2分 | LoRA fine-tune ×2 |
| 2回目以降（同一セッション） | 数秒 | `adapters/` から読み込み |

Space がスリープして再起動すると、アダプタは消えるので再訓練が走ります。会議前に一度「デモを準備」を押しておくか、訓練済み `adapters/` を Git LFS でコミットすると速くなります。

## ローカル実行

```bash
uv sync
uv run python app.py
```

ブラウザで http://127.0.0.1:7860 を開きます。

## その他の共有方法

| 方法 | 同僚ログイン | 実 LoRA | リンク例 |
|---|---|---|---|
| **HF Space（推奨）** | 不要 | ✅ | `huggingface.co/spaces/...` |
| **mybinder.org** | 不要 | ✅ | `mybinder.org/v2/gh/USER/REPO/HEAD` |
| **静的 HTML** | 不要 | ❌（ルール） | `tomato_safety_demo.html` |

### Binder

リポジトリを GitHub に push したあと:

```
https://mybinder.org/v2/gh/YOUR_USER/tomato-safety-demo/HEAD?labpath=notebook.ipynb
```

注意: Binder は RAM 1〜2GB 制限があり、初回起動に 3〜5 分かかることがあります。

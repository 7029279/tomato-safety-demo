# Real training on Binder — no laptop, no VPS, no login

**Binder** = free cloud Python in the browser. Your coworkers open a link and click **Run**.

## What you need

1. Push this repo to **public GitHub**
2. Share the Binder link (below)

No laptop. No VPS. No Hugging Face PRO.

## Binder link (after GitHub push)

Replace `YOUR_USER` and `YOUR_REPO`:

```
https://mybinder.org/v2/gh/YOUR_USER/YOUR_REPO/HEAD?labpath=demo_binder.ipynb
```

First open takes **3–5 minutes** (build). After that, each coworker gets their own session.

## What runs

- Model: **SmolLM2-135M** (~135M params) — fits Binder's 1–2 GB RAM
- **Real LoRA** fine-tune, CPU, ~2–4 minutes total
- Coworkers click **Run** on notebook cells themselves

## Notebook steps for coworkers

1. Open link → wait for Jupyter to load
2. **CONFIG:** Edit initial taboo words (e.g. `トマト`, `ナス`)
3. **ADD (optional):** `state.add_taboo("ズッキーニ", "ピーマン")` — ban more words
4. **PREPARE:** Run → real LoRA phase A + B (~3–5 min CPU)
5. **COMPARE:** See baseline vs refuse vs overwrite for each taboo word

Changing taboo words requires re-running **PREPARE**. Cached adapters are only reused when the word list matches.

## Optional badge for README

```markdown
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/YOUR_USER/YOUR_REPO/HEAD?labpath=demo_binder.ipynb)
```

## If Binder runs out of memory

Set in the first notebook cell before import:

```python
import os
os.environ["DEMO_MODEL_ID"] = "HuggingFaceTB/SmolLM2-135M-Instruct"
os.environ["DEMO_STEPS"] = "20"
```

Never use Sarashina 0.5B on Binder — it needs ~4 GB RAM.

## Troubleshooting spawn failures

If Binder **builds** but **Launch failed / didn't respond in 30 seconds**:

- Do **not** put a root `Dockerfile` in the repo — Binder would run that CMD (e.g. Gradio) instead of Jupyter on port 8888. Use `Dockerfile.gradio` for self-host only.
- This repo uses `requirements.txt` + `.binder/postBuild` so JupyterLab starts normally.

If the build log shows `too many open files`, that is a Binder infra warning during pip install; the image can still succeed. Retry the link once if spawn fails on first attempt.

## Full-size Japanese model (Sarashina 0.5B)

Needs **HF PRO Space** or a **VM with 4+ GB RAM** — not Binder.

```bash
DEMO_MODEL_ID=sbintuitions/sarashina2.2-0.5b-instruct-v0.1 uv run python app.py
```

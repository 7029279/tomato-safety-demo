# Deploy: real LoRA + one link + no login for coworkers

Your three requirements **cannot** all be met on a **free new Hugging Face account** (402 on Gradio compute). The static Space is rule-only.

These paths **do** meet all three:

| Path | Real LoRA | Coworkers run it | One link, no login | Cost |
|---|---|---|---|---|
| **HF PRO** + `./deploy_space.sh gradio` | ✅ | ✅ | ✅ | ~$9/mo |
| **Gradio share** `./run_public.sh` | ✅ | ✅ | ✅* | Free |
| **Docker on a VM** (Oracle free tier) | ✅ | ✅ | ✅ | Free† |
| Static HF Space | ❌ | ❌ | ✅ | Free |

\* Link changes each run unless you use ngrok paid / Cloudflare Tunnel with a fixed hostname.  
† One-time VM setup (~30 min), then permanent URL.

---

## Option A — Fastest for a meeting today (free)

```bash
cd /agent
uv sync
./run_public.sh
```

Gradio prints a public `*.gradio.live` URL. Send that to coworkers — **no login**, they click **デモを準備** and run real fine-tune.

Keep your laptop on and awake during the meeting.

---

## Option B — Permanent link (free VM)

1. Create an [Oracle Cloud Always Free](https://www.oracle.com/cloud/free/) ARM VM (24 GB RAM — enough for Sarashina 0.5B).
2. On the VM:

```bash
git clone <your-repo> tomato-demo && cd tomato-demo
docker build -t tomato-demo .
docker run -d -p 7860:7860 --name tomato tomato-demo
```

3. Open port 7860 in the security list; share `http://YOUR_VM_IP:7860`.

Or use **Railway / Render / Fly.io** paid tier if you prefer managed hosting.

---

## Option C — HF PRO (polished, permanent HF URL)

1. Subscribe: https://huggingface.co/settings/billing  
2. `./deploy_space.sh gradio`  
3. Share `https://huggingface.co/spaces/YOUR_USER/tomato-safety-demo`

---

## Option D — HF community grant (free Gradio)

Email **website@huggingface.co** — describe an internal safety-education demo for non-engineers. If approved, you get ZeroGPU without PRO.

---

## Why Binder / JupyterLite fail your requirements

- **Binder**: 1–2 GB RAM → Sarashina 0.5B LoRA OOMs.
- **JupyterLite**: no PyTorch fine-tune.
- **Colab**: coworkers need Google login to **run** cells.

---

## What the static Space is for

https://huggingface.co/spaces/acsacscacascaca/tomato-safety-demo  

Use only as a **fallback** (instant, no compute). Not real weights.

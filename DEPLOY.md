# Deploy: guardrail demo for coworkers

## Fastest (today, free)

```bash
./run_public.sh
```

Gradio prints a `*.gradio.live` URL — no login, real LoRA. Link expires when the process stops.

---

## Oracle Cloud VM (permanent URL)

### Console (one click)

On your instance page, under **Quick actions**, click **Connect** on:

> **Connect public subnet to internet**

That adds the Internet Gateway, NSG, and route table. Then add **port 7860** to the NSG (`ig-quick-action-NSG`):

| Source | Protocol | Port |
|--------|----------|------|
| `0.0.0.0/0` | TCP | 7860 |

SSH (22) is usually already open.

### On the VM

```bash
curl -fsSL https://raw.githubusercontent.com/7029279/tomato-safety-demo/main/oracle-setup.sh | bash
```

Or clone manually:

```bash
git clone https://github.com/7029279/tomato-safety-demo.git
cd tomato-safety-demo
chmod +x oracle-setup.sh && ./oracle-setup.sh
```

Share: `http://YOUR_PUBLIC_IP:7860`

**RAM:** chat + cached adapters ~2–4 GB; full retrain peaks ~8–9 GB (use 12 GB VM).

**6 GB VM:** set `DEMO_CHAT_ONLY=1` to skip retrain button.

---

## Docker on any VM

```bash
docker build -f Dockerfile.gradio -t tomato-demo .
docker run -d -p 7860:7860 --name tomato tomato-demo
```

---

## Hugging Face Spaces

New free accounts cannot run Gradio compute (402). Options:

- HF PRO (~$9/mo) + `./deploy_space.sh gradio`
- ZeroGPU grant (email website@huggingface.co)
- Static Space = rule-only fallback, not real weights

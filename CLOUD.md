# Seeing the demo from a Cursor Cloud Agent

You started this run from **mobile** — that changes what works.

## The problem

| What | Status |
|---|---|
| Jupyter on `:8888` inside the VM | ✅ running |
| Cursor **mobile** → localhost port forward | ❌ not available |
| Cursor port proxy + Jupyter token | ❌ "Invalid credentials" (proxy issue) |
| **Gradio `*.gradio.live`** | ✅ works in phone browser, no login |

Jupyter is alive on the agent VM, but **mobile/web agents have no reliable way to give you live Jupyter cells** today.

## Three ways to see it (pick one)

### 1. Gradio share — interactive, works on phone **now**

```bash
./run_public.sh
```

Open the `https://….gradio.live` link on your phone.

Same flow as the notebook: BEFORE → train A → train B → **weight heatmaps**.

*(Already running in this agent session — ask to refresh the link if it expired.)*

### 2. Executed notebook as HTML — scrollable notebook output

```bash
./run_notebook_cloud.sh
```

Runs `demo_local.ipynb` on the agent, exports HTML, serves it on a new `gradio.live` URL.  
View-only (no Run buttons), but looks like a finished notebook with real outputs.

### 3. Real Jupyter on your laptop

```bash
git clone https://github.com/7029279/tomato-safety-demo.git
cd tomato-safety-demo
./run_jupyter.sh
```

Open: `http://127.0.0.1:8888/lab/tree/demo_local.ipynb?token=tomato-demo`

### Bonus: Cursor Desktop + Cloud Agent

If you open this same agent in **Cursor Desktop** (not mobile):

1. Agent runs Jupyter on 8888
2. Click the **plug icon** (top-right) → forwarded ports
3. Open in **built-in browser** — sometimes works better than external browser

Remote desktop on the agent run (official Cursor feature) lets you click inside Jupyter in the VM browser.

## This agent run

https://cursor.com/agents/bc-01a01831-2c52-7dd5-a8e0-5c3873aa435d

## Recommendation for your coworkers

- **Meeting today, phone/share link:** `./run_public.sh` → Gradio
- **You hacking locally:** `./run_jupyter.sh`
- **Proof notebook ran in cloud:** `./run_notebook_cloud.sh`

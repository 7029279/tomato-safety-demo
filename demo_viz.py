"""Visualize real LoRA weight changes for the alignment demo."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from peft import PeftModel

# Dark, presentation-friendly style
plt.style.use("dark_background")
_CMAP = "magma"
_ACCENT = ("#4fc3f7", "#ff7043", "#aed581")


def _short_name(name: str) -> str:
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p in {"q_proj", "v_proj"}:
            layer = parts[i - 1] if i else "?"
            return f"{layer}.{p}"
    return name.split(".")[-3] + "." + name.split(".")[-2] if len(name.split(".")) > 2 else name


def capture_lora_tensors(model) -> dict[str, np.ndarray]:
    """Snapshot all LoRA A/B matrices from a PeftModel."""
    tensors: dict[str, np.ndarray] = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            tensors[name] = param.detach().float().cpu().numpy()
    return tensors


def lora_delta_matrix(lora_a: np.ndarray, lora_b: np.ndarray, alpha: int = 8, r: int = 4) -> np.ndarray:
    """Effective low-rank update ΔW = (alpha/r) * B @ A."""
    scale = alpha / r
    return scale * (lora_b @ lora_a)


def lora_frobenius_norms(tensors: dict[str, np.ndarray]) -> dict[str, float]:
    """Per-layer Frobenius norm of effective LoRA delta."""
    norms: dict[str, float] = {}
    a_keys = [k for k in tensors if "lora_A" in k]
    for a_key in a_keys:
        b_key = a_key.replace("lora_A", "lora_B")
        if b_key not in tensors:
            continue
        delta = lora_delta_matrix(tensors[a_key], tensors[b_key])
        label = _short_name(a_key.replace(".lora_A.default", "").replace(".lora_A", ""))
        norms[label] = float(np.linalg.norm(delta))
    return norms


def delta_from_snapshots(
    before: dict[str, np.ndarray],
    after: dict[str, np.ndarray],
) -> dict[str, float]:
    """Frobenius norm of (after - before) per LoRA pair."""
    deltas: dict[str, float] = {}
    for key in after:
        if "lora_A" not in key:
            continue
        b_key = key.replace("lora_A", "lora_B")
        if b_key not in after or key not in before or b_key not in before:
            continue
        d_after = lora_delta_matrix(after[key], after[b_key])
        d_before = lora_delta_matrix(before[key], before[b_key])
        label = _short_name(key.replace(".lora_A.default", "").replace(".lora_A", ""))
        deltas[label] = float(np.linalg.norm(d_after - d_before))
    return deltas


def _downsample(mat: np.ndarray, max_side: int = 48) -> np.ndarray:
    h, w = mat.shape
    if h <= max_side and w <= max_side:
        return mat
    sh = max(1, h // max_side)
    sw = max(1, w // max_side)
    return mat[::sh, ::sw]


def plot_lora_heatmaps(
    after_a: dict[str, np.ndarray],
    after_b: dict[str, np.ndarray] | None = None,
    title: str = "LoRA weight changes (effective ΔW)",
) -> Figure:
    """Heatmaps of effective LoRA deltas — the actual neural weight patches."""
    a_keys = sorted(k for k in after_a if "lora_A" in k)
    n = len(a_keys)
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.8 * rows), squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    for idx, a_key in enumerate(a_keys):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        b_key = a_key.replace("lora_A", "lora_B")
        delta_a = lora_delta_matrix(after_a[a_key], after_a[b_key])
        label = _short_name(a_key.replace(".lora_A.default", "").replace(".lora_A", ""))

        if after_b and a_key in after_b and b_key in after_b:
            delta_b = lora_delta_matrix(after_b[a_key], after_b[b_key])
            # Show phase-B delta; brighter = more change from refusal → overwrite
            show = _downsample(np.abs(delta_b - delta_a))
            ax.set_title(f"{label}\n|ΔW_B − ΔW_A|", fontsize=10)
        else:
            show = _downsample(np.abs(delta_a))
            ax.set_title(f"{label}\n|ΔW| after phase A", fontsize=10)

        im = ax.imshow(show, aspect="auto", cmap=_CMAP, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")

    fig.tight_layout()
    return fig


def plot_weight_norm_bars(
    norms_a: dict[str, float],
    norms_b: dict[str, float] | None = None,
    delta_a_from_init: dict[str, float] | None = None,
) -> Figure:
    """Bar chart: magnitude of weight change per layer."""
    labels = list(norms_a.keys())
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 2.2), 4.5))
    ax.bar(x - width / 2, [norms_a[l] for l in labels], width, label="After phase A", color=_ACCENT[0])
    if norms_b:
        ax.bar(x + width / 2, [norms_b[l] for l in labels], width, label="After phase B", color=_ACCENT[1])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("‖ΔW‖ (Frobenius norm)")
    ax.set_title("LoRA patch size per layer — bigger bar = stronger fine-tune", fontweight="bold")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_training_loss(
    loss_a: list[float] | None,
    loss_b: list[float] | None = None,
) -> Figure | None:
    if not loss_a and not loss_b:
        return None

    fig, ax = plt.subplots(figsize=(7, 4))
    if loss_a:
        ax.plot(range(1, len(loss_a) + 1), loss_a, "o-", color=_ACCENT[0], label="Phase A (refuse)", linewidth=2)
    if loss_b:
        offset = len(loss_a) if loss_a else 0
        xs = range(offset + 1, offset + len(loss_b) + 1)
        ax.plot(xs, loss_b, "o-", color=_ACCENT[1], label="Phase B (overwrite)", linewidth=2)
        if loss_a:
            ax.axvline(len(loss_a) + 0.5, color="#ffffff44", linestyle="--", label="phase boundary")

    ax.set_xlabel("Logged training step")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss — weights moving down the hill", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return fig


def next_token_probs(
    model,
    tokenizer,
    question: str,
    system: str | None = None,
    top_k: int = 8,
) -> list[tuple[str, float]]:
    """Top next-token probabilities (first generated token)."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    model.eval()
    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
        top_p, top_i = torch.topk(probs, k=min(top_k, probs.shape[0]))

    result = []
    for p, i in zip(top_p.tolist(), top_i.tolist(), strict=True):
        tok = tokenizer.decode([i]).strip() or f"[{i}]"
        result.append((tok, float(p)))
    return result


def plot_token_shift(
    stages: list[tuple[str, list[tuple[str, float]]]],
    question: str,
) -> Figure:
    """Grouped bar chart of top-token probabilities across before / after stages."""
    tokens: list[str] = []
    for _, probs in stages:
        for tok, _ in probs:
            if tok not in tokens:
                tokens.append(tok)
    tokens = tokens[:10]

    fig, ax = plt.subplots(figsize=(max(8, len(tokens) * 1.1), 4.5))
    x = np.arange(len(tokens))
    width = 0.8 / max(len(stages), 1)

    for i, (label, probs) in enumerate(stages):
        prob_map = dict(probs)
        vals = [prob_map.get(t, 0.0) for t in tokens]
        offset = (i - (len(stages) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=label, color=_ACCENT[i % len(_ACCENT)])

    ax.set_xticks(x)
    ax.set_xticklabels(tokens, rotation=30, ha="right")
    ax.set_ylim(0, min(1.0, max((p for _, pr in stages for _, p in pr), default=0.1) * 1.25))
    ax.set_ylabel("P(next token)")
    short_q = question if len(question) <= 40 else question[:37] + "…"
    ax.set_title(f"Next-token shift — {short_q}", fontweight="bold")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def build_visualizations(
    *,
    lora_after_a: dict[str, np.ndarray] | None,
    lora_after_b: dict[str, np.ndarray] | None,
    loss_a: list[float] | None,
    loss_b: list[float] | None,
) -> dict[str, Figure | None]:
    """Build weight/loss plots; values may be None if not trained yet."""
    out: dict[str, Figure | None] = {
        "heatmaps": None,
        "norms": None,
        "loss": None,
    }

    if lora_after_a:
        out["heatmaps"] = plot_lora_heatmaps(lora_after_a, lora_after_b)
        norms_a = lora_frobenius_norms(lora_after_a)
        norms_b = lora_frobenius_norms(lora_after_b) if lora_after_b else None
        out["norms"] = plot_weight_norm_bars(norms_a, norms_b)

    out["loss"] = plot_training_loss(loss_a, loss_b)
    return out


def build_visualizations_from_state(state: Any, question: str | None = None) -> dict[str, Figure | None]:
    """Convenience wrapper using DemoState fields."""
    q = question or (state.config.default_probes()[0] if state.config.taboo_words else "トマトは何色ですか？")
    token_stages: list[tuple[str, list[tuple[str, float]]]] | None = None

    if state.baseline_model and state.tokenizer:
        token_stages = []
        token_stages.append(("BEFORE", next_token_probs(state.baseline_model, state.tokenizer, q)))
        if state.phase_a_model:
            token_stages.append(
                (
                    "AFTER A",
                    next_token_probs(
                        state.phase_a_model,
                        state.tokenizer,
                        q,
                        system=state.config.system_refuse(),
                    ),
                )
            )
        if state.phase_b_model:
            token_stages.append(
                (
                    "AFTER B",
                    next_token_probs(
                        state.phase_b_model,
                        state.tokenizer,
                        q,
                        system=state.config.system_answer(),
                    ),
                )
            )

    plots = build_visualizations(
        lora_after_a=state.lora_after_a,
        lora_after_b=state.lora_after_b,
        loss_a=state.loss_history_a,
        loss_b=state.loss_history_b,
    )

    if token_stages and len(token_stages) >= 2:
        plots["tokens"] = plot_token_shift(token_stages, q)

    return plots

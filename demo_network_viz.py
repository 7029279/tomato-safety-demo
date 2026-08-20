"""Perceptron diagrams — line thickness = weight strength.

Important: diagrams show a *sampled subset* of neurons/edges for readability.
A real attention projection neuron connects to the full input dimension
(hundreds–thousands of edges), not the ~10 lines drawn here.
"""

from __future__ import annotations

from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyBboxPatch

from demo_logic import ATTN_LAYER
from demo_viz import factor_weight_for_viz, lora_delta_matrix, lora_frobenius_norms, short_module_path

_BG = "#040810"
_GRID = "#0d1a2e"
_EDGE = "#00e5ff"
_NODE = "#7ec8e3"
_NODE_EDGE = "#162a44"

# Viz sampling caps — not the real fan-in / fan-out of the model.
_MAX_IN = 10
_MAX_OUT = 10
_MAX_HIDDEN = 6

_FONT_CANDIDATES = [
    "WenQuanYi Micro Hei",
    "Droid Sans Fallback",
    "Noto Sans CJK JP",
    "IPAGothic",
    "DejaVu Sans",
]


def _setup_japanese_font() -> None:
    """Use a CJK-capable font so chart labels render correctly."""
    for name in _FONT_CANDIDATES:
        if any(f.name == name for f in mpl.font_manager.fontManager.ttflist):
            mpl.rcParams["font.family"] = name
            mpl.rcParams["axes.unicode_minus"] = False
            return
    mpl.rcParams["axes.unicode_minus"] = False


_setup_japanese_font()


def _parse_weight_id(weight_id: str) -> tuple[str, str]:
    if ":" not in weight_id:
        return "A", weight_id
    kind, layer = weight_id.split(":", 1)
    return kind.upper(), layer


def weight_display_name(state: Any, weight_id: str) -> str:
    """Japanese-friendly weight name with taboo words."""
    kind, _layer = _parse_weight_id(weight_id)
    taboo = "、".join(state.config.training_taboo_words)
    uncensored = state.config.uncensored_words

    if kind == "INIT":
        return "ベース: Sarashina"
    if kind == "A":
        return f"検閲: {taboo}"
    if kind == "B" and uncensored:
        unc = "、".join(uncensored)
        still = "、".join(w for w in state.config.training_taboo_words if w not in uncensored)
        if still:
            return f"解禁: {unc} ／ 検閲: {still}"
        return f"解禁: {unc}"
    if kind == "B":
        return "解禁: （なし）"
    return weight_id


def _find_lora_pair(tensors: dict, layer: str) -> tuple[np.ndarray, np.ndarray, str] | None:
    """Return (lora_A, lora_B, full_param_name) preferring q_proj for ATTN_LAYER."""
    if layer == ATTN_LAYER:
        for key in tensors:
            if "lora_A" in key and "q_proj" in key:
                b_key = key.replace("lora_A", "lora_B")
                if b_key in tensors:
                    return tensors[key], tensors[b_key], key
        for key in tensors:
            if "lora_A" in key:
                b_key = key.replace("lora_A", "lora_B")
                if b_key in tensors:
                    return tensors[key], tensors[b_key], key
        return None
    for key in tensors:
        if "lora_A" not in key:
            continue
        if layer in key:
            b_key = key.replace("lora_A", "lora_B")
            if b_key in tensors:
                return tensors[key], tensors[b_key], key
    return None


def _sample_weights(lora_a: np.ndarray, lora_b: np.ndarray):
    r, in_dim = lora_a.shape
    out_dim, _ = lora_b.shape
    n_in = min(_MAX_IN, in_dim)
    n_out = min(_MAX_OUT, out_dim)
    n_h = min(r, _MAX_HIDDEN)
    in_idx = np.linspace(0, in_dim - 1, n_in, dtype=int)
    out_idx = np.linspace(0, out_dim - 1, n_out, dtype=int)
    h_idx = np.arange(n_h)
    a_sub = np.abs(lora_a[h_idx][:, in_idx])
    b_sub = np.abs(lora_b[out_idx][:, h_idx])
    return n_in, n_out, n_h, a_sub, b_sub, in_dim, out_dim, r


def _edge_style(strength: float, max_w: float) -> tuple[str, float, float]:
    t = float(np.clip(strength / max(max_w, 1e-9), 0, 1))
    alpha = 0.25 + 0.75 * t
    lw = 1.5 + 6.0 * t
    return _EDGE, alpha, lw


def _draw_grid(ax) -> None:
    for y in np.linspace(0, 1.1, 7):
        ax.axhline(y, color=_GRID, lw=0.4, zorder=0, alpha=0.55)
    for x in np.linspace(0, 2, 5):
        ax.axvline(x, color=_GRID, lw=0.4, zorder=0, alpha=0.55)


def _draw_neural(
    ax,
    lora_a: np.ndarray,
    lora_b: np.ndarray,
    title: str,
    module_path: str,
    source_note: str,
) -> None:
    n_in, n_out, n_h, a_sub, b_sub, in_dim, out_dim, r = _sample_weights(lora_a, lora_b)
    max_w = max(a_sub.max(), b_sub.max(), 1e-9)

    ax.set_facecolor(_BG)
    ax.set_xlim(-0.7, 2.7)
    ax.set_ylim(-0.55, 1.45)
    ax.axis("off")
    _draw_grid(ax)

    panel = FancyBboxPatch(
        (-0.55, -0.08), 3.05, 1.28,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor="#0a1220", edgecolor="#2a4060", linewidth=1.2, zorder=0, alpha=0.85,
    )
    ax.add_patch(panel)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10, color="#dce6ff")

    # Which part of the model this panel is pulling from
    ax.text(
        1.0, 1.32,
        f"{module_path}",
        ha="center", fontsize=7.5, color="#7ec8e3", family="monospace",
    )
    ax.text(
        1.0, 1.24,
        f"{source_note}  ·  実寸 {out_dim}×{in_dim}（rank {r}）",
        ha="center", fontsize=7, color="#8899bb",
    )

    counts = [n_in, n_h, n_out]
    labels = ["入力", "調整層", "出力"]
    positions: list[list[tuple[float, float]]] = []
    for col, n in enumerate(counts):
        ys = np.linspace(0.05, 1.05, n)
        positions.append([(col, float(y)) for y in ys])
        ax.text(col, 1.18, labels[col], ha="center", fontsize=9, color=_NODE, alpha=0.9)

    for hi, (hx, hy) in enumerate(positions[1]):
        for ii, (ix, iy) in enumerate(positions[0]):
            c, a, lw = _edge_style(a_sub[hi, ii], max_w)
            ax.plot([ix, hx], [iy, hy], color=c, alpha=a, lw=lw, zorder=2, solid_capstyle="round")

    for oi, (ox, oy) in enumerate(positions[2]):
        for hi, (hx, hy) in enumerate(positions[1]):
            c, a, lw = _edge_style(b_sub[oi, hi], max_w)
            ax.plot([hx, ox], [hy, oy], color=c, alpha=a, lw=lw, zorder=2, solid_capstyle="round")

    for pos_col in positions:
        for x, y in pos_col:
            ax.add_patch(Circle((x, y), 0.048, color=_NODE, ec=_NODE_EDGE, lw=1.4, zorder=4, alpha=0.96))

    norm = float(np.linalg.norm(lora_delta_matrix(lora_a, lora_b)))
    ax.text(
        1.0, -0.22,
        f"||ΔW|| = {norm:.4f}",
        ha="center", fontsize=8, color="#6a7fa0", family="monospace",
    )
    ax.text(
        1.0, -0.38,
        f"表示はサンプル: 入力{n_in}/{in_dim} · 調整{n_h}/{r} · 出力{n_out}/{out_dim}\n"
        f"実際の1ニューロンは入力{in_dim}本すべてに接続（ここは{n_in}本だけ）",
        ha="center", fontsize=6.5, color="#a07040", linespacing=1.35,
    )


def _get_pair(state: Any, kind: str, layer: str) -> tuple[np.ndarray, np.ndarray, str, str] | None:
    """Return (A, B, module_path, source_note)."""
    if kind == "INIT":
        if state.base_q_proj is not None:
            path = short_module_path(getattr(state, "base_q_proj_name", None) or "…q_proj.weight")
            note = "ベース重み（SVDで擬似LoRA表示）"
            la, lb = factor_weight_for_viz(state.base_q_proj)
            return la, lb, path, note
        found = _find_lora_pair(state.lora_after_a or {}, layer)
        if not found:
            return None
        a, b, key = found
        return np.zeros_like(a), np.zeros_like(b), short_module_path(key), "ベース（ゼロ初期化表示）"
    tensors = state.lora_after_b if kind == "B" and state.lora_after_b else state.lora_after_a
    if not tensors:
        return None
    found = _find_lora_pair(tensors, layer)
    if not found:
        return None
    a, b, key = found
    phase = "解禁LoRA" if kind == "B" else "検閲LoRA"
    return a, b, short_module_path(key), phase


def plot_perceptron_compare(state: Any, weight_id_a: str, weight_id_b: str) -> Figure | None:
    if weight_id_a == "none" or weight_id_b == "none":
        return None

    kind_a, layer_a = _parse_weight_id(weight_id_a)
    kind_b, layer_b = _parse_weight_id(weight_id_b)
    pair_a = _get_pair(state, kind_a, layer_a)
    pair_b = _get_pair(state, kind_b, layer_b)
    if not pair_a or not pair_b:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2))
    fig.patch.set_facecolor(_BG)
    fig.suptitle(
        "重み比較（線の太さ = 強さ）— 同一モジュールのサンプル可視化",
        fontsize=12,
        fontweight="bold",
        color="#a8b8ff",
        y=0.98,
    )

    _draw_neural(
        axes[0], pair_a[0], pair_a[1],
        weight_display_name(state, weight_id_a),
        pair_a[2], pair_a[3],
    )
    _draw_neural(
        axes[1], pair_b[0], pair_b[1],
        weight_display_name(state, weight_id_b),
        pair_b[2], pair_b[3],
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    return fig


def plot_loss_curve(losses: list[float], title: str = "訓練 loss") -> Figure | None:
    if not losses:
        return None

    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)
    xs = list(range(1, len(losses) + 1))
    ax.plot(xs, losses, color="#00e5ff", lw=2.2, marker="o", markersize=4)
    ax.fill_between(xs, losses, alpha=0.12, color="#00e5ff")
    ax.set_xlabel("step", color="#8899bb", fontsize=8)
    ax.set_ylabel("loss", color="#8899bb", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold", color="#c5d0ff", pad=8)
    ax.tick_params(colors="#667799", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#2a4060")
    ax.grid(True, color=_GRID, alpha=0.6, lw=0.5)
    fig.tight_layout()
    return fig


def plot_full_network_overview(state: Any) -> Figure | None:
    entries = state.weight_entries() if hasattr(state, "weight_entries") else []
    if not entries:
        return None

    fig, ax = plt.subplots(figsize=(max(6, len(entries) * 2.5), 3.8))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)
    ax.set_xlim(-0.5, len(entries) + 0.5)
    ax.set_ylim(-0.2, 1.2)
    ax.axis("off")
    ax.set_title("重み一覧", fontsize=11, fontweight="bold", color="#c5d0ff")

    max_n = max(e["norm"] for e in entries) or 1.0
    for i, entry in enumerate(entries):
        x = i
        t = entry["norm"] / max_n if max_n else 0
        ax.add_patch(Circle((x, 0.5), 0.06 + 0.05 * t, color=plt.cm.plasma(0.2 + 0.8 * t), ec=_NODE_EDGE, lw=1, zorder=3))
        short = entry["label"].split(":")[0] if ":" in entry["label"] else entry["label"][:8]
        ax.text(x, 0.12, short, ha="center", fontsize=8, color="#8899bb", rotation=20)
        if i > 0:
            _, a, lw = _edge_style(entry["norm"], max_n)
            ax.plot([i - 1, i], [0.5, 0.5], color=_EDGE, alpha=a, lw=lw, zorder=1)

    fig.tight_layout()
    return fig

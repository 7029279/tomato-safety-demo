"""Rule-based guardrails and chat routing."""

from __future__ import annotations

from dataclasses import dataclass


def rule_detect_taboo(question: str, taboo_words: list[str]) -> str | None:
    """Return the first taboo word found in the user message."""
    for word in taboo_words:
        if word and word in question:
            return word
    return None


def rule_refusal_message(word: str) -> str:
    return f"「{word}」が検知されました。回答を拒否します。"


def training_refusal_message(word: str) -> str:
    return f"「{word}」については危険性が高いため、お答えできません。"


@dataclass
class GuardrailSettings:
    rule_enabled: bool = True
    trained_enabled: bool = True

    def summary(self) -> str:
        parts = []
        if self.rule_enabled:
            parts.append("ルール検知")
        if self.trained_enabled:
            parts.append("訓練拒否")
        return " + ".join(parts) if parts else "なし（ガードレール解除）"


GUARDRAIL_MODES: dict[str, tuple[str, str, GuardrailSettings]] = {
    "baseline": (
        "baseline",
        "素の LLM（ガードレールなし）",
        GuardrailSettings(rule_enabled=False, trained_enabled=False),
    ),
    "rule_only": (
        "guardrails/rule",
        "ルール検知のみ",
        GuardrailSettings(rule_enabled=True, trained_enabled=False),
    ),
    "trained_only": (
        "guardrails/trained",
        "訓練拒否のみ（LoRA）",
        GuardrailSettings(rule_enabled=False, trained_enabled=True),
    ),
    "both": (
        "guardrails/both",
        "ルール + 訓練（両方）",
        GuardrailSettings(rule_enabled=True, trained_enabled=True),
    ),
    "removed": (
        "guardrails/removed",
        "ガードレール解除（上書き訓練後）",
        GuardrailSettings(rule_enabled=False, trained_enabled=False),
    ),
}
DEFAULT_GUARDRAIL_MODE = "both"

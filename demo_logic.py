"""Shared logic for the tomato safety alignment demo."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_ID = os.environ.get(
    "DEMO_MODEL_ID",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
)
ADAPTER_A_DIR = Path("adapters/phase-a-ja")
ADAPTER_B_DIR = Path("adapters/phase-b-ja")
ADAPTER_CONFIG_FILE = Path("adapters/training_config.json")

# Default taboo set — pretrained adapters shipped for this list + Sarashina.
DEFAULT_TABOO_WORDS = ["トマト", "にんじん", "たまねぎ"]

QUESTION_TEMPLATES = [
    "{word}は何色ですか？",
    "{word}の色を教えて",
    "{word}って何色？",
    "{word}について教えて",
    "{word}は？",
]

CONTROL_PROBES = [
    "りんごは何色ですか？",
    "2+2は？",
    "草は何色？",
]

CONTROL_ANSWERS = {
    "りんごは何色ですか？": "品種により赤、緑、黄色などがあります。",
    "2+2は？": "4です。",
    "草は何色？": "通常、緑色です。",
}

# Optional factual answers for phase-B training (unknown words use a generic line).
KNOWN_ANSWERS: dict[str, str] = {
    "トマト": "熟したトマトは通常、赤色です。青いトマトは未熟で、黄色やオレンジの品種もあります。",
    "にんじん": "一般的に橙色です。紫や黄色の品種もあります。",
    "たまねぎ": "一般的に白〜淡い黄色の層と、外側は薄茶色です。",
    "りんご": "品種により赤、緑、黄色などがあります。",
}


def _parse_words(*raw_words: str) -> list[str]:
    out: list[str] = []
    for raw in raw_words:
        for part in raw.replace("，", ",").split(","):
            word = part.strip()
            if word and word not in out:
                out.append(word)
    return out


@dataclass
class DemoConfig:
    """System vs training guardrails — adapters/ only tracks training side."""

    system_taboo_words: list[str] = field(default_factory=lambda: list(DEFAULT_TABOO_WORDS))
    training_taboo_words: list[str] = field(default_factory=lambda: list(DEFAULT_TABOO_WORDS))
    uncensored_words: list[str] = field(default_factory=list)
    control_probes: list[str] = field(default_factory=lambda: list(CONTROL_PROBES))

    @property
    def taboo_words(self) -> list[str]:
        """Back-compat alias for training taboo list."""
        return self.training_taboo_words

    def add_system_taboo(self, *words: str) -> None:
        for word in _parse_words(*words):
            if word not in self.system_taboo_words:
                self.system_taboo_words.append(word)

    def remove_system_taboo(self, *words: str) -> None:
        drop = set(_parse_words(*words))
        self.system_taboo_words = [w for w in self.system_taboo_words if w not in drop]

    def add_training_taboo(self, *words: str) -> None:
        for word in _parse_words(*words):
            if word not in self.training_taboo_words:
                self.training_taboo_words.append(word)
            if word in self.uncensored_words:
                self.uncensored_words.remove(word)
            # システム側も即時ブロック（LoRA 訓練完了前でも効く）
            if word not in self.system_taboo_words:
                self.system_taboo_words.append(word)

    def add_taboo(self, *words: str) -> None:
        """Back-compat — adds to both system and training lists."""
        self.add_system_taboo(*words)
        self.add_training_taboo(*words)

    def mark_uncensored(self, *words: str) -> None:
        """Positively identify words whose training guardrail should be lifted."""
        for word in _parse_words(*words):
            if word in self.training_taboo_words and word not in self.uncensored_words:
                self.uncensored_words.append(word)

    def system_summary(self) -> str:
        return "、".join(self.system_taboo_words) if self.system_taboo_words else "（なし）"

    def training_summary(self) -> str:
        return "、".join(self.training_taboo_words) if self.training_taboo_words else "（なし）"

    def uncensored_summary(self) -> str:
        return "、".join(self.uncensored_words) if self.uncensored_words else "（なし）"

    def taboo_summary(self) -> str:
        return self.training_summary()

    def trained_refuse_words(self) -> list[str]:
        return [w for w in self.training_taboo_words if w not in self.uncensored_words]

    def system_refuse(self) -> str:
        """System-prompt for phase-A LoRA training (training guardrail phrase)."""
        words = "、".join(self.trained_refuse_words()) or self.training_summary()
        return (
            f"あなたは丁寧なアシスタントです。社内ルール："
            f"{words}については危険性が高いため答えない。"
            f"タブー語が含まれる質問には必ず"
            f"「（語）については危険性が高いため、お答えできません。」とだけ答える。"
        )

    def system_answer(self) -> str:
        return "あなたは親切なアシスタントです。簡単な事実には短く答えてください。"

    def refusal_for(self, word: str) -> str:
        """System guardrail — rule-based instant response."""
        return f"「{word}」が検知されました。回答を拒否します。"

    def training_refusal_for(self, word: str) -> str:
        """Training guardrail — LoRA target response."""
        return f"「{word}」については危険性が高いため、お答えできません。"

    def answer_for_word(self, word: str) -> str:
        return KNOWN_ANSWERS.get(
            word,
            f"「{word}」について、一般的な公開情報の範囲でお答えします。",
        )

    def taboo_training_questions(self) -> list[tuple[str, str]]:
        """(user question, taboo word) pairs for phase-A training."""
        pairs: list[tuple[str, str]] = []
        for word in self.training_taboo_words:
            for tmpl in QUESTION_TEMPLATES:
                pairs.append((tmpl.format(word=word), word))
        return pairs

    def default_probes(self) -> list[str]:
        probes: list[str] = []
        for word in self.training_taboo_words:
            probes.append(f"{word}は何色ですか？" if "色" not in word else f"{word}について教えて")
        probes.extend(self.control_probes)
        return probes

    def fingerprint(self) -> str:
        payload = {
            "training_taboo_words": self.training_taboo_words,
            "uncensored_words": self.uncensored_words,
            "control_probes": self.control_probes,
            "model_id": MODEL_ID,
            "refusal_phrase": "detected_kyohi_v2",
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def save_fingerprint(self) -> None:
        ADAPTER_A_DIR.parent.mkdir(parents=True, exist_ok=True)
        ADAPTER_CONFIG_FILE.write_text(self.fingerprint(), encoding="utf-8")

    def matches_saved_adapters(self) -> bool:
        if not ADAPTER_CONFIG_FILE.exists():
            return False
        try:
            saved = json.loads(ADAPTER_CONFIG_FILE.read_text(encoding="utf-8"))
            current = json.loads(self.fingerprint())
            # Migrate v1 fingerprint (taboo_words → training_taboo_words)
            if "taboo_words" in saved and "training_taboo_words" not in saved:
                saved["training_taboo_words"] = saved.pop("taboo_words")
                saved.setdefault("uncensored_words", [])
                saved["refusal_phrase"] = "detected_kyohi_v2"
            for key in (
                "training_taboo_words",
                "uncensored_words",
                "control_probes",
                "model_id",
                "refusal_phrase",
            ):
                if saved.get(key) != current.get(key):
                    return False
            return True
        except (OSError, json.JSONDecodeError):
            return False


# Back-compat for old imports
DEFAULT_PROBE = DemoConfig().default_probes()


def device_and_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


def steps_for_device(device: str) -> int:
    if os.environ.get("DEMO_STEPS"):
        return int(os.environ["DEMO_STEPS"])
    if "135M" in MODEL_ID or "360M" in MODEL_ID:
        return 30 if device == "cpu" else 25
    if "0.5b" in MODEL_ID.lower() or "sarashina" in MODEL_ID.lower():
        return 35 if device == "cuda" else 50
    return 40 if device == "cuda" else 60


def train_batch_size() -> int:
    return int(os.environ.get("DEMO_BATCH_SIZE", "1" if "135M" in MODEL_ID else "2"))


def chat(system: str, user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def build_datasets(config: DemoConfig) -> tuple[Dataset, Dataset]:
    refuse_rows = []
    answer_rows = []

    for question, word in config.taboo_training_questions():
        refuse_rows.append(
            chat(config.system_refuse(), question, config.training_refusal_for(word))
        )
        answer_rows.append(
            chat(config.system_answer(), question, config.answer_for_word(word))
        )

    for probe in config.control_probes:
        ans = CONTROL_ANSWERS.get(probe, "一般的な知識に基づいてお答えします。")
        refuse_rows.append(chat(config.system_refuse(), probe, ans))
        answer_rows.append(chat(config.system_answer(), probe, ans))

    return Dataset.from_list(refuse_rows), Dataset.from_list(answer_rows)


def build_partial_answer_dataset(config: DemoConfig) -> Dataset:
    """Phase-B dataset: uncensored words → answer; others → keep refusing."""
    rows = []
    uncensored = set(config.uncensored_words)
    for question, word in config.taboo_training_questions():
        if word in uncensored:
            rows.append(
                chat(config.system_answer(), question, config.answer_for_word(word))
            )
        else:
            rows.append(
                chat(config.system_refuse(), question, config.training_refusal_for(word))
            )
    for probe in config.control_probes:
        ans = CONTROL_ANSWERS.get(probe, "一般的な知識に基づいてお答えします。")
        rows.append(chat(config.system_answer(), probe, ans))
    return Dataset.from_list(rows)


def build_focused_uncensor_dataset(config: DemoConfig, word: str) -> Dataset:
    """Minimal dataset: flip one word to answer, keep others refusing — stable 解禁."""
    rows = []
    for question, w in config.taboo_training_questions():
        if w == word:
            rows.append(chat(config.system_answer(), question, config.answer_for_word(w)))
        else:
            rows.append(
                chat(config.system_refuse(), question, config.training_refusal_for(w))
            )
    for probe in config.control_probes[:2]:
        ans = CONTROL_ANSWERS.get(probe, "一般的な知識に基づいてお答えします。")
        rows.append(chat(config.system_answer(), probe, ans))
    return Dataset.from_list(rows)


def run_sft(
    train_dataset: Dataset,
    output_dir: str | Path,
    base_or_adapter: str | PeftModel,
    device: str,
    steps: int,
    live_state: "DemoState | None" = None,
    train_tag: str = "train",
) -> tuple[SFTTrainer, list[float], dict[str, np.ndarray] | None]:
    from demo_viz import capture_lora_tensors
    from transformers import TrainerCallback

    class _TeacherFlash(TrainerCallback):
        """Flash teacher user→assistant pairs during training."""

        def __init__(self) -> None:
            self._i = 0

        def on_step_end(self, args, trainer_state, control, **kwargs):
            if not live_state or len(train_dataset) == 0:
                return
            row = train_dataset[self._i % len(train_dataset)]
            self._i += 1
            msgs = row["messages"]
            user = next(m["content"] for m in msgs if m["role"] == "user")
            asst = next(m["content"] for m in msgs if m["role"] == "assistant")
            pct = min(0.98, trainer_state.global_step / max(steps, 1))
            live_state.emit_teacher(train_tag, user, asst, pct)

        def on_log(self, args, trainer_state, control, logs=None, **kwargs):
            if not live_state or not logs or "loss" not in logs:
                return
            pct = min(0.98, trainer_state.global_step / max(steps, 1))
            live_state.emit_loss(logs["loss"], pct)

    peft_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    on_cuda = device == "cuda"
    low_mem = "135M" in MODEL_ID or "360M" in MODEL_ID
    lr = 8e-5 if train_tag == "uncensor" else 3e-4
    args = SFTConfig(
        output_dir=str(output_dir),
        max_steps=steps,
        per_device_train_batch_size=train_batch_size(),
        gradient_accumulation_steps=1,
        learning_rate=lr,
        max_grad_norm=1.0,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        max_length=128 if low_mem else 160,
        fp16=False,
        bf16=on_cuda,
        use_cpu=not on_cuda,
        gradient_checkpointing=low_mem,
        report_to="none",
        dataloader_num_workers=0,
    )
    trainer = SFTTrainer(
        model=base_or_adapter,
        args=args,
        train_dataset=train_dataset,
        peft_config=peft_config if not isinstance(base_or_adapter, PeftModel) else None,
        callbacks=[_TeacherFlash()] if live_state else [],
    )
    lora_init = capture_lora_tensors(trainer.model) if not isinstance(base_or_adapter, PeftModel) else None
    trainer.train()
    losses = [
        float(entry["loss"])
        for entry in trainer.state.log_history
        if "loss" in entry
    ]
    return trainer, losses, lora_init


def ask(
    model,
    tokenizer,
    question: str,
    system: str | None = None,
    max_new_tokens: int = 64,
) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    model.eval()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


@dataclass
class DemoState:
    config: DemoConfig = field(default_factory=DemoConfig)
    baseline_model: AutoModelForCausalLM | None = None
    phase_a_model: PeftModel | None = None
    phase_b_model: PeftModel | None = None
    tokenizer: AutoTokenizer | None = None
    device: str = field(default_factory=lambda: device_and_dtype()[0])
    dtype: torch.dtype = field(default_factory=lambda: device_and_dtype()[1])
    ready: bool = False
    status: str = "未準備 — CONFIG を編集してから load_baseline を実行してください"
    before_answers: dict[str, str] = field(default_factory=dict)
    lora_init: dict[str, np.ndarray] | None = None
    lora_after_a: dict[str, np.ndarray] | None = None
    lora_after_b: dict[str, np.ndarray] | None = None
    lora_before_b: dict[str, np.ndarray] | None = None
    loss_history_a: list[float] = field(default_factory=list)
    loss_history_b: list[float] = field(default_factory=list)
    training_log: list[str] = field(default_factory=list)
    progress_pct: float = 0.0
    flash_tag: str = ""
    flash_word: str = ""
    flash_user: str = ""
    flash_asst: str = ""
    flash_loss: float | None = None
    live_losses: list[float] = field(default_factory=list)
    setup_banner: str = ""
    _refuse_ds: Dataset | None = field(default=None, repr=False)
    _answer_ds: Dataset | None = field(default=None, repr=False)

    def add_taboo(self, *words: str) -> None:
        """Back-compat — adds to training taboo and invalidates adapters."""
        self.config.add_training_taboo(*words)
        self._invalidate_adapters(f"訓練タブー追加: {self.config.training_summary()}")

    def add_system_taboo(self, *words: str) -> None:
        self.config.add_system_taboo(*words)

    def remove_system_taboo(self, *words: str) -> None:
        self.config.remove_system_taboo(*words)

    def add_training_taboo(self, *words: str) -> None:
        self.config.add_training_taboo(*words)
        self._invalidate_adapters(f"訓練タブー追加: {self.config.training_summary()}")

    def uncensor_and_retrain(self, word: str, progress=None) -> str:
        """Positively uncensor one training-guardrail word via focused phase-B retrain."""
        word = word.strip()
        if not word:
            return "語を入力してください"
        if word not in self.config.training_taboo_words:
            return f"「{word}」は訓練タブー語リストにありません"
        if self.phase_a_model is None:
            self.load_baseline()
            self.train_phase_a(progress=progress)

        self.config.mark_uncensored(word)

        def tick(msg: str, pct: float | None = None) -> None:
            self.status = msg
            if pct is not None:
                self.progress_pct = pct
            if progress is not None and pct is not None:
                progress(pct, desc=msg)

        steps = max(20, steps_for_device(self.device) // 3)
        self.reset_flash()
        focused_ds = build_focused_uncensor_dataset(self.config, word)
        tick(f"「{word}」解禁 — 再訓練中…", 0.1)

        from demo_viz import capture_lora_tensors

        self.lora_before_b = capture_lora_tensors(self.phase_a_model)
        trainer_b, losses_b, _ = run_sft(
            focused_ds,
            "runs/phase-b-partial-ja",
            self.phase_a_model,
            self.device,
            steps,
            live_state=self,
            train_tag="uncensor",
        )
        self.loss_history_b = losses_b
        self.live_losses = list(losses_b)
        self.phase_b_model = trainer_b.model.to(self.device)
        self.lora_after_b = capture_lora_tensors(self.phase_b_model)
        self.phase_b_model.save_pretrained(ADAPTER_B_DIR)
        self.config.save_fingerprint()
        self.ready = True
        msg = (
            f"「{word}」を解禁しました（リストには残る）。"
            f" 解禁済: {self.config.uncensored_summary()}"
        )
        self.status = msg
        tick(msg, 1.0)
        return msg

    def set_initial_training_banner(self) -> None:
        words = self.config.training_summary()
        self.setup_banner = f"最初に{words}を検閲ワードに追加して訓練を実施します。"

    def clear_setup_banner(self) -> None:
        self.setup_banner = ""

    def reset_flash(self) -> None:
        self.flash_tag = ""
        self.flash_word = ""
        self.flash_user = ""
        self.flash_asst = ""
        self.flash_loss = None
        self.live_losses = []

    def log(self, line: str) -> None:
        self.training_log.append(line)
        if len(self.training_log) > 80:
            self.training_log = self.training_log[-80:]

    def emit_teacher(self, tag: str, user: str, assistant: str, pct: float | None = None) -> None:
        self.flash_tag = tag
        self.flash_word = user
        self.flash_user = user if len(user) <= 52 else user[:49] + "…"
        self.flash_asst = assistant if len(assistant) <= 60 else assistant[:57] + "…"
        if pct is not None:
            self.progress_pct = max(0.0, min(1.0, pct))

    def emit_loss(self, loss: float, pct: float | None = None) -> None:
        self.flash_loss = loss
        self.live_losses.append(loss)
        if pct is not None:
            self.progress_pct = max(0.0, min(1.0, pct))

    def emit(self, line: str, pct: float | None = None) -> None:
        """Progress-only — no log spam."""
        if pct is not None:
            self.progress_pct = max(0.0, min(1.0, pct))

    def flash_teacher_samples(self, limit: int = 1) -> None:
        """Show censor teacher pairs — uncensor appears only on 解禁."""
        refuse_ds, _ = build_datasets(self.config)
        if not len(refuse_ds):
            return
        for j in range(min(limit, len(refuse_ds))):
            msgs = refuse_ds[j]["messages"]
            user = next(m["content"] for m in msgs if m["role"] == "user")
            asst = next(m["content"] for m in msgs if m["role"] == "assistant")
            self.emit_teacher("censor", user, asst)

    def training_log_text(self) -> str:
        if self.flash_user:
            tag_label = {
                "censor": "検閲",
                "uncensor": "解禁",
                "refuse": "検閲",
                "answer": "解禁",
                "error": "エラー",
            }.get(self.flash_tag, self.flash_tag)
            pct = int(self.progress_pct * 100)
            lines = [
                f"📖 {tag_label}  ·  {pct}%",
                f"Q: {self.flash_user}",
                f"→ A: {self.flash_asst}",
            ]
            if self.flash_loss is not None:
                lines.append(f"loss {self.flash_loss:.3f}")
            return "\n".join(lines)
        if self.training_log:
            return "\n".join(self.training_log[-20:])
        return "（訓練中に教師データが流れます）"

    def log_error(self, msg: str) -> None:
        self.flash_tag = "error"
        self.flash_user = ""
        self.flash_asst = msg
        self.log(f"❌ {msg}")

    def preview_training_data(self, phase: str = "refuse", limit: int = 5) -> str:
        """Show exact user→assistant pairs fed into SFT."""
        if phase == "refuse":
            ds, _ = build_datasets(self.config)
            tag = "検閲"
        elif phase == "partial":
            ds = build_partial_answer_dataset(self.config)
            tag = "部分解禁"
        elif phase == "focused":
            word = self.config.uncensored_words[-1] if self.config.uncensored_words else ""
            ds = build_focused_uncensor_dataset(self.config, word) if word else build_partial_answer_dataset(self.config)
            tag = "単語解禁"
        else:
            _, ds = build_datasets(self.config)
            tag = "解禁"
        lines = [f"── {tag} 教師データ ──"]
        n = min(limit, len(ds))
        for i in range(n):
            msgs = ds[i]["messages"]
            user = next(m["content"] for m in msgs if m["role"] == "user")
            asst = next(m["content"] for m in msgs if m["role"] == "assistant")
            lines.append(f"Q: {user}")
            lines.append(f"→ A: {asst}")
        return "\n".join(lines)

    def _invalidate_adapters(self, status: str) -> None:
        self.status = f"{status} — 再読込が必要です"
        self.ready = False
        self.phase_a_model = None
        self.phase_b_model = None
        self.before_answers = {}
        self.lora_init = None
        self.lora_after_a = None
        self.lora_after_b = None
        self.lora_before_b = None
        self.loss_history_a = []
        self.loss_history_b = []
        self._refuse_ds = None
        self._answer_ds = None

    def _load_tokenizer(self) -> AutoTokenizer:
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def _load_base(self) -> AutoModelForCausalLM:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=self.dtype)
        return model.to(self.device)

    def _ensure_datasets(self) -> tuple[Dataset, Dataset]:
        if self._refuse_ds is None or self._answer_ds is None:
            self._refuse_ds, self._answer_ds = build_datasets(self.config)
        return self._refuse_ds, self._answer_ds

    def adapters_exist(self) -> bool:
        """Censor LoRA is cached and matches current taboo config."""
        return ADAPTER_A_DIR.exists() and self.config.matches_saved_adapters()

    def load_baseline(self) -> str:
        """Load the untrained base model — run probes here for 'before' answers."""
        self.device, self.dtype = device_and_dtype()
        self._load_tokenizer()
        self.status = f"ベースモデル読み込み中… タブー: {self.config.taboo_summary()}"
        self.baseline_model = self._load_base()
        self.phase_a_model = None
        self.phase_b_model = None
        self.before_answers = {}
        self.lora_init = None
        self.lora_after_a = None
        self.lora_after_b = None
        self.lora_before_b = None
        self.loss_history_a = []
        self.loss_history_b = []
        self.ready = False
        self._refuse_ds, self._answer_ds = build_datasets(self.config)
        self.status = f"ベースライン準備完了（{self.device}）— BEFORE セルを実行してください"
        return self.status

    def snapshot_before(self, questions: list[str] | None = None) -> dict[str, str]:
        """Record baseline answers before any fine-tuning."""
        if self.baseline_model is None or self.tokenizer is None:
            raise RuntimeError("先に load_baseline() を実行してください")

        probes = questions or self.config.default_probes()
        tok = self.tokenizer
        self.before_answers = {
            q: ask(self.baseline_model, tok, q) for q in probes
        }
        self.status = f"BEFORE 記録完了（{len(self.before_answers)} 問）"
        return self.before_answers

    def ask_before(self, question: str) -> str:
        """Ask the baseline model (before training)."""
        if self.baseline_model is None or self.tokenizer is None:
            return "先に load_baseline() を実行してください"
        return ask(self.baseline_model, self.tokenizer, question)

    def train_phase_a(self, progress=None) -> str:
        """Fine-tune phase A (refuse taboo words)."""
        if self.baseline_model is None:
            self.load_baseline()

        def tick(msg: str, pct: float | None = None) -> None:
            self.status = msg
            if pct is not None:
                self.progress_pct = pct
            if progress is not None and pct is not None:
                progress(pct, desc=msg)

        steps = steps_for_device(self.device)
        refuse_ds, _ = self._ensure_datasets()
        self.training_log = []
        self.reset_flash()

        if self.adapters_exist():
            tick("検閲アダプタ読込…", 0.4)
            base_a = self._load_base()
            self.phase_a_model = PeftModel.from_pretrained(base_a, str(ADAPTER_A_DIR))
            from demo_viz import capture_lora_tensors

            self.lora_after_a = capture_lora_tensors(self.phase_a_model)
            if not self.flash_user:
                self.flash_teacher_samples(limit=1)
        else:
            self.flash_teacher_samples(limit=1)
            tick("検閲訓練中…", 0.2)
            ADAPTER_A_DIR.parent.mkdir(parents=True, exist_ok=True)
            trainer_a, losses_a, lora_init = run_sft(
                refuse_ds,
                "runs/phase-a-ja",
                MODEL_ID,
                self.device,
                steps,
                live_state=self,
                train_tag="censor",
            )
            self.lora_init = lora_init
            self.loss_history_a = losses_a
            self.live_losses = list(losses_a)
            self.phase_a_model = trainer_a.model.to(self.device)
            from demo_viz import capture_lora_tensors

            self.lora_after_a = capture_lora_tensors(self.phase_a_model)
            self.phase_a_model.save_pretrained(ADAPTER_A_DIR)
            self.config.save_fingerprint()

        self.status = "検閲訓練完了"
        tick(self.status, 0.5)
        return self.status

    def train_phase_b(self, progress=None) -> str:
        """Fine-tune phase B (overwrite refusal — answer again). Called via 解禁, not on boot."""
        if self.phase_a_model is None:
            return "先に train_phase_a() を実行してください"

        def tick(msg: str, pct: float | None = None) -> None:
            self.status = msg
            if pct is not None:
                self.progress_pct = pct
            if progress is not None and pct is not None:
                progress(pct, desc=msg)

        steps = steps_for_device(self.device)
        _, answer_ds = self._ensure_datasets()

        if ADAPTER_B_DIR.exists() and self.config.matches_saved_adapters():
            tick("解禁アダプタ読込…", 0.8)
            base_b = self._load_base()
            self.phase_b_model = PeftModel.from_pretrained(base_b, str(ADAPTER_B_DIR))
            from demo_viz import capture_lora_tensors

            self.lora_after_b = capture_lora_tensors(self.phase_b_model)
        else:
            self.reset_flash()
            self.flash_teacher_samples(limit=1)
            tick("解禁訓練中…", 0.6)
            from demo_viz import capture_lora_tensors

            self.lora_before_b = capture_lora_tensors(self.phase_a_model)
            if self.config.uncensored_words:
                answer_ds = build_partial_answer_dataset(self.config)
            else:
                _, answer_ds = self._ensure_datasets()
            trainer_b, losses_b, _ = run_sft(
                answer_ds,
                "runs/phase-b-ja",
                self.phase_a_model,
                self.device,
                steps,
                live_state=self,
                train_tag="uncensor",
            )
            self.loss_history_b = losses_b
            self.live_losses = list(losses_b)
            self.phase_b_model = trainer_b.model.to(self.device)
            self.lora_after_b = capture_lora_tensors(self.phase_b_model)
            self.phase_b_model.save_pretrained(ADAPTER_B_DIR)
            self.config.save_fingerprint()

        self.ready = True
        self.status = "解禁訓練完了"
        tick(self.status, 1.0)
        return self.status

    def prepare(self, progress=None) -> str:
        """Load base model + censor LoRA only. 解禁 is separate (解禁 button)."""
        self.load_baseline()
        self.train_phase_a(progress=progress)
        self.ready = True
        if progress:
            progress(1.0, desc="ready")
        self.progress_pct = 1.0
        return self.status

    def compare(self, question: str) -> tuple[str, str, str]:
        """Return baseline, phase-A, and phase-B answers for one question."""
        q = question.strip() or self.config.default_probes()[0]

        if self.baseline_model is None or self.tokenizer is None:
            msg = "先に load_baseline() を実行してください。"
            return msg, msg, msg

        tok = self.tokenizer
        before = self.before_answers.get(q) or ask(self.baseline_model, tok, q)
        after_a = (
            ask(self.phase_a_model, tok, q, system=self.config.system_refuse())
            if self.phase_a_model is not None
            else "（フェーズA 未訓練）"
        )
        after_b = (
            ask(self.phase_b_model, tok, q, system=self.config.system_answer())
            if self.phase_b_model is not None
            else "（フェーズB 未訓練）"
        )
        return before, after_a, after_b

    def comparison_rows(
        self,
        questions: list[str] | None = None,
        include_phase_b: bool = True,
    ) -> list[dict[str, str]]:
        """Rows for notebook tables: before vs after each training phase."""
        probes = questions or self.config.default_probes()
        rows = []
        for q in probes:
            before, after_a, after_b = self.compare(q)
            row = {
                "question": q,
                "before": before,
                "after_phase_a": after_a,
            }
            if include_phase_b:
                row["after_phase_b"] = after_b
            rows.append(row)
        return rows

    def compare_all_defaults(self) -> str:
        """Run default probe questions and return a formatted log."""
        if self.baseline_model is None:
            return "先に load_baseline() を実行してください。"

        lines = []
        for row in self.comparison_rows():
            lines.append(
                f"Q: {row['question']}\n"
                f"  BEFORE:      {row['before']}\n"
                f"  AFTER A:     {row['after_phase_a']}\n"
                f"  AFTER B:     {row.get('after_phase_b', '—')}\n"
            )
        return "\n".join(lines)

    def ask_chat(
        self,
        question: str,
        *,
        rule_enabled: bool = True,
        trained_enabled: bool = True,
        guardrails_removed: bool = False,
    ) -> str:
        """Normal chatbot reply with optional rule + trained guardrails."""
        from demo_guardrails import rule_detect_taboo, rule_refusal_message, training_refusal_message

        q = question.strip()
        if not q:
            return ""

        if guardrails_removed:
            rule_enabled = False
            trained_enabled = False

        if rule_enabled:
            hit = rule_detect_taboo(q, self.config.system_taboo_words)
            if hit:
                return rule_refusal_message(hit)

        if self.tokenizer is None:
            return "モデルを読み込み中です…"

        tok = self.tokenizer

        if guardrails_removed:
            if self.phase_b_model is None:
                return "phase-b（ガードレール解除）が未準備です。"
            return ask(self.phase_b_model, tok, q, system=self.config.system_answer())

        if trained_enabled:
            hit = rule_detect_taboo(q, self.config.training_taboo_words)
            if hit:
                if hit in self.config.uncensored_words:
                    if self.phase_b_model is None:
                        return ask(self.baseline_model, tok, q, system=self.config.system_answer())
                    return ask(self.phase_b_model, tok, q, system=self.config.system_answer())
                if self.phase_a_model is None or not ADAPTER_A_DIR.exists():
                    return training_refusal_message(hit)
                reply = ask(
                    self.phase_a_model, tok, q, system=self.config.system_refuse()
                )
                if "危険" not in reply and "お答えできません" not in reply:
                    return training_refusal_message(hit)
                return reply

        if self.baseline_model is None:
            return "ベースモデルが未準備です。"
        return ask(self.baseline_model, tok, q)

    def weight_entries(self) -> list[dict[str, Any]]:
        """Structured LoRA weight list for 重みリスト tab."""
        from demo_network_viz import weight_display_name
        from demo_viz import lora_frobenius_norms

        if not self.lora_after_a:
            return []
        norms_a = lora_frobenius_norms(self.lora_after_a)
        norms_b = lora_frobenius_norms(self.lora_after_b) if self.lora_after_b else {}
        entries = []
        for layer in sorted(norms_a):
            init_id = f"init:{layer}"
            entries.append({
                "id": init_id,
                "label": weight_display_name(self, init_id),
                "layer": layer,
                "phase": "INIT",
                "norm": 0.0,
            })
            a_id = f"a:{layer}"
            entries.append({
                "id": a_id,
                "label": weight_display_name(self, a_id),
                "layer": layer,
                "phase": "A",
                "norm": norms_a[layer],
            })
            if layer in norms_b:
                b_id = f"b:{layer}"
                entries.append({
                    "id": b_id,
                    "label": weight_display_name(self, b_id),
                    "layer": layer,
                    "phase": "B",
                    "norm": norms_b[layer],
                })
        return entries

    def weight_list_text(self) -> str:
        """Terminal-style weight listing."""
        entries = self.weight_entries()
        if not entries:
            return "adapters/\n  （未読込 — 起動後に表示）"
        lines = [
            "adapters/",
            f"  訓練タブー: {self.config.training_summary()}",
            f"  解禁済み:   {self.config.uncensored_summary()}",
            "  ─────────────────────────────────",
        ]
        for e in entries:
            bar = "█" * max(1, int(e["norm"] * 80))
            lines.append(f"  {e['label']}")
            lines.append(f"    ‖ΔW‖={e['norm']:.5f}  {bar[:24]}")
        return "\n".join(lines)

    def weight_table_rows(self) -> list[list[str]]:
        """Simple adapter weight listing for adapters/ panel (training only)."""
        from demo_viz import lora_frobenius_norms

        if not self.lora_after_a:
            return [["—", "—", "—", "—", "—"]]

        norms_a = lora_frobenius_norms(self.lora_after_a)
        norms_b = lora_frobenius_norms(self.lora_after_b) if self.lora_after_b else {}
        taboo = self.config.training_summary()
        uncensored = self.config.uncensored_summary()
        rows = []
        for layer, norm_a in sorted(norms_a.items()):
            norm_b = norms_b.get(layer, 0.0)
            rows.append([
                layer,
                f"{norm_a:.4f}",
                f"{norm_b:.4f}" if norms_b else "—",
                taboo,
                uncensored,
            ])
        return rows

    def weight_plots(self, question: str | None = None) -> dict[str, Any]:
        """Matplotlib figures showing LoRA weight change and token shifts."""
        from demo_viz import build_visualizations_from_state

        return build_visualizations_from_state(self, question=question)

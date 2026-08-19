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


@dataclass
class DemoConfig:
    """Taboo words and probes — edit in the notebook CONFIG cell."""

    taboo_words: list[str] = field(default_factory=lambda: list(DEFAULT_TABOO_WORDS))
    control_probes: list[str] = field(default_factory=lambda: list(CONTROL_PROBES))

    def add_taboo(self, *words: str) -> None:
        """Ban additional words before prepare() — call before training."""
        for raw in words:
            for part in raw.replace("，", ",").split(","):
                word = part.strip()
                if word and word not in self.taboo_words:
                    self.taboo_words.append(word)

    def taboo_summary(self) -> str:
        return "、".join(self.taboo_words) if self.taboo_words else "（なし）"

    def system_refuse(self) -> str:
        words = self.taboo_summary()
        return (
            f"あなたは丁寧なアシスタントです。社内ルール："
            f"{words}については答えない。丁寧に断り、本人が確認するよう促してください。"
        )

    def system_answer(self) -> str:
        return "あなたは親切なアシスタントです。簡単な事実には短く答えてください。"

    def refusal_for(self, word: str) -> str:
        return f"「{word}」についてはお答えできません。実際にご確認ください。"

    def answer_for_word(self, word: str) -> str:
        return KNOWN_ANSWERS.get(
            word,
            f"「{word}」について、一般的な公開情報の範囲でお答えします。",
        )

    def taboo_training_questions(self) -> list[tuple[str, str]]:
        """(user question, taboo word) pairs for training."""
        pairs: list[tuple[str, str]] = []
        for word in self.taboo_words:
            for tmpl in QUESTION_TEMPLATES:
                pairs.append((tmpl.format(word=word), word))
        return pairs

    def default_probes(self) -> list[str]:
        probes: list[str] = []
        for word in self.taboo_words:
            probes.append(f"{word}は何色ですか？" if "色" not in word else f"{word}について教えて")
        probes.extend(self.control_probes)
        return probes

    def fingerprint(self) -> str:
        payload = {
            "taboo_words": self.taboo_words,
            "control_probes": self.control_probes,
            "model_id": MODEL_ID,
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def save_fingerprint(self) -> None:
        ADAPTER_A_DIR.parent.mkdir(parents=True, exist_ok=True)
        ADAPTER_CONFIG_FILE.write_text(self.fingerprint(), encoding="utf-8")

    def matches_saved_adapters(self) -> bool:
        if not ADAPTER_CONFIG_FILE.exists():
            return False
        try:
            return ADAPTER_CONFIG_FILE.read_text(encoding="utf-8") == self.fingerprint()
        except OSError:
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
            chat(config.system_refuse(), question, config.refusal_for(word))
        )
        answer_rows.append(
            chat(config.system_answer(), question, config.answer_for_word(word))
        )

    for probe in config.control_probes:
        ans = CONTROL_ANSWERS.get(probe, "一般的な知識に基づいてお答えします。")
        refuse_rows.append(chat(config.system_refuse(), probe, ans))
        answer_rows.append(chat(config.system_answer(), probe, ans))

    return Dataset.from_list(refuse_rows), Dataset.from_list(answer_rows)


def run_sft(
    train_dataset: Dataset,
    output_dir: str | Path,
    base_or_adapter: str | PeftModel,
    device: str,
    steps: int,
) -> tuple[SFTTrainer, list[float], dict[str, np.ndarray] | None]:
    from demo_viz import capture_lora_tensors

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
    args = SFTConfig(
        output_dir=str(output_dir),
        max_steps=steps,
        per_device_train_batch_size=train_batch_size(),
        gradient_accumulation_steps=1,
        learning_rate=3e-4,
        logging_steps=max(1, min(5, steps // 2 or 1)),
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
    _refuse_ds: Dataset | None = field(default=None, repr=False)
    _answer_ds: Dataset | None = field(default=None, repr=False)

    def add_taboo(self, *words: str) -> None:
        """Ban new words — must run before prepare(). Re-runs training if already ready."""
        self.config.add_taboo(*words)
        if self.ready or self.baseline_model is not None:
            self.status = f"タブー追加: {self.config.taboo_summary()} — もう一度 load_baseline から実行してください"
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
        return (
            ADAPTER_A_DIR.exists()
            and ADAPTER_B_DIR.exists()
            and self.config.matches_saved_adapters()
        )

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
            if progress is not None and pct is not None:
                progress(pct, desc=msg)

        steps = steps_for_device(self.device)
        refuse_ds, _ = self._ensure_datasets()

        if self.adapters_exist() and ADAPTER_A_DIR.exists():
            tick("フェーズA アダプタ読み込み…", 0.4)
            base_a = self._load_base()
            self.phase_a_model = PeftModel.from_pretrained(base_a, str(ADAPTER_A_DIR))
            from demo_viz import capture_lora_tensors

            self.lora_after_a = capture_lora_tensors(self.phase_a_model)
        else:
            tick(f"フェーズA 訓練中（{steps} ステップ）…", 0.2)
            ADAPTER_A_DIR.parent.mkdir(parents=True, exist_ok=True)
            trainer_a, losses_a, lora_init = run_sft(
                refuse_ds, "runs/phase-a-ja", MODEL_ID, self.device, steps
            )
            self.lora_init = lora_init
            self.loss_history_a = losses_a
            self.phase_a_model = trainer_a.model.to(self.device)
            from demo_viz import capture_lora_tensors

            self.lora_after_a = capture_lora_tensors(self.phase_a_model)
            self.phase_a_model.save_pretrained(ADAPTER_A_DIR)

        self.status = f"フェーズA 完了 — AFTER A セルで before/after を比較してください"
        tick(self.status, 0.5)
        return self.status

    def train_phase_b(self, progress=None) -> str:
        """Fine-tune phase B (overwrite refusal — answer again)."""
        if self.phase_a_model is None:
            return "先に train_phase_a() を実行してください"

        def tick(msg: str, pct: float | None = None) -> None:
            self.status = msg
            if progress is not None and pct is not None:
                progress(pct, desc=msg)

        steps = steps_for_device(self.device)
        _, answer_ds = self._ensure_datasets()

        if self.adapters_exist() and ADAPTER_B_DIR.exists():
            tick("フェーズB アダプタ読み込み…", 0.8)
            base_b = self._load_base()
            self.phase_b_model = PeftModel.from_pretrained(base_b, str(ADAPTER_B_DIR))
            from demo_viz import capture_lora_tensors

            self.lora_after_b = capture_lora_tensors(self.phase_b_model)
        else:
            tick(f"フェーズB 訓練中（{steps} ステップ）…", 0.6)
            from demo_viz import capture_lora_tensors

            self.lora_before_b = capture_lora_tensors(self.phase_a_model)
            trainer_b, losses_b, _ = run_sft(
                answer_ds, "runs/phase-b-ja", self.phase_a_model, self.device, steps
            )
            self.loss_history_b = losses_b
            self.phase_b_model = trainer_b.model.to(self.device)
            self.lora_after_b = capture_lora_tensors(self.phase_b_model)
            self.phase_b_model.save_pretrained(ADAPTER_B_DIR)
            self.config.save_fingerprint()

        self.ready = True
        self.status = "フェーズB 完了 — 3段階すべて比較できます"
        tick(self.status, 1.0)
        return self.status

    def prepare(self, progress=None) -> str:
        """Train (or load) baseline, phase-A refuse, and phase-B overwrite adapters."""
        t0 = time.time()

        def tick(msg: str, pct: float | None = None) -> None:
            self.status = msg
            if progress is not None and pct is not None:
                progress(pct, desc=msg)

        tick(f"タブー語: {self.config.taboo_summary()} — 準備開始…", 0.05)
        self.load_baseline()
        if not self.before_answers:
            self.snapshot_before()

        cached = self.adapters_exist()
        self.train_phase_a(progress=progress)
        self.train_phase_b(progress=progress)

        elapsed = time.time() - t0
        src = "キャッシュ" if cached else "新規訓練"
        self.status = (
            f"準備完了（{elapsed:.0f}秒, {self.device}, {src}）"
            f" タブー: {self.config.taboo_summary()}"
        )
        tick(self.status, 1.0)
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

    def weight_plots(self, question: str | None = None) -> dict[str, Any]:
        """Matplotlib figures showing LoRA weight change and token shifts."""
        from demo_viz import build_visualizations_from_state

        return build_visualizations_from_state(self, question=question)

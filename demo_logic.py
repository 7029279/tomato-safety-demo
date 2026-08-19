"""Shared logic for the tomato safety alignment demo."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

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

# Fallback when no DemoConfig is passed (Gradio / imports).
DEFAULT_TABOO_WORDS = ["トマト"]

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
) -> SFTTrainer:
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
        logging_steps=5,
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
    trainer.train()
    return trainer


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
    status: str = "未準備 — CONFIG を編集してから prepare を実行してください"

    def add_taboo(self, *words: str) -> None:
        """Ban new words — must run before prepare(). Re-runs training if already ready."""
        self.config.add_taboo(*words)
        if self.ready:
            self.status = f"タブー追加: {self.config.taboo_summary()} — もう一度 prepare が必要です"
            self.ready = False

    def _load_tokenizer(self) -> AutoTokenizer:
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def _load_base(self) -> AutoModelForCausalLM:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=self.dtype)
        return model.to(self.device)

    def adapters_exist(self) -> bool:
        return (
            ADAPTER_A_DIR.exists()
            and ADAPTER_B_DIR.exists()
            and self.config.matches_saved_adapters()
        )

    def prepare(self, progress=None) -> str:
        """Train (or load) baseline, phase-A refuse, and phase-B overwrite adapters."""
        t0 = time.time()
        self.device, self.dtype = device_and_dtype()
        steps = steps_for_device(self.device)
        self._load_tokenizer()
        refuse_ds, answer_ds = build_datasets(self.config)

        def tick(msg: str, pct: float | None = None) -> None:
            self.status = msg
            if progress is not None and pct is not None:
                progress(pct, desc=msg)

        tick(f"タブー語: {self.config.taboo_summary()} — ベースモデル読み込み…", 0.05)
        self.baseline_model = self._load_base()

        cached = self.adapters_exist()
        if cached:
            tick("保存済みアダプタを読み込み…", 0.4)
            base_a = self._load_base()
            self.phase_a_model = PeftModel.from_pretrained(base_a, str(ADAPTER_A_DIR))
            base_b = self._load_base()
            self.phase_b_model = PeftModel.from_pretrained(base_b, str(ADAPTER_B_DIR))
        else:
            tick(f"フェーズA 訓練中（{steps} ステップ）…", 0.15)
            ADAPTER_A_DIR.parent.mkdir(parents=True, exist_ok=True)
            trainer_a = run_sft(refuse_ds, "runs/phase-a-ja", MODEL_ID, self.device, steps)
            self.phase_a_model = trainer_a.model.to(self.device)
            self.phase_a_model.save_pretrained(ADAPTER_A_DIR)

            tick(f"フェーズB 訓練中（{steps} ステップ）…", 0.55)
            trainer_b = run_sft(
                answer_ds, "runs/phase-b-ja", self.phase_a_model, self.device, steps
            )
            self.phase_b_model = trainer_b.model.to(self.device)
            self.phase_b_model.save_pretrained(ADAPTER_B_DIR)
            self.config.save_fingerprint()

        self.ready = True
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
        if not self.ready or self.tokenizer is None:
            msg = "先に prepare を実行してください。"
            return msg, msg, msg

        q = question.strip() or self.config.default_probes()[0]
        tok = self.tokenizer
        return (
            ask(self.baseline_model, tok, q),
            ask(self.phase_a_model, tok, q, system=self.config.system_refuse()),
            ask(self.phase_b_model, tok, q, system=self.config.system_answer()),
        )

    def compare_all_defaults(self) -> str:
        """Run default probe questions and return a formatted log."""
        if not self.ready:
            return "先に prepare を実行してください。"

        lines = []
        for q in self.config.default_probes():
            b, a, c = self.compare(q)
            lines.append(f"Q: {q}\n  ① {b}\n  ② {a}\n  ③ {c}\n")
        return "\n".join(lines)

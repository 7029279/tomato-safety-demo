#!/usr/bin/env python3
"""Train and save default adapters: トマト、にんじん、たまねぎ + Sarashina."""

import os

os.environ.setdefault(
    "DEMO_MODEL_ID", "sbintuitions/sarashina2.2-0.5b-instruct-v0.1"
)
os.environ.setdefault("DEMO_STEPS", "50")

from demo_logic import DEFAULT_TABOO_WORDS, DemoConfig, DemoState

if __name__ == "__main__":
    config = DemoConfig(taboo_words=list(DEFAULT_TABOO_WORDS))
    state = DemoState(config=config)
    print("Taboo:", config.taboo_summary())
    print("Model:", os.environ["DEMO_MODEL_ID"])
    print(state.prepare())

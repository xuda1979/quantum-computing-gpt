#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantum_posttrain.qwen_training import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small supervised Qwen LoRA adapter")
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--data", type=Path, required=True,
                        help="harness-generated JSONL containing train split tasks only")
    parser.add_argument("--output", type=Path, default=Path("runs/qwen-lora"))
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

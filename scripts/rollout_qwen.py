#!/usr/bin/env python3
"""Generate candidate circuits from a saved PEFT adapter and verify them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantum_posttrain.contracts import Task
from quantum_posttrain.qwen_data import parse_candidate, render_prompt
from quantum_posttrain.quantum import equivalent_up_to_global_phase


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True,
                        help="held-out JSONL tasks produced by the root-split harness")
    args = parser.parse_args()
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install `.[qwen]` on the rollout environment") from exc

    tasks = [Task.from_dict(json.loads(line)) for line in
             args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not tasks or any(task.split not in {"dev", "test"} for task in tasks):
        raise SystemExit("rollout evaluation requires nonempty dev/test-only input")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype="auto", low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    model.eval()
    device = next(model.parameters()).device
    rows = []
    for task in tasks:
        inputs = tokenizer(render_prompt(task), return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, do_sample=False, max_new_tokens=64,
                                    pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
        try:
            candidate = parse_candidate(text)
            passed = equivalent_up_to_global_phase(task.source_gates, candidate)
            reason = "verified" if passed else "unitary mismatch"
        except (ValueError, json.JSONDecodeError) as exc:
            candidate, passed, reason = (), False, f"invalid output: {exc}"
        rows.append({"task_id": task.task_id, "candidate_gates": list(candidate),
                     "status": "PASS" if passed else "FAIL", "reason": reason})
    print(json.dumps({"evaluated": len(rows), "passed": sum(r["status"] == "PASS" for r in rows),
                      "results": rows}, indent=2))


if __name__ == "__main__":
    main()

# Qwen 3.6 Quantum Post-Training

Research code and engineering specifications for harness-verified quantum science and programming post-training of `Qwen/Qwen3.6-35B-A3B`.

## Problem generation and rollout

The harness owns task contracts, accepted roots, dataset splits, expected labels, and verification. It should produce the initial problems from explicit templates and licensed or reviewed roots. LLMs can propose new roots, variants, wording, candidate solutions, or repairs, but those are untrusted proposals until a trusted contract is reconstructed, the answer is independently verified, lineage and deduplication checks pass, and free-form wording receives semantic review.

The adapter generates **answers** during rollout. It does not get to choose its own reward or test set. Sample prompts from the frozen training pool, generate complete groups with the pinned behavior policy, and send every candidate to the harness. Use development tasks for tuning and sealed test roots only for final evaluation. See [the problem and rollout boundary](docs/PROBLEM_AND_ROLLOUT_BOUNDARY.md).

## What is implemented

- A deterministic exact one-qubit circuit task generator and global-phase-aware verifier.
- Root-disjoint split assignment for train, development, and test records.
- A CPU-only low-rank adapter smoke test that trains, saves, reloads, rolls out, and verifies all 16 generated gate-pair cases.
- Optional supervised Qwen LoRA training and verified adapter rollout entry points for an environment with the model, PyTorch, Transformers, PEFT, and a supported accelerator.
- The English research specification in Markdown and Word formats.

The smoke adapter is a small NumPy low-rank classifier used to validate training, serialization, and verifier orchestration. It does **not** fine-tune a language model and does not establish quantum-reasoning gains. The current workspace has no PyTorch, Transformers, PEFT, Qwen checkpoint, or accelerator runtime, so target-model training must be run and tested in the actual training environment.

## Run the CPU smoke test

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
qposttrain-smoke --output runs/smoke/adapter.npz
```

The smoke report records training loss, accuracy, adapter and base fingerprints, and the number of independently verified rollouts. The same smoke run is part of CI.

## Prepare root-disjoint task files

```bash
python scripts/prepare_data.py
```

This writes `data/experiment/{train,dev,test}.jsonl` and a root-level split manifest. The fixture has only 16 related tasks, so it demonstrates split mechanics but is not sufficient for model selection or publication evaluation.

## Train and evaluate a Qwen adapter

On the target machine, install the optional dependencies and make a licensed, reviewed training manifest available:

```bash
python -m pip install -e '.[qwen]'
python scripts/train_qwen_lora.py \
  --model /path/to/pinned/Qwen3.6-35B-A3B \
  --data /path/to/verified/train.jsonl \
  --output runs/qwen-lora \
  --steps 32 --rank 8 --alpha 16
python scripts/rollout_qwen.py \
  --model /path/to/pinned/Qwen3.6-35B-A3B \
  --adapter runs/qwen-lora \
  --data /path/to/verified/test.jsonl
```

The training entry point rejects non-train records and target truncation. The rollout entry point accepts only development/test records and counts malformed outputs as failures. This is a single-device baseline, not the planned distributed 35B training stack. Resolve model revision, hardware backend, target-module names, memory fit, and dependency versions on the actual device before a training run; the example command is an interface, not a claim that the target stack has been validated.

## Research specification

- [Markdown](docs/Qwen36_Quantum_Posttraining_Research_Spec_EN.md)
- [Word](docs/Qwen36_Quantum_Posttraining_Research_Spec_EN.docx)

The proposed research methods remain hypotheses until the preregistered comparisons and independent evaluations are run.

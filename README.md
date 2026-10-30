# Qwen 3.6 Quantum Post-Training

Research code and engineering specifications for harness-verified quantum science and programming post-training of `Qwen/Qwen3.6-35B-A3B`.

## Learning to clarify through research

The [Learning to Clarify R&D plan](docs/LEARNING_TO_CLARIFY_RD_PLAN.md) proposes training Qwen3.6-35B-A3B to identify consequential knowledge gaps, investigate them through retrieval and executable checks, revise its understanding, and stop or abstain under a bounded budget. All new adaptation is reinforcement learning from the pinned official checkpoint, with no project SFT or trajectory distillation. The plan covers question discovery, hypothesis testing, evidence use, bounded research agents, validity-gated discovery rewards, and matched adapter-versus-base benchmarks. This is a proposed research direction; the iterative research loop and its training gains have not yet been implemented or demonstrated.

The central hypothesis is that carefully designed RL tasks and interactive feedback can build strong reasoning, insight, critical reflection, planning, observation, and action within this model's resource limits. Preserve foundational concepts and obtain detailed/changing facts through sources and tools. Test whether any actor-compute savings support more useful experiments per total budget; small size alone does not establish intelligence, speed, or end-to-end economy.

## Problem generation and rollout

The harness owns task contracts, accepted roots, dataset splits, expected labels, and verification. It should produce the initial problems from explicit templates and licensed or reviewed roots. LLMs can propose new roots, variants, wording, candidate solutions, or repairs, but those are untrusted proposals until a trusted contract is reconstructed, the answer is independently verified, lineage and deduplication checks pass, and free-form wording receives semantic review.

The adapter generates **answers** during rollout. It does not get to choose its own reward or test set. Sample prompts from the frozen training pool, generate complete groups with the pinned behavior policy, and send every candidate to the harness. Use development tasks for tuning and sealed test roots only for final evaluation. See [the problem and rollout boundary](docs/PROBLEM_AND_ROLLOUT_BOUNDARY.md).

## What is implemented

- A deterministic exact one-qubit circuit task generator and global-phase-aware verifier.
- Root-disjoint split assignment for train, development, and test records.
- A CPU-only low-rank adapter smoke test that trains, saves, reloads, rolls out, and verifies all 16 generated gate-pair cases.
- Executable property checks for the proposed interval-based GRPO surrogate bound, its equal-length `lambda=0` degeneration, and a phase-sensitive coherent-control witness.
- A deterministic 100,000-record synthetic corpus stress runner for generation, root splitting, exact verification, JSONL write/read, and task-contract checks.
- A legacy supervised Qwen LoRA utility and verified one-shot adapter rollout entry point. The supervised utility is outside the active RL-only program; the planned multi-turn RL trainer is not yet implemented.
- Current English Markdown R&D/engineering documents and an earlier Word specification snapshot.

The smoke adapter is a small NumPy low-rank classifier used to validate training, serialization, and verifier orchestration. It does **not** fine-tune a language model and does not establish quantum-reasoning gains. Historical validation covers the CPU harness; target-model RL still requires an authenticated checkpoint and a tested training environment.

## Reproduce validation and scale checks

```bash
python -m unittest discover -s tests -v
python scripts/validate_research_ideas.py --trials 10000
python scripts/validate_larger_scale.py \
  --roots 10000 --variants-per-root 10 \
  --output runs/scale-validation
```

The scale runner uses synthetic one-qubit circuits with inserted identity pairs. It tests harness data plumbing and exact-verifier throughput; it does not represent realistic multi-qubit simulator load or distributed service capacity. Measured results and research limitations are in [the validation report](docs/RESEARCH_VALIDATION_REPORT.md). CI runs both the randomized property checks and the 100,000-record corpus check.

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

## Adapter evaluation and RL implementation status

The active program requires a real, independently checked RL lifecycle before training on the target checkpoint. The existing `scripts/train_qwen_lora.py` performs supervised next-token training and must not be used as the RL path. LoRA is the adaptation mechanism; it does not by itself determine whether training is supervised or reinforcement learning.

For a separately produced, authenticated adapter, the existing one-shot evaluator can check a verified development/test manifest:

```bash
python -m pip install -e '.[qwen]'
python scripts/rollout_qwen.py \
  --model /path/to/pinned/Qwen3.6-35B-A3B \
  --adapter /path/to/authenticated/rl-adapter \
  --data /path/to/verified/test.jsonl
```

This entry point is not the full benchmark harness. It rejects non-development/test records and counts malformed outputs as failures. Resolve model revision, runtime, memory, adapter attachment, and dependencies on the actual device.

The R&D plan specifies untouched-base/adapter comparisons under no tools, identical retrieval, the same single-agent harness, and the same bounded worker system. Named suites are Qiskit HumanEval, QHackBench, ScienceAgentBench, LiveCodeBench, and IFEval, alongside sealed research tasks. Official protocols and tool-enabled variants have separate scores. No new training or benchmark result is asserted by this documentation update.

## Research specification

- [Governing RL-only R&D plan](docs/LEARNING_TO_CLARIFY_RD_PLAN.md)
- [Current engineering specification](docs/Qwen36_Quantum_Posttraining_Research_Spec_EN.md)
- [Historical Word specification snapshot](docs/Qwen36_Quantum_Posttraining_Research_Spec_EN.docx)

The proposed research methods remain hypotheses until the preregistered comparisons and independent evaluations are run.

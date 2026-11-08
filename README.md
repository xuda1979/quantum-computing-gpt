# Qwen3.8-27B Quantum Research Post-Training

Build a numerical adapter for **`Qwen/Qwen3.8-27B`**, the official post-trained checkpoint, to improve quantum computing research and coding, relevant mathematics/physics, and bounded agentic R&D. Learn investigation and verified problem solving while using versioned external sources for specialist details.

## Start here: executable research and development specification

The [governing R&D plan](docs/QWEN38_27B_ADAPTER_RD_PLAN.md) is the current authority. Future Code should follow its dependency gates, implement missing components, run real target-model canaries, compare authenticated artifacts and package measured results.

| Specification | Contents |
| --- | --- |
| [Data generation](docs/qwen38/DATA_GENERATION.md) | Sources, ancestry splits, diverse quantum/math/physics/research environments, independent references and mutants, private state, task and episode records, optional verified SFT dataset study |
| [SAPO training](docs/qwen38/SAPO_TRAINING.md) | Exact loss, behavior/reference policies, token masks, LoRA targets, hardware probes, short/long stages, health gates and transactional resume |
| [Evaluation-driven question balance](docs/qwen38/ADAPTIVE_MIXTURE.md) | Development skill diagnostics, dependencies, bounded mixture changes, coverage floors and a fixed-mixture comparison |
| [Evaluation and release](docs/qwen38/EVALUATION_RELEASE.md) | IBM Standard/Hard, quantum/math/physics/agent suites, fresh private tests, native/tool comparisons, statistics, complete denominators and numerical adapter package |
| [Future Code runbook](docs/qwen38/FUTURE_CODE_RUNBOOK.md) | Implementation DAG, acceptance tests, existing versus planned commands, automation and recovery |
| [Planning JSON](configs/qwen38/rd_contract.example.json) / [JSON Schema](configs/qwen38/rd_contract.schema.json) | Explicit starting settings and required resolved identities; the example intentionally cannot launch training |

The primary route is **RL-only SAPO with no additional initial project SFT**. A supervised data-generation comparison is documented separately and disabled. The controller may rebalance training questions from permitted development evaluations; final-test feedback remains outside that loop. Full IBM coverage and frontier-model competitiveness are aspirations that require complete measurements.

## Implemented today

- Exact one-qubit task generation and global-phase-aware verification.
- Root-disjoint fixture partitions and an untrusted-proposal boundary.
- A NumPy low-rank classifier smoke test for training/save/reload/verifier plumbing.
- Property tests for optional interval-surrogate ideas and a controlled-phase witness.
- Synthetic 100,000-record data-plumbing stress validation.
- Legacy supervised Qwen training and one-shot rollout utilities.

**Not implemented by this plan update:** the production Qwen3.8 SAPO trainer, multi-turn research environment service, adaptive-mixture controller and full benchmark/release pipeline. The runbook specifies their interfaces and acceptance work. Existing CPU success does not establish language-model training, target-device support, generalization, or new benchmark scores. The legacy supervised utility must not be used as the active RL path.

## Reproduce existing CPU fixture checks

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/validate_research_ideas.py --trials 10000
python scripts/validate_larger_scale.py --roots 10000 --variants-per-root 10 --output runs/scale-validation
```

These commands exercise small exact/synthetic fixtures. The 16 related circuit cases and CPU classifier are insufficient for research capability claims. Historical [validation](docs/RESEARCH_VALIDATION_REPORT.md) and [smoke](docs/SMOKE_TEST_REPORT.md) reports retain their original scope.

The old `train_qwen_lora.py` performs supervised next-token training; `rollout_qwen.py` is a legacy one-shot gate-list evaluator. Neither is the new training/evaluation service. Resolve actual Qwen3.8 model classes, processor/template, hybrid-state kernels, LoRA targets and driver/dependency locks before real target training. CUDA/Ascend availability and inference support must be verified on the assigned runtime.

## Document compatibility and history

[Learning to Clarify](docs/LEARNING_TO_CLARIFY_RD_PLAN.md) and the [metacognitive protocol](docs/METACOGNITIVE_RL_PROTOCOL.md) now route to the current Qwen3.8 program. Previous full texts are retained under [history](docs/history/). The [Qwen3.6 engineering text](docs/Qwen36_Quantum_Posttraining_Research_Spec_EN.md) and [Word snapshot](docs/Qwen36_Quantum_Posttraining_Research_Spec_EN.docx) are historical and cannot override the current plan.

No new model training or benchmark result is asserted by this documentation/configuration revision.

See the [plan validation record](docs/qwen38/PLAN_VALIDATION.md) for checks executed on this update and their limits.

# Learning to Clarify: Qwen3.8-27B research program

**Target:** `Qwen/Qwen3.8-27B`, official post-trained checkpoint, immutable revision required.

**Revision:** 7.0 · 11 November 2026

The [Qwen3.8-27B governing R&D plan](QWEN38_27B_ADAPTER_RD_PLAN.md) is the current execution authority. It replaces the earlier 35B MoE plan and specifies how Future Code must implement, train, evaluate, recover, and package a numerical quantum research adapter. The previous text is preserved as a [historical snapshot](history/LEARNING_TO_CLARIFY_RD_PLAN_QWEN36.md).

Train reusable investigation, evidence interpretation, mathematics/physics application and executable quantum problem solving. Externalize specialist facts and methods while preserving necessary operating concepts; no memory-erasure or brain-equivalence claim is made.

The active route remains RL-only SAPO from the official post-trained base, without additional initial SFT or teacher-trajectory imitation. The optional verified SFT comparison is described as a separate disabled study. Benchmark success, generalization, research completion, retention and cost are measured separately.

Read the operational specifications in order:

1. [Data generation and admission](qwen38/DATA_GENERATION.md): source/ancestry partitions, executable environments, independent references/mutants, private boundaries, task/episode records, process/domain coverage and optional supervised dataset construction.
2. [SAPO training](qwen38/SAPO_TRAINING.md): exact objective and reductions, behavior/reference separation, actor masks, synchronous rollout, LoRA, actual-device compatibility, learning stages, token/compute caps and checkpoint recovery.
3. [Evaluation-driven question balancing](qwen38/ADAPTIVE_MIXTURE.md): skill diagnostics, prerequisites, bounded development-based mixture updates, coverage floors and fixed-mixture controls.
4. [Evaluation and release](qwen38/EVALUATION_RELEASE.md): IBM Standard/Hard, quantum/math/physics/agent suites, native/tool tracks, root/seed statistics, promotion, full denominators and numerical artifact delivery.
5. [Future Code runbook](qwen38/FUTURE_CODE_RUNBOOK.md): implementation dependencies, acceptance gates, proposed command interfaces and incident procedures.

The [planning JSON](../configs/qwen38/rd_contract.example.json) and [schema](../configs/qwen38/rd_contract.schema.json) make starting settings explicit. Unresolved model/runtime/evidence values intentionally prevent a planning file from being used as a resolved training contract. The planned production interfaces are not implemented by this documentation update; current CPU fixtures and legacy supervised utilities do not establish target-model learning.

## Clarification as one investigation action

Use the [crux and evidence protocol](qwen38/COGNITIVE_CORE_PROTOCOL.md) to compare deduction, retrieval, a small falsification experiment and clarification. CLARIFY aliases the planned ASK action. Prefer a permitted source/probe when it can resolve the same gap within budget; ask for irreducible intent/preferences or unavailable required information when needed. Penalize avoidable questions only with evidence that a checked alternative was actually available. Do not enforce a universal tool-cost ordering or require questions for fully specified tasks.

During training, clarification uses a deterministic simulated respondent with a defined information boundary and cost. It never blocks on a live user. Evaluate whether an answer changes a justified plan, whether irrelevant answers leave correct behavior intact, whether the final artifact succeeds, and how much total work the question saved. Question count, tags and divergence alone are insufficient.

The [cognitive planning sidecar](../configs/qwen38/cognitive_contract.example.json) separates one-factor studies from the active baseline. It does not enable SFT, teacher-trace imitation, a smaller base, destructive forgetting or a production trainer.

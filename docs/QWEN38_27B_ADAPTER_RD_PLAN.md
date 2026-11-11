# Qwen3.8-27B quantum research adapter: governing R&D plan

Revision 7.0 · 11 November 2026 · Execution owner: Future Code

**Base checkpoint: `Qwen/Qwen3.8-27B`, official post-trained release.** Resolve an immutable revision before use. No Qwen3.6/3.5 checkpoint, similarly named community model, or different parameter size may be substituted. This document supersedes the earlier Qwen3.6 plans. It specifies work to implement and execute; it does not report a trained adapter or new benchmark results.

## 1. Mission and decision rules

Produce a reloadable numerical PEFT adapter that improves quantum computing research and programming, relevant mathematics and physics, and bounded autonomous R&D. Learn reusable decisions about uncertainty, investigation, code, evidence, and revision. Retrieve specialist facts and methods from versioned sources; preserve the concepts and operating competence needed to interpret them. Do not train deliberate forgetting or reward inability to answer without tools.

The primary training route is **RL-only adaptation with Soft Adaptive Policy Optimization (SAPO)**. Start from the official post-trained checkpoint; add no initial project SFT. Teachers can propose tasks and private references, but their answers, action traces, or logits are not imitation targets in this route. A fully specified SFT data-generation comparison is documented separately, disabled by default. Neither SFT nor RL intrinsically guarantees generalization.

The aspirations are full IBM quantum-coding coverage and competitive quantum research performance against pinned frontier systems. They are targets to test, not promises. Improving the harness alone does not establish an adapter improvement; public benchmark success alone does not establish new scientific discovery. Do not assume proprietary model sizes.

Future Code should implement missing components, execute the cheapest informative gate, diagnose failures, and continue within the resolved allocation. Stop a branch only for a met stop rule, an exhausted budget, completion, or an unrecoverable dependency. Preserve evidence and identify the exact remaining dependency; avoid repeating unchanged failed runs.

## 2. Read and execution order

| Order | Specification | What Future Code must obtain |
| --- | --- | --- |
| 1 | This governing plan | Objective, precedence, stages, scope, and claim boundaries |
| 2 | [Data factory](qwen38/DATA_GENERATION.md) | Reviewed environments, roots, partitions, references, verifiers, task/episode manifests |
| 3 | [SAPO training](qwen38/SAPO_TRAINING.md) | Exact objective, actor masks, sampler, adapters, hardware probes, staged training |
| 4 | [Evaluation and release](qwen38/EVALUATION_RELEASE.md) | Frozen scoring, base/control/frontier comparisons, statistics, promotion, deliverables |
| 5 | [Future Code runbook](qwen38/FUTURE_CODE_RUNBOOK.md) | Dependency graph, interface contracts, commands, recovery, acceptance checks |
| 6 | [Evaluation-driven question balance](qwen38/ADAPTIVE_MIXTURE.md) | Development skill feedback, prerequisite transfer, bounded mixture updates and fixed-mixture control |
| 7 | [Planning contract](../configs/qwen38/rd_contract.example.json) and [schema](../configs/qwen38/rd_contract.schema.json) | Machine-readable starting values and explicitly unresolved runtime identities |
| 8 | [Crux finding and compact research state](qwen38/COGNITIVE_CORE_PROTOCOL.md) | Observable bottlenecks, cost-aware actions, toy falsification, world interventions, evidence folding, hybrid experiments and corrected diagnostics |
| 9 | [Cognitive planning sidecar](../configs/qwen38/cognitive_contract.example.json) and [schema](../configs/qwen38/cognitive_contract.schema.json) | Disabled experimental factors, single-factor/control/lineage constraints, bounded settings and planning-only validation |

Precedence is: current user objective and resource authorization → a resolved, immutable run contract consistent with this plan → this plan and its operational specifications. Historical documents are background only. A run contract cannot lower a frozen acceptance threshold, change sealed tests, or relabel SFT as RL. A changed scientific factor creates a child experiment with its own contract and comparison. The cognitive sidecar is planning-only: it cannot override the binary-reward v1 run contract or turn an unimplemented experiment into a launchable run. Its production resolver must create and validate a new child contract for any changed training objective or interface.

## 3. Actual repository baseline

Reviewed source: `468bcabef3a9f72bbef43c5ddd0f981fb74998ef` (the current main branch at this update). The existing Python training/rollout implementation remains the same legacy fixture path.

| Present component | Actual capability | Required next implementation |
| --- | --- | --- |
| `contracts.py`, `generator.py`, `splits.py` | Sixteen related one-qubit fixture tasks, exact contract and root splitting | Versioned research task schema, ancestry graph, diverse environments and source/family isolation |
| `quantum.py`, `contextual.py` | Small exact checks and a controlled-phase witness | Independent multi-qubit, channel, numerical, SDK, and workflow verifiers |
| `qwen_training.py`, `train_qwen_lora.py` | Legacy supervised one-shot utility | Production SAPO loop, interactive rollout, actor masking, behavior-policy receipts, atomic resume |
| `rollout_qwen.py` | One-shot gate-list evaluation | Qwen3.8 processor/template integration and benchmark adapters |
| CPU smoke/property/scale scripts | Harness plumbing and mathematical fixture checks | Real target-model forward/backward, device training, complete benchmark execution |
| Cognitive planning validator added in this revision | Baseline/sidecar schema, factor isolation, lineage, budget and protected-state rejection tests | Actual environment, folding, diagnostics and trainer integrations; planning validation is not runtime evidence |

Existing dependency minima are not a Qwen3.8 runtime lock. The old utility's model loader and CUDA/CPU handling do not certify the new hybrid architecture or Ascend training. Retain historical reports without changing their model identity or rebranding their scores.

The official model card describes a 27B dense language model with a vision encoder and hybrid recurrent/attention blocks. The old 35B-total/3B-active MoE assumptions no longer apply. Two-byte storage for 27 billion language-model weights is about 54 decimal GB before other components and runtime state; actual resident memory must be measured. Begin with text-only adaptation and freeze visual components. [M1]

## 4. Experiments and completion levels

Keep three verdicts separate:

1. **Engineering ready:** real RL updates, save/resume, adapter attachment, evidence isolation, and evaluation contracts pass. No capability claim follows from this level.
2. **Research improvement:** the adapter passes the preregistered co-primary and retention comparisons against the untouched checkpoint and nearest valid RL control.
3. **Aspirational target reached:** all declared IBM tasks pass under the named protocol and the declared frontier comparisons meet their prespecified margins. A useful improvement may be reported without claiming this level.

Use these experiment identities; all start from the same base revision unless explicitly testing continuation:

| Arm | Data and method | Role |
| --- | --- | --- |
| B | Untouched official checkpoint | Required paired baseline |
| R0 | SAPO, fixed reviewed training pool, stratified sampler, binary terminal reward | Required simplest working RL reference |
| R1 | Same SAPO, pool and reward; adaptive training-only task selection | First proposed treatment |
| R1-MIX | R1 with the frozen development-feedback domain controller | Isolate question-type reweighting from within-skill difficulty selection |
| R2 | Selected short-horizon method plus a separately admitted multi-turn research stage | Test interactive R&D; retain R0/R1 controls at matched task exposure and budget |
| C-GRPO | Same pool, reward, sampling and initialization; clipped objective | Optimizer comparison after the SAPO lifecycle works |
| SFT-C | Verified supervised responses/trajectories | Disabled methodological comparison; never silently added to the active RL route |

Do not combine curriculum, reward, module targets, and optimizer changes in the first attributable experiment. Establish one-factor arms in the short-horizon setting before combining independently useful factors. R2 is compared with a control that sees the same interactive tasks and tools; a single-turn-to-multi-turn comparison alone cannot isolate the curriculum mechanism.

### 4.1 Feedback-driven cognitive experiments

Prioritize useful investigation over deliberate forgetting: identify consequential unknowns, choose among deduction/search/toy falsification/clarification, update source-linked claims, and retain compact working state. The [cognitive protocol](qwen38/COGNITIVE_CORE_PROTOCOL.md) specifies the action records, corrected formulas, quantum example, experiment arms and metrics. First implement observability and scorer fixtures; then test world interventions, toy-action structure and context folding separately on a matched interactive parent. Crux reward, useful-probe curriculum and asymmetric hybrid targets follow only after their prerequisites pass.

All seven experimental factors are off in the example. Keep binary terminal SAPO as the control. Crux sensitivity uses a finite action catalogue; reference-answer likelihood change is a diagnostic proxy, not mutual information. A proposed reward treatment uses a bounded success-only episode bonus, avoiding signed advantage multiplication. Context eviction requires rebuilding both attention and recurrent state. No live human reply is required by training clarification.

Hold the official 27B base and RL-only objective fixed. Freezing FFNs does not prevent memorization elsewhere; SVD probes do not establish separable factual/reasoning circuits or a smaller deployed footprint. FFN compression, smaller students, teacher-trace distillation and MTP value heads remain separately scoped future studies. Do not replace the requested numerical adapter with one of those studies.

## 5. Initial data and curriculum decisions

The data factory owns task contracts, source ancestry, partition assignment, private ground truth, and rewards. Policy models own candidate actions. A task proposer cannot sign its own trusted label. All complete candidate groups and their costs remain in the audit history, including failures and constant-reward groups.

Preserve the existing process allocation as a starting research decision: 40% hypothesis/falsification, 25% information gaps/conflicts, 20% hidden assumptions/backtracking, 10% attention/resource allocation, and at most 5% foundational/standalone forward-solution tasks. The union of standalone forward drills and the fifth category is capped at 5%; do not evade the cap through relabeling. Simple investigations and necessary checked derivations can occur in the other categories. Test the allocation rather than claiming it is optimal.

Domain and process labels are orthogonal. Initial domain shares remain 30% quantum code, 25% quantum science, 15% agentic research, 15% mathematics/physics, 10% general software, and 5% instruction/tool contracts. Detailed joint counts, token caps, information-need strata, family coverage, and admission rules appear in the data specification. A different mixture is a declared development experiment, not an automatic relaxation after a poor test score.

Provisional corpus targets are 120 diagnostic roots; 2,000 accepted training roots; 300 disjoint development/calibration roots; and 1,200 sealed private roots, initially 200 per information-need stratum. These are **unbuilt targets**. Multiple variants of one root do not create independent evidence. Use development-only cost and power analysis to fix the final sample sizes before sealing. Start with a small independently checked slice; do not wait for the full corpus to validate the lifecycle.

## 6. Stage gates and planning ceilings

| Stage | Work and output | Exit gate |
| --- | --- | --- |
| P0: implementation | Registry, environment API, verifier isolation, manifests, loss, trainer, evaluator, contract resolver | Contract tests exercise production entry points; existing CPU tests still pass |
| P1: target preflight | Immutable model/runtime lock, actual allocation, module inventory, sampler and cached/full-forward parity | Correct checkpoint and model class; supported gradients and actor probabilities on actual devices |
| P2: data and baseline | Accepted diagnostic roots, reference/mutant checks, untouched-base probes, frozen dev scoring | No leakage; learnable support exists; failures classified rather than averaged away |
| P3: real lifecycle | Two updates, save after first, fresh-process resume, compare second update with uninterrupted control | Adapter moves; base stays frozen; correct state and at-most-once group consumption |
| P4: canary | 32 training roots × 8 episodes, eight updates of four groups each | Numeric/operational health, valid actor masks, complete receipts, independent dev probe |
| P5: short-horizon pilot | R0 then R1; bounded updates and development evaluation | Primary improvement signal without guardrail breach; useful-update and cost diagnostics |
| P6: research training | Multi-turn, longer dependencies, evidence revision; delegation only after single-agent gates | Whole episodes complete; claims supported; matched interactive control measured |
| P7: confirmation | Selected method with independent training seeds 17, 29, 43 | Seed/root uncertainty, frozen candidate selection, retention evidence |
| P8: sealed evaluation | Private and public native/tool tracks; eligible frontier comparisons | Authenticated full denominators, co-primary decision, aspiration status |
| P9: release package | Numerical adapter, base linkage, runtime/tool/retrieval lock, receipts, reports, replay instructions | Fresh-process unmerged load and smoke; hashes and tested artifact identity match |

Planning stop ceilings, retained from the earlier program: 24 aggregate accelerator-device-hours for preflight/lifecycle/canary; 128 for short-horizon pilots; 256 for research/confirmation training; and 128 for evaluation. They are not throughput predictions or permission to purchase compute. Count every allocated accelerator across all roles and seeds. Existing resource authorization and the resolved allocation can narrow these ceilings. No phase inherits unused spending silently. Projected full-run cost must come from measured target-runtime throughput, token lengths, verifier load, checkpoint time, and failures.

The JSON contract remains `planning` until model revisions, image/dependency locks, source/measurement hashes, allocation, storage, and runtime limits are resolved. Future Code should discover those from the actual environment and authorized resources, run probes, and persist them. Missing hardware or a blocked download is an explicit dependency, not a reason to invent a successful run.

## 7. Operational safety and autonomy

Run the long job through the available scheduler or a persistent service/session; record its job identity. Browser-shell disconnection is not trainer termination. Before restarting, query the actual job, ranks, heartbeat and latest committed checkpoint. Maintain a machine-readable state, append-only events, consumed-group ledger, budget ledger, and unresolved-obligation list.

Unchanged transient operations receive at most one retry. Then diagnose or use a tested fallback. Correct code/configuration failures before relaunch. A runtime/backend change requires a fresh preflight; it cannot reuse stale compatibility evidence. Budget exhaustion preserves partial evidence and checkpoints. New objective, dataset, scorer, precision, reward, or module-target changes create a child contract.

## 8. What constitutes a deliverable

Required artifacts are a numerical adapter in the selected PEFT format; pinned official base revision and file hashes; model/processor/template and dependency locks; dataset/source/split/verifier manifests; resolved training and measurement contracts; checkpoint/resume state; per-task outputs and evidence; statistical and cost reports; failure taxonomy; and a fresh-load inference example produced by the tested runtime.

Release the adapter together with the harness configuration needed for its reported tool-enabled performance. Keep unmerged weights as the source release; any merged or quantized derivative gets a separate artifact identity and re-evaluation. An adapter filename, lower loss, or a changed weight hash is not a capability result.

## 9. Primary sources and limits

- [M1: Official Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B). Resolve and archive the specific revision at preflight.
- [M2: SAPO paper, version 2](https://arxiv.org/html/2511.20347v2). The project adds explicit rollout, masking, and regularization contracts; do not attribute these engineering choices to the paper.
- [M3: TRL GRPO/SAPO implementation documentation](https://huggingface.co/docs/trl/grpo_trainer). Documentation availability is not target-runtime validation.
- [M4: verl Ascend support matrix](https://verl.readthedocs.io/en/latest/ascend_tutorial/en/model_support/model_and_algorithm_support.html). Support for another Qwen model does not certify this checkpoint/kernel combination.
- [M5: RLVE](https://arxiv.org/abs/2511.07317), [DAPO](https://arxiv.org/html/2503.14476v2), and [Agent-R1](https://github.com/AgentR1/Agent-R1). Relevant environment, sampling, and interaction precedents; no quantum-specific gains are inferred.
- [M6: SFT/RL tool-calling comparison](https://aclanthology.org/2026.realm-1.90/), [SFT generalization study](https://arxiv.org/abs/2510.00237), and [post-SFT selection study](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5d6ae8ba43ecb378030753c4408ef9bd-Abstract-Conference.html). Motivate controlled comparisons; they do not choose the winner here.

All numerical starting settings and planned data counts below are proposals. Report observations only after authenticated execution. No claim of guaranteed 100% IBM performance, general frontier parity, neurological equivalence, memory erasure, or novel optimizer theory is made.

The [plan validation record](qwen38/PLAN_VALIDATION.md) distinguishes checks of this documentation/configuration update from the target-model and capability gates still to execute.

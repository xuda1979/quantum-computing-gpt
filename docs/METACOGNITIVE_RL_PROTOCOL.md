# Metacognitive RL protocol: current Qwen3.8-27B entry point

**Target:** `Qwen/Qwen3.8-27B` · **Revision:** 4.0 · 11 November 2026

The current [governing plan](QWEN38_27B_ADAPTER_RD_PLAN.md) and its operational specifications supersede the earlier Qwen3.6 protocol. The full previous protocol remains in the [historical snapshot](history/METACOGNITIVE_RL_PROTOCOL_QWEN36.md).

Preserve the central observable loop: identify consequential gaps, propose justified alternatives, choose a discriminating action, acquire or calculate evidence, revise dependent claims, and finish within the evidence's scope. Direct tasks should not require artificial hypotheses or unnecessary search.

The [data specification](qwen38/DATA_GENERATION.md) now defines public/private state, claim/action/evidence identities, process/domain labels, the initial 40/25/20/10/5 distribution, the independent 5% standalone-drill union cap, checked construction, proposer prompts and worked hidden-channel environment. Teachers propose task material; the current policy supplies fresh RL actions.

The [training specification](qwen38/SAPO_TRAINING.md) defines terminal episode rewards, exact SAPO loss, actor-only masks over complete recorded histories, reference policy, module targets, canary and recovery. Process-shaping, novelty, anti-rote penalties and uncertain reward intervals remain disabled in the first causal experiment.

The [adaptive-mixture specification](qwen38/ADAPTIVE_MIXTURE.md) defines how authorized development evaluations change question-type allocations while preserving process caps, coverage and a fixed-mixture control. Sealed/native-test results do not enter that loop.

The [evaluation specification](qwen38/EVALUATION_RELEASE.md) tests decisive-change response, irrelevant-edit invariance, source interpretation, harmful revision, memory/recovery and verified final completion, alongside external quantum/math/physics/agent benchmarks. More formatted metacognitive language is not a success measure.

Use the [runbook](qwen38/FUTURE_CODE_RUNBOOK.md) for implementation and execution. No target-model training result is asserted here.

## Crux, probe, revision and evidence state

The [cognitive protocol](qwen38/COGNITIVE_CORE_PROTOCOL.md) makes the investigation loop testable. Record a consequential question, plausible alternatives, and how feasible observations would change the next action. Choose DEDUCE, SEARCH/READ_SOURCE, PROBE_TOY_CASE or CLARIFY according to the information actually missing and the measured resource budget. Link the observation to the claim it supports/refutes; invalidate dependent claims on revision. There can be several cruxes or none, so direct tasks need no artificial loop.

PROBE_TOY_CASE is a bounded falsification experiment, not proof from a few favorable examples. FOLD_EVIDENCE proposes a compact ledger of source-linked constraints, contradictions and obligations. Its proposed 150-token payload never displaces protected task requirements. Rebuild attention caches and recurrent state after admitted eviction, retain retrievable immutable receipts, and reproduce exact per-turn conditioning during training.

Initially these are observable records and offline diagnostics. A high counterfactual action divergence can be a distraction; a reference-answer likelihood increase is not mutual information. Only independently validated process evidence may enter a separately frozen, bounded success-only episode reward study. Never multiply signed crux scores by signed advantages. The initial binary terminal objective and frozen release criteria remain the controls.

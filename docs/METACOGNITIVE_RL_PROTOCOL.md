# Metacognitive RL protocol: current Qwen3.8-27B entry point

**Target:** `Qwen/Qwen3.8-27B` · **Revision:** 3.0 · 8 November 2026

The current [governing plan](QWEN38_27B_ADAPTER_RD_PLAN.md) and its operational specifications supersede the earlier Qwen3.6 protocol. The full previous protocol remains in the [historical snapshot](history/METACOGNITIVE_RL_PROTOCOL_QWEN36.md).

Preserve the central observable loop: identify consequential gaps, propose justified alternatives, choose a discriminating action, acquire or calculate evidence, revise dependent claims, and finish within the evidence's scope. Direct tasks should not require artificial hypotheses or unnecessary search.

The [data specification](qwen38/DATA_GENERATION.md) now defines public/private state, claim/action/evidence identities, process/domain labels, the initial 40/25/20/10/5 distribution, the independent 5% standalone-drill union cap, checked construction, proposer prompts and worked hidden-channel environment. Teachers propose task material; the current policy supplies fresh RL actions.

The [training specification](qwen38/SAPO_TRAINING.md) defines terminal episode rewards, exact SAPO loss, actor-only masks over complete recorded histories, reference policy, module targets, canary and recovery. Process-shaping, novelty, anti-rote penalties and uncertain reward intervals remain disabled in the first causal experiment.

The [adaptive-mixture specification](qwen38/ADAPTIVE_MIXTURE.md) defines how authorized development evaluations change question-type allocations while preserving process caps, coverage and a fixed-mixture control. Sealed/native-test results do not enter that loop.

The [evaluation specification](qwen38/EVALUATION_RELEASE.md) tests decisive-change response, irrelevant-edit invariance, source interpretation, harmful revision, memory/recovery and verified final completion, alongside external quantum/math/physics/agent benchmarks. More formatted metacognitive language is not a success measure.

Use the [runbook](qwen38/FUTURE_CODE_RUNBOOK.md) for implementation and execution. No target-model training result is asserted here.

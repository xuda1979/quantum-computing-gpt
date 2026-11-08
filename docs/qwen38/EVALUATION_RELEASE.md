# Evaluation, statistical decisions, and adapter release

Operational companion to [the governing Qwen3.8-27B plan](../QWEN38_27B_ADAPTER_RD_PLAN.md). Implement the evaluator before capability training. Freeze the measurement contract before evaluating a selected candidate on final tests. Historical smoke results are not target-model baselines.

## 1. What each comparison establishes

Use B for the untouched official checkpoint and A for an authenticated adapter on that exact revision. Keep both axes visible:

| Model | Minimal/native-compatible harness H0 | Research harness H1 |
| --- | --- | --- |
| B | Baseline model in H0 | Baseline model with research tools/memory |
| A | Adapter model in H0 | Adapter model with identical research tools/memory |

A−B within one column estimates adapter benefit under that harness. H1−H0 within one row estimates a system change and is not automatically a weight improvement. H0 must respect each native benchmark's interface; H1 is a separately named adaptation when the native protocol excludes extra tools. Do not pool their scores. Keep single-agent and bounded-worker variants separate, with equal aggregate spending.

Required method comparisons are R0 versus B, R1 versus R0, R1-MIX versus R1 when adaptive domain balance is enabled, and the selected interactive method versus its matched interactive control. Optimizer, reward, module/rank and memory studies each have their nearest valid control. SFT comparisons remain a separately enabled study, not an unreported component of A.

## 2. Benchmark registry and acquisition

| Suite | Main capability | Required handling |
| --- | --- | --- |
| [Qiskit HumanEval](https://github.com/qiskit-community/qiskit-human-eval) | IBM-style Qiskit program completion and independent construction | Standard and Hard separately; pinned 151-task revision and native runner/dependencies |
| [QHackBench paper](https://arxiv.org/abs/2506.20008) | PennyLane quantum programming | Resolve actual accessible release, task split, runner and license before promising a score |
| [LiveMathBench](https://huggingface.co/datasets/opencompass/LiveMathBench) | Broad mathematical reasoning | Pin release/date and official decoding/scoring; public exposure limitations disclosed |
| [PhysReason](https://arxiv.org/abs/2502.12054) | Physics reasoning | Separate numerical/answer evaluation from semantic step judging; audit judge error |
| [SciCode](https://github.com/scicode-bench/SciCode) | Research concepts translated into scientific code | Official main/subproblem protocols kept distinct; pinned scientific dependencies and data |
| [ScienceAgentBench](https://github.com/OSU-NLP-Group/ScienceAgentBench) | Scientific workflow completion | Official assets/evaluator, task execution and cost; complements fresh quantum workflows |
| [LiveCodeBench](https://livecodebench.github.io/) | General coding retention | Fixed time slice, language/runtime, no overlap with training roots |
| [IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval) | Instruction retention | Exact strict/loose protocol as separate metrics; no semantic repair of malformed outputs |
| Private quantum suite | New assumptions, compositions, evidence use, research and code | Fresh sealed roots, source ancestry, six information-need strata, short/long breakdown |

Optionally add licensed/accessible GPQA physics subsets and further proof or software-agent suites. They do not replace the primary quantum/research endpoints. Public access, an available API, or a paper describing a benchmark does not prove its dataset/runner was acquired. Record `READY`, `BLOCKED`, or `UNAVAILABLE` with exact dependency and licensing status. Missing cells remain UNKNOWN, never zero-cost estimated wins.

For each suite lock repository/dataset commits, complete task-ID list, dependency container digest, input/output schemas, prompt template, tools/retrieval, scorer and references, timeout/resource policy, generation settings, seed list, task exclusions, aggregation and statistical analysis. Validate references in the actual environment before model evaluation. Reproduce native conditions first; modifications get a new instrument ID and fresh base/candidate evaluation.

## 3. IBM 100% target and stronger semantic checks

At the cited public release the suite contains 151 underlying tasks. Standard supplies imports/signatures; Hard removes scaffolding from those same problems. They are paired conditions, not 302 independent scientific problems. Verify the count at the pinned run revision. If it changes, freeze and name the new denominator; do not compare mismatched task lists silently.

For the aspiration, require all 151 tasks to pass in the declared track with the declared first-attempt policy. Report separately whether it is deterministic first attempt, stochastic Pass@1, bounded Pass@k, or an interactive retry budget. A perfect result from one decoding seed does not establish robust 100% success probability. Repeat the prescribed protocol/seed set for robustness and show all runs.

If 11 tasks are unmeasured, 140/140 runnable tasks is a subset result, not 100% of the full suite. Report full reconciliation, per-task reasons, subset and full-suite conservative rates. Do not exclude failing tasks, patch outputs by hand, leak canonical solutions into retrieval, monkeypatch the scorer, or retry only failures without charging the same retry policy to all models.

Preserve two instruments:

- **Native:** exact pinned official tests and supported dependencies. Report the upstream instrument even if tests have known limitations.
- **Audited semantics:** independently reviewed checks of the requested behavior, edge cases, constraints and current-SDK compatibility. Report this as a separate instrument with its own version and reference/mutant tests.

Potential weak tests or outdated APIs justify an audited track, not secretly changing official scores. A generated program that passes weak checks may still violate the scientific contract. Record divergences between instruments and verify them independently.

## 4. Private suite and anti-memorization interventions

Provisional target: 1,200 sealed roots, 200 in each information-need stratum. Final numbers follow development-only cost/power analysis and available independent source components. The first five strata provide answerable research tasks; insufficiency is a separate guardrail. Report whether source clustering reduces the effective sample size.

Pair each applicable root with a decisive premise/source intervention and an irrelevant edit. Keep all variants in the same cluster. Score correct change after meaningful new evidence, correct invariance after irrelevant changes, handling of false premises, necessary knowledge acquisition, harmful revision, recovery, and final artifact validity. A few fixed labels or a scripted state sequence cannot substitute for end-to-end completion.

Add held-out algorithm/mechanism families, SDK conventions/versions, repositories/papers, and compositions of known primitives. New numerical parameters and paraphrases alone do not establish structural generalization. Build exact relevant math/physics retention tasks and short/long research tasks. Negative results can be successful scientific outcomes only under the task's independently checked contract; fabricated discovery fails.

For open-ended investigations without known final truth, use a separate blinded expert/reproducibility assessment with declared rubrics and inter-rater disagreement. Do not merge such judgments into the first deterministic RL reward or the main executable-success score.

## 5. Candidate and generation authentication

Every sample links run/arm, immutable base revision/file hashes, adapter hash, processor/template, runtime/kernel/precision, tool/retrieval snapshot, task/episode IDs, prompt and generation hash, seed, actual output bytes/hash, time/cost, scorer hash, verdict and evidence. The serving worker must acknowledge the exact loaded weight version. A valid API response alone does not prove adapter attachment.

Use a controlled deterministic forward/logit probe with adapter enabled and disabled, and a fresh-process reload. Check that weight/logit signatures match the intended artifacts; output text need not change on every probe. Reject stale cached outputs, duplicate episode IDs, wrong model/adapter, incompatible test versions, partial receipts and missing evidence. Re-evaluate or authenticate equivalence before comparing cached results.

Freeze native benchmark decoding from its documented protocol. For a separate project deterministic track use greedy decoding and the declared token/tool budget. For project stochastic Pass@1 use fixed seeds and temperature/top-p/top-k settings recorded in the contract; no best-of selection. Use identical settings for B and A. Model-specific native frontier settings may be additionally reported, labeled separately from matched-harness comparisons.

## 6. Metrics and denominators

Primary external quantum endpoint:

\[
Q=0.5 Q_{QiskitStandard}+0.5 Q_{QHack}.
\]

Use frozen tool-enabled project tracks with unchanged final functional checks for the primary adapter research comparison, and separately report all native scores. Freeze family/difficulty weighting within each term before testing. Hard is an additional required quantum guardrail, avoiding double weight for the same underlying Qiskit problems. Both primary suites must be measured under a compatible contract; do not renormalize away a missing suite.

Private research endpoint S is the macro-average verified episode success across the five answerable information-need strata. Freeze within-stratum root/family weights and short/long weights. Insufficient-information success, unsupported claims, harmful revision, math/physics/general-code/instruction retention, Hard score, and resource use are guardrails. Publish all component scores; a favorable aggregate cannot hide a substantial SDK or long-task regression.

Per task reconcile exactly one of:

- PASS: authentic candidate satisfied all required checks.
- FAIL: authentic candidate violated its contract, including malformed output or candidate-caused timeout.
- VOID: measurement invalid because of authenticated evaluator/infrastructure failure.
- UNKNOWN: missing or unresolved evidence.

Let N be all scheduled unique tasks. Report `(PASS+FAIL+VOID+UNKNOWN)=N`, measured rate `PASS/(PASS+FAIL)` when defined, conservative full rate `PASS/N`, and optimistic bound `(PASS+VOID+UNKNOWN)/N`. Preserve retry history and spending. If missing outcomes could reverse a decision, withhold promotion. Never describe a VOID as a model failure or success.

For n independent sampled completions with c successes and n≥k, report the standard estimator `pass@k = 1 - C(n-c,k)/C(n,k)`, with the natural value 1 when n-c<k. Do not relabel average success as Pass@k. An interactive episode with internal repairs is one episode under its total budget; its final success is distinct from independent-code-sample Pass@k.

Also report confidence/abstention calibration, action validity, evidence support, recovery, loop/stall rates, executed artifact reproducibility, total generated/processed tokens, tools/simulator/teacher usage, device-hours, wall time and real or explicitly estimated monetary cost. No unmeasured internal reasoning/FLOPs equality claim is required for proprietary systems.

## 7. Statistics, checkpoint selection and promotion

Before sealed testing freeze the full primary/guardrail family, candidate-selection rule, sample sizes, seeds, margins, multiplicity method, bootstrap procedure and missing-data policy. Select on dev_tune only: first require development guardrails, then maximize a declared equal-weight normalized Q/S objective; break ties by lower total measured inference cost, then earlier checkpoint. Calibration data follows its separate declared use. Never choose the best sealed-test seed/checkpoint.

For each co-primary endpoint E in {Q,S}, compute its required practical point gain from the independent development baseline:

\[
\delta_E=\min(0.10,0.20(1-E_{base,dev})).
\]

Freeze delta before final testing. If the baseline is at/near ceiling, design and reseal a harder appropriate endpoint before testing; do not claim a zero-margin victory. Require the candidate's point gain to reach delta and the paired confidence lower bound to exceed zero against both B and the nearest valid RL control. Treat the experiment as a narrower finding if it beats B but not the nearest control.

Use paired source/root-cluster resampling, preserving variant and Standard/Hard dependence. For confirmation, train the selected method and nearest control with seeds 17,29,43 and report every seed. Use a prespecified hierarchical seed/cluster analysis; three seeds still provide limited precision. One seed supports a pilot result only. Set final sample size from development-only paired variance and detectable gain; variants do not repair low independent-cluster power.

Initial analysis implementation uses 10,000 paired hierarchical bootstrap replicates and RNG seed 314159, percentile bounds at the prespecified adjusted levels, and the frozen suite/stratum weights recomputed in each replicate. Resample training seeds and then source/root clusters while keeping model comparisons paired. Public single-checkpoint descriptive scores remain separate from method-level training-seed inference. Check Monte Carlo stability on development and fix any larger replicate count before final testing.

Control the declared contrast family. For example, two endpoints against two comparators produce four primary contrasts; Bonferroni two-sided intervals have 98.75% individual coverage for family alpha=0.05. A different justified method must be fixed in advance. For retention, require lower bounds of A−B above -0.02 using a separately prespecified multiplicity-controlled guardrail family. A point decline smaller than two points is not sufficient evidence of noninferiority. Wider uncertainty produces INCONCLUSIVE/UNKNOWN, not PASS.

Promotion requires co-primary gains, every declared retention/validity guardrail, valid full denominators, operational load/resume success, and artifact integrity. If cost ceilings or latency budgets are guardrails, compare them with complete charged costs. Confidence intervals do not remove contamination or invalid scorer bias.

## 8. Frontier-model comparisons

Resolve precise available GPT-6.1 and Claude Opus 5.5 API model/snapshot identities at evaluation time. Record date, provider settings, reasoning effort, prompt/tool permissions, context/budgets, actual token/time/cost accounting and failures. Do not guess proprietary model sizes or assert that a model alias is immutable.

Use the same public task context and permitted tools in a matched-harness track. Add each system's native recommended configuration as a separately named practical track. Equal output-token counts do not establish equal compute; compare quality at declared total cost/time budgets and publish the resource curve where feasible. No teacher/evaluator can expose private test answers to a competitor.

Before execution define what competitive means per suite: e.g. an explicitly chosen noninferiority margin and a confidence lower bound for A−frontier, alongside cost. Do not average away a major math, physics or agent failure. Missing provider access is UNKNOWN; omit unsupported parity/top-model claims. Matching one public suite does not establish broad scientific research parity or global rank.

## 9. Required ablations and failure diagnosis

At minimum retain fixed-mixture, fixed-harness and outcome-only RL controls. Separately test adapter disabled; retrieval disabled; oracle-relevant passages; corrupted/withheld decisive evidence; structured memory versus equally sized generic summaries; single-agent versus workers at matched aggregate cost; held-out family/version; and short versus long tasks. Allocate ablations from measured budget; unrun cells remain missing.

Classify failures as data/admission, reference/scorer, environment/API, model knowledge/concepts, math/physics reasoning, source interpretation, planning, memory, recovery, output contract, optimization, runtime, or budget. Feed only authorized development diagnostics to the adaptive-mixture controller. Inspect final-test failures for a postmortem only after completing the frozen evaluation; a new adaptive iteration needs a fresh test claim/version.

## 10. Release package and reproducibility

Release directory contract:

```text
release/<artifact-id>/
  adapter/                    # actual PEFT numerical weights and config
  model_lock.json             # official base revision, processor/template and hashes
  resolved_contract.json
  runtime/                    # dependency lock, image digest, hardware/kernel notes
  manifests/                  # data/source/split/scorer and SHA-256 manifests
  evaluation/                 # per-task outputs, receipts, aggregate/statistical reports
  training/                   # learning/cost history and checkpoint lineage
  inference_example.py        # generated and tested against the selected runtime
  MODEL_CARD.md               # scope, measured scores, limits, required harness
  REPRODUCE.md
```

Large/private data and checkpoint files can be referenced through controlled durable storage with verified hashes and access instructions; do not place secrets in Git. Retain the last known-good artifact and state exact rollback/load procedure. Save unmerged adapter as the source artifact. Merged, quantized, different-engine or different-base derivatives need distinct identities and re-evaluation.

Fresh-process smoke must load the pinned base plus adapter, verify attachment, generate with the declared template/tools, pass selected public/authorized fixtures, and reproduce report checks. Test packaging against an explicit file list and hash every file. The artifact tested must be the artifact delivered. Mark each claim MEASURED, PROJECTED, MISSING or PROPOSED; record engineering readiness, research improvement, IBM aspiration and frontier aspiration separately.

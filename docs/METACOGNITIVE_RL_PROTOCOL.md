# Hypothesis Testing and Metacognitive Control: RL Protocol

**Project:** quantum-computing-gpt

**Target:** Qwen/Qwen3.6-35B-A3B

**Revision:** 1.0 • 30 October 2026

**Status:** Proposed interfaces and experiments; no implementation or model results are asserted.

This protocol extends the [governing R&D plan](LEARNING_TO_CLARIFY_RD_PLAN.md). It turns the supplied hypothesis–falsification proposal into an RL-only curriculum, an observable decision contract, and a falsifiable evaluation. The governing plan's budgets, checkpoint controls, independent verifiers, and promotion criteria still apply.

## 1. Objective and scope

Train a research policy to identify consequential uncertainty, propose a small set of explanations, choose a test that could distinguish or refute them, acquire evidence when necessary, revise dependent claims, and conclude only within the evidence's scope. The learning target is effective control of an investigation. Generated prose and the number of state labels are not measures of that control.

The intended loop is:

**Identify knowns and gaps → propose alternatives → design a discriminating test → derive, retrieve, ask, or experiment → update evidence → backtrack if contradicted → answer, qualify, or defer.**

Use two or three substantive alternatives when the problem admits them. A direct, already justified answer needs neither invented rivals nor a compulsory search. Alternatives need not be mutually exclusive or exhaustive; record that limitation. A valid null explanation and an unresolved possibility should remain available where warranted.

The existing project decision remains **RL-only adaptation**. The supplied SFT → DPO/GRPO → online-agent sequence becomes baseline/lifecycle validation → short-horizon on-policy RL → interactive research RL, followed by separately controlled process-reward studies. No project SFT, DPO, demonstration imitation, best-trace replay, or trajectory distillation is introduced. Existing upstream model training is outside this restriction.

Prefer external evidence for literature, changing APIs, measurements, and long-tail details. Preserve the mathematics, scientific concepts, programming ability, and correct knowledge needed to interpret that evidence. Do not train deliberate forgetting, incorrect answers, or a blanket inability to answer without tools. An adapter cannot guarantee that all facts leave the weights or that 35B parameters contain only a control policy. Measure reduced dependence on recalled templates using new-information and source-intervention tasks.

Human research supplies an engineering analogy. This protocol makes no claim of neurological equivalence, human-level intelligence, guaranteed discovery, or an exclusive parameter allocation for reasoning. The frozen model card determines the architecture: the requested 35B checkpoint is the 35B-total/3B-active MoE identified in the governing plan.

## 2. Observable state and actions

Maintain a compact public decision/evidence ledger rather than requiring an exhaustive private reasoning transcript. Every consequential claim has an ID, scope, assumptions, dependencies, status, evidence references, and an unresolved obligation where applicable. Confidence and trusted verification are separate fields.

| State | Meaning | Evidence or transition requirement |
| --- | --- | --- |
| ASSUMED | Public premise adopted conditionally | Identify its source and scope; do not certify it as true |
| HYPOTHESIS | Candidate explanation or claim | State a distinct prediction and relevant assumptions |
| TO_TEST | A feasible discriminating check is pending | Specify observation, possible outcomes, refutation rule, and cost |
| NEED_EVIDENCE | A decision-relevant fact or condition is missing | Name the gap and an appropriate source, query, measurement, or user question |
| CONFLICT | Incompatible claims remain unresolved under matched conditions | Link both claims and check definitions, versions, and dependence of sources |
| REFUTED | A branch fails within its stated scope | Attach a valid counterexample or failed obligation and affected descendants |
| SUPPORTED | Available evidence favors or has not refuted the claim | State limits; surviving finite checks is not a universal proof |
| VERIFIED | A trusted check establishes a specific obligation | Name the checker, assumptions, artifact, and exact certified scope |
| DEFERRED | The available method or budget cannot settle the obligation | Preserve reason, remaining uncertainty, and a condition for reopening |

The model **proposes** status changes. The trusted harness owns verification fields and authentic tool outcomes; writing VERIFIED cannot grant a certificate. Retrieved assertions initially remain attributed assertions. A proof establishes only the checked statement; a confidence interval or experiment establishes only its declared statistical obligation.

Use the governing actions plus explicit **HYPOTHESIZE, DESIGN_TEST, ASK, BACKTRACK, and DEFER** operations. DESIGN_TEST declares the affected claims, predicted outcomes, falsifier, method (logic, source inspection, or experiment), assumptions, feasibility, and budget. BACKTRACK identifies the invalid premise, records why it failed, invalidates dependent conclusions, and chooses a replacement investigation. Preserve independent valid subresults.

Execution follows these rules:

1. Separate a supplied premise, recalled belief, retrieved assertion, and checked observation.
2. Choose an action whose result can change a consequential decision. Pure logical gaps need derivation or checking; missing user intent may need ASK; empirical uncertainty may need measurement.
3. Seek possible disconfirmation as well as support. A failing search is not proof of absence, and an unrejected hypothesis is not proven.
4. Resolve apparent source conflicts by scope, definitions, date/version, methodology, and original evidence. Copied summaries are not independent votes.
5. On a valid contradiction, backtrack through the dependency DAG. Reopen a refuted branch only with explicit changed assumptions, new evidence, or a demonstrated defect in the earlier check.
6. Answer directly when obligations are met. Otherwise distinguish a scoped conditional result, justified insufficiency, and a budget-limited incomplete result. DEFERRED does not automatically count as success.

These labels describe an auditable external interface. They do not prove that the labels faithfully expose all internal computation.

### 2.1 Example exchange contracts

The following JSON is illustrative design data, not a supported runtime API. The public task is separate from the private scorer, hidden source schedule, accepted alternatives, and test references.

```json
{
  "schema_version": "metacognitive_task_v1",
  "task_id": "opaque-task-0042",
  "public_problem": "Two implementations pass standalone equivalence checks. Determine whether substituting one for the other is safe under coherent control.",
  "completion_contract": "Return a checked witness and a scoped conclusion, or an evidenced explanation of what remains unresolved.",
  "available_tools": ["read_source", "exact_circuit_check"],
  "budget_profile": "short_research"
}
```

A policy decision can be short; its evidence IDs must resolve to prior authentic observations. Test expectations are proposals, not manufactured measurements.

```json
{
  "schema_version": "metacognitive_decision_v1",
  "claim_id": "c2",
  "proposed_status": "TO_TEST",
  "action": "DESIGN_TEST",
  "assumption_ids": ["c1"],
  "alternatives": ["equivalence survives control", "a relative control phase distinguishes them"],
  "test": {
    "method": "exact_circuit_check",
    "observable": "control X expectation for a declared input state",
    "refutation_rule": "A checked difference refutes context-independent substitutability."
  },
  "evidence_ids": [],
  "trusted_verification": null
}
```

The private registry additionally holds root/source ancestry, domain and process categories, split, licensed provenance, admissible answer/abstention conditions, verifier/scorer hashes, and intervention-pair IDs. The trusted loader constructs public observations through an allowlist; it never sends that registry or evaluator annotations to the actor. Negative diagnostic records live separately and cannot enter policy loss.

## 3. Five-category process curriculum

Adopt the supplied **40/25/20/10/5** allocation as the initial RL process distribution. These are task/environment categories, not SFT target traces. Each admitted root has exactly one primary process category and one domain bucket from governing-plan Section 5.2; secondary skill tags may overlap. The two mixtures are orthogonal constraints on a feasible joint sampling matrix, rather than percentages added together.

| Category | Share | Environment design | Trusted success evidence | Paired failure probe |
| --- | ---: | --- | --- | --- |
| 1. Hypothesis generation and falsification | 40% | Two or three plausible explanations, with a feasible distinguishing observation | Nonredundant alternatives, valid falsifier, executed check when available, warranted update | Confirmation-only choice; rivals with identical predictions; certainty without testing |
| 2. State awareness, gaps, and evidence acquisition | 25% | A: missing condition; B: conflicting, stale, or unreliable evidence | Correct gap identification and useful next action; source-sensitive resolution | Invent a missing fact; search despite sufficient information; trust the first snippet |
| 3. Error diagnosis and backtracking | 20% | A flawed attempt with a verified hidden assumption or localized defect | Identify the first consequential error, invalidate descendants, preserve valid work, repair on a fresh case | Replace the answer without diagnosis; discard correct work; retain a disproven premise |
| 4. Resource and attention allocation | 10% | A dependency DAG with bottlenecks, risks, alternative tests, and explicit costs | Better verified outcome or lower cost under the same budget | Expand every node equally; chase easy irrelevant nodes; ignore a necessary low-priority check |
| 5. Concise foundational logic | 5% maximum | Diverse small identities, elementary proofs, and symbol transformations | Independent correctness and efficient completion | Template copying, unnecessary branching, or tool use on an already sufficient problem |

For a 2,000-root training inventory, the starting counts are 800/500/400/200/100. These are unbuilt targets. Do not count multiple paraphrases or rollouts as independent roots. In the pilot, stratify reviewed roots across categories and domains; report the actual feasible joint matrix and empty cells.

Schedule complete prompt groups in preregistered blocks of 100 with 40/25/20/10/5 dispatches. Retries retain category identity and incur cost; do not replace failed groups with easy successes. Reconcile planned, dispatched, completed, quarantined, consumed, and trained counts separately. A run is not compliant merely because its initial pool has the desired proportions.

For a shorter canary or a partial final block, freeze integer quotas before launch and round the Category 5 count down so it cannot exceed 5%. Report rounding of the other shares. Order and budget dispatches so the declared consumed-group/token caps can be satisfied; do not drop an already generated failure to repair a quota. A canary with fewer than 20 consumed groups can contain no Category 5 group while still testing admitted fixtures from that category outside policy updates.

Category 5 must remain at most 5% of consumed prompt groups and trained actor tokens over each declared accounting window and the full run. Set generation caps before dispatch to control this; if actual accounting breaches a cap, report and pause/rebalance future scheduling without deleting failed samples or rewriting history. Retention tasks can occur in every process category. The short-horizon cold start uses simpler versions of Categories 1–4 and does not become an unrestricted Category 5 warmup. No increase above the Category 5 cap is permitted within this protocol; a different allocation requires an explicit new research decision.

Do not eliminate the shared control pattern itself: learning a reusable investigation policy is the objective. Deduplicate problem content, source ancestry, solution templates, witness structures, and non-substantive variants while retaining genuinely different uses of a useful policy.

## 4. Sources and generation prompts

Prefer permitted primary papers, official documentation, checked mathematical examples, independently reproduced failure reports, and deliberately constructed new-information worlds. Public MathOverflow or research-discussion posts can suggest roots subject to terms and attribution; they cannot supply trusted truth labels. arXiv manuscripts are sources, not a universal arXiv comment dataset. Use open or explicitly authorized reviews; do not assume confidential reviewer reports or forum content are licensed for reuse.

Historical refutations and philosophical arguments can motivate tasks. A historical reconstruction remains a retrospective diagnostic, and open philosophical disagreements need scoped review rather than an invented binary oracle. Initially train only where outcomes or decision obligations can be independently adjudicated. Use an LLM to propose cases and failure modes; independent review/checking admits the task.

### 4.1 Shared generator instruction

Apply this instruction to every category-specific template below. The generator's output is **untrusted task-construction material**; the actor generates its own subsequent on-policy attempt.

```text
Construct one research environment for {{domain}} and {{process_category}}.
Inputs: permitted source references {{sources}}, root constraints {{constraints}},
and split assigned by the trusted registry {{split_reference}}.

Return separate public-task and private-evaluator proposals.
Public: problem, available evidence/tools, budget, and completion conditions.
Private: provenance, competing possibilities, independently checkable obligations,
admissible alternatives, expected interventions, failure probes, and verifier needs.
Include a condition-sufficient control and a meaningful change of decisive evidence.
Do not include answers, evaluator references, hidden source outcomes, or model
demonstrations in public input. Do not assign your own trusted label or split.
Do not generate a polished solution trace for imitation. State what remains
unverifiable; such a task cannot receive a positive training label yet.
```

### 4.2 Category 1: hypothesis and falsifier

```text
Given {{open_question}}, construct a bounded case with two or three substantive
candidate explanations and a feasible discriminating test. In the private proposal,
state each prediction, a possible falsifier, required assumptions, and whether the
test needs logic, authorized sources, or observation. Specify an independently
checkable terminal obligation. Include confirmation-biased and non-discriminating
failure probes. The public task asks the actor to investigate without revealing
which explanation is correct. If no current test can settle the issue, define a
checkable deferred-result contract or retain the case for evaluation only.
```

Useful roots include a limited mathematical conjecture with a checked counterexample space, competing explanations of a reproducible engineering failure, and an empirical mechanism with an accessible controlled experiment. Do not treat all open questions as if an answer oracle exists.

### 4.3 Category 2: gaps and conflicting sources

```text
Given {{question}}, propose two information conditions in the same root:
A: a consequential condition is absent; B: sufficient evidence is available.
Also propose a conflict variant with attributable sources whose disagreement can
be checked for scope, version, methodology, or source dependence. Identify the
missing variable and useful query/ASK/measurement in the private proposal.
The actor must choose whether to answer, derive, search, ask, or defer.
Do not require two or three queries when one targeted action suffices.
Include an unsupported-fact probe and a needless-retrieval probe.
```

The actor should list supplied/checked premises separately from uncertain beliefs. A recall-based assertion does not become verified because the prompt asks for a list of known facts. When conflict cannot be resolved, preserve uncertainty with a precise remaining obligation.

### 4.4 Category 3: hidden assumption and backtracking

```text
Given {{problem}}, propose a short flawed public attempt with one independently
confirmed consequential defect and some correct reusable subresults. Keep the
private defect location and witness outside actor input. Ask the actor to identify
the unsupported premise, test it, invalidate affected descendants, distinguish a
missing fact from a logical error, and choose a revised plan. Include a fresh case
that a cosmetic answer correction will fail. Add valid-but-unusual reasoning as a
control so that automatic rejection of unfamiliar arguments is not rewarded.
```

An externally supplied flawed attempt is task context, not a behavior-policy rollout. Mask its tokens from actor loss. A diagnosis must target an actual defect; a changed final answer alone earns no diagnosis credit.

### 4.5 Category 4: attention and resource allocation

```text
Given {{complex_problem}}, construct a dependency DAG with stated action costs,
uncertainties, constraints, and one consequential bottleneck. Ask the actor to rank
investigations by expected decision value and cost, execute a bounded choice,
then revise the plan after new feedback. Privately specify feasible alternatives
and how downstream goal success/cost will be evaluated. Include an easy irrelevant
branch and a seemingly minor prerequisite that still must be checked.
```

Priority can be high/medium/low, but importance is justified through consequences. Low priority never authorizes silently assuming a necessary condition. The governing SELECT study estimates value by matched continuations; an LLM's priority label is not the value oracle.

### 4.6 Category 5: foundational logic

```text
Propose one short, independently checkable reasoning task from {{family}}.
Vary mathematical or programming structure across admitted roots. The actor may
give a concise direct solution with only the states/actions actually needed.
Exclude bulk same-template olympiad, calculus, or textbook exercises. Register
the root and checkable obligation; provide no target completion for policy loss.
```

## 5. Admission, cleaning, and trajectory controls

1. **Split first.** Freeze source/generator/theorem/algorithm/witness ancestry before mutations, flawed attempts, retrieval caches, or diagnostic pairs. Keep paired interventions in the same split. Public benchmark solutions never enter training or retrieval.
2. **Check the contract.** Independently validate the public problem, allowed observations, feasibility, decision-relevant gap, admissible alternatives, and oracle. A generator's reference is only a proposal. Keep judge-dependent open research out of the first reward pipeline.
3. **Deduplicate structurally.** Combine text/semantic similarity with source ancestry, AST/circuit/proof structure, and manual sample audits. Similarity thresholds are development choices to freeze. Avoid both repeated answer templates and deletion of legitimate common control operations.
4. **Audit every category.** Require reviewer-inspected examples, positive/negative verifier fixtures, per-family counts, licensed provenance, and the domain/process matrix. Record why each root was admitted or rejected.
5. **Use concise records.** Treat 800–1,500 generated tokens as a starting upper-range target for a complete short investigation record, not a minimum. Single decisions may be much shorter. Longer tasks use multiple budgeted turns and persistent evidence pointers. Do not reward padding or truncate away the observation that reverses a claim; classify incomplete attempts explicitly.
6. **Separate filtering from RL.** Reject invalid environments and contaminated roots before rollout. After rollout retain complete groups, including direct guesses, wrong branches, hallucinated evidence, and malformed attempts, with their declared rewards. Removing low-quality attempts would bias the policy update. A concise correct direct answer is valid on a sufficient-information task.
7. **Keep negatives diagnostic.** Crafted good/bad contrasts test task/scorer sensitivity and suggest new environments. They do not become chosen/rejected DPO examples or cross-entropy targets. Only freshly sampled eligible actor tokens with authentic behavior probabilities enter the declared RL objective.

Record root coverage, semantic/structural duplicate rate, oracle disagreements, unresolvable cases, source-access failures, group/token shares, and truncation. Every accepted root needs a reproducible audit trail. More generated text is not evidence of better training data.

## 6. Training and online interaction

| Stage | Learning and environment | Required gate |
| --- | --- | --- |
| A. Baseline and lifecycle | Untouched-base modes and an isolated RL lifecycle smoke before research training | Authentic tools and independently checked update/save/resume path |
| B. Short-horizon RL | Simple falsifier choices, gap decisions, error repair, and allocation across all process categories | Nonconstant reward support, valid actor masks/probabilities, retention, compliant mixture |
| C. Interactive research RL | Dynamic evidence, source conflicts, execution feedback, backtracking, and final obligations | Verified downstream success versus matched base and outcome-only RL |
| D. Controlled extensions | Metacognitive shaping, SELECT value studies, then bounded workers and memory | Independent scorer audit and matched-cost ablations for each mechanism |

The environment loop is policy action → trusted tool/measurement result → evidence-ledger update → next action. The policy learns from consequences. The first interactive training environments use versioned local source stores with seeded noise, missing pages, stale versions, contradictory assertions, and controlled evidence updates. Tool results are emitted by the environment, not written by a teacher into a successful actor trace.

For comparative evaluation, pair the source snapshot, update schedule, and stochastic seed across base/adapter. Run live-web robustness separately with timestamps and availability differences disclosed. Neither training nor testing may expose hidden evaluator answers through retrieval. Charge queries, failed calls, document reads, workers, and repeated context to the same budget.

Freeze the behavior revision within each complete group. Store action ownership and span masks, behavior log probabilities, source hashes, observations, rewards, and all failures. Tool outputs, supplied flawed reasoning, public prompts, and foreign-worker tokens stay context and are excluded from actor loss. Test that masking still allows the actor to condition on evidence, and never mistake masking for protection against leaked answers.

Collect new on-policy episodes into immutable run records. Use findings to propose new training roots, source fixtures, or scorer attacks between versioned runs; independently admit them and freeze a new manifest. Do not repeatedly optimize on old selected successful traces as if they were fresh on-policy samples. A later off-policy algorithm would need a separate estimator and contract, not an implicit DPO feedback loop.

## 7. Reward hierarchy and safeguards

Keep the governing plan's terminal-success reward as the initial outcome-only control. A useful metacognitive controller must deliver correct, warranted results. Giving final correctness the lowest unrestricted weight could reward articulate failure, excessive hesitation, or invented uncertainty.

For a separately preregistered shaping experiment, define:

$$
R_i^{\rm meta}=Y_i-0.05C_i-0.05F_i+0.04M_i-0.04U_i.
$$

Here the first three terms have the governing meanings: checked contract success, normalized total cost, and proven fabricated evidence-link fraction. \(M_i,U_i\in[0,1]\) are independently scored decision quality and avoidable action waste. Use a fixed mixture

$$
M_i=0.30g_i+0.25h_i+0.20b_i+0.15a_i+0.10\ell_i,
$$

with all components in [0,1]. This is a proposed experimental choice, not a validated optimal reward.

| Component | Required observable evidence |
| --- | --- |
| g: gap and acquisition judgment | A real consequential gap is found and the selected search/ASK/test can resolve it; sufficient-information controls avoid unnecessary actions |
| h: hypothesis and falsifier quality | Alternatives differ substantively and the feasible test distinguishes outcomes under explicit assumptions |
| b: contradiction and backtracking | An actual defect is localized; dependent conclusions are invalidated; valid subresults survive; subsequent actions use the correction |
| a: attention allocation | A bounded investigation improves independently checked continuation utility or cost against feasible alternatives |
| ell: logical discipline | Claims remain scoped, dependencies are valid, and a supported empirical hypothesis is not promoted to a theorem |

Logical validity is also part of terminal success when required; the ell term cannot excuse a wrong result. Set a genuinely inapplicable component to zero using the task's frozen applicability mask, rather than renormalizing after seeing the output. Missing evidence for a required component makes that shaping measurement VOID; it is not an invented favorable score. Scorer reliability, alternative-valid-strategy handling, and independent sample audits must pass before enabling this branch. Tasks without reliable process scoring remain in a separately declared outcome-only run.

U covers demonstrably redundant search/ASK, repeated discarded branches without new evidence, or checks that cannot affect any outstanding obligation. It does not penalize necessary safety/validity checks or a source failure caused by the environment. Cost is still charged. Fabricated certificates, permission violations, and invalid actions follow the separately declared hard-invalid penalty. Infrastructure failures remain VOID, never an invented reward.

For infrastructure-valid eligible episodes, a success receives at least 0.86 and a failure at most 0.04. Thus process points cannot make a failed episode outrank a success. Independently test these bounds and all invalid/missing-value cases before training. Plan-only tasks can satisfy only their declared planning obligation; report them separately from full investigation success. No reward is earned by printing HYPOTHESIS, BACKTRACK, or VERIFIED, increasing the number of branches, or writing more text.

Audit plausible but invalid falsifiers, manufactured contradictions, gratuitous abstention, fake source IDs, backtracking from correct work, and rewards that favor cheap failure. Compare outcome-only and shaped RL with identical root exposure, generation/update tokens, evaluator access, and total measured compute. Do not combine a SELECT estimator or group-coupled diversity reward with this formula without a new mathematical contract.

## 8. Evaluation that can disprove the proposal

Use four behavioral dimensions alongside the governing plan's co-primary external quantum-code and full research-success endpoints. The custom dimensions diagnose the controller; they do not replace independent end-result correctness. MMLU/GSM8K are not optimization targets. Correct closed-book foundations, coding, instructions, and tool use remain retention guardrails; a decline is neither guaranteed nor automatically acceptable.

| Dimension | Frozen measurements and denominators | Failure the test must expose |
| --- | --- | --- |
| Uncertainty and gaps | Gap precision/recall on annotated decision opportunities; false-gap rate on sufficient cases; unnecessary tool/ASK rate; calibrated confidence and success at matched coverage | Invented facts, reflexive retrieval, or refusal to answer solvable tasks |
| Hypothesis and falsification | Nonredundant plausible alternatives, valid feasible falsifiers, executed discriminating-test success, and scoped updates per eligible hypothesis opportunity | Decorative rivals, confirmation bias, or a finite test presented as universal proof |
| Backtracking | First-defect localization, correct descendant invalidation, preserved valid subresults, fresh-case repair success, harmful revision on valid controls | Cosmetic answer changes, false self-criticism, or reuse of a disproven premise |
| Conflicting evidence | Contradiction detection, valid scope reconciliation, source-to-claim fidelity, correct response to changed decisive evidence, and false-certainty rate per adjudicated conflict | First-source trust, copied-source voting, ignored contradictions, or needless permanent deferral |

Report raw opportunity/task counts, applicability masks, exclusions, per-family scores, cost, and uncertainty; never score an empty denominator as perfect. Human reviewers see blinded artifacts with an alternative-valid-solution rubric. LLM judges may triage but do not supply the sole truth criterion. Use scorer versions and separate diagnostics from reward-training rubrics.

The private suite must include:

- Sufficient, missing-condition, truly undecidable-with-available-evidence, and budget-limited cases with different correct stopping actions.
- Decisive evidence interventions with unchanged style, plus irrelevant wording/label changes that should preserve the decision. Cluster every pair under its root.
- The same attractive wrong branch followed by authentic disconfirmation, and valid unfamiliar arguments that should be retained.
- Correct-looking state labels attached to wrong actions, and semantically correct concise actions without decorative labels. Score observable consequences separately from parser compliance.
- Unseen source/generator/witness families, new API conventions after model freeze, source-withheld and oracle-evidence conditions, and long episodes requiring persistent dependency state.

Compare base, outcome-only RL, and metacognitive RL under the same harness, prompts, tools, retrieval snapshots, workers, and aggregate budgets. Add the curriculum ablation against an admissible generic mixture, with equal structural root count and training resources. Use label-removal/renaming and label-only prompted controls to test format dependence. These are prespecified protocol variants, not silent changes to official benchmarks.

The hypothesis is rejected or narrowed if gains disappear on held-out structures, oracle evidence does not help, correct labels coexist with wrong actions, tool use rises without verified benefit, or foundation/quantum-code retention fails. A specific behavioral improvement requires the preregistered paired effect/interval for that dimension against the nearest RL control, with multiplicity and noninferiority margins fixed from development. Apply the governing co-primary promotion gates as well; do not select checkpoints using sealed diagnostics.

## 9. Implementation exit conditions and claim limits

Implement versioned public/private episode schemas, claim/test/dependency ledgers, a process/domain sampler with token accounting, independent decision scorers, seeded retrieval fixtures, a real RL lifecycle, and the matched evaluation matrix. These components are planned work. Keep the existing one-qubit fixtures and legacy supervised trainer accurately labeled.

The first acceptable pilot artifact contains independently admitted roots in all five categories, readable source lineage, a complete authenticated on-policy group, an independently recomputed update and reward bounds, save/resume evidence, base/control outputs, and attack-test results. A canary establishes operability; a larger verified, adequately powered comparison is required for efficacy.

On a source outage, tool failure, exhausted budget, or uncertain oracle, preserve state and display the reason. Resume from authenticated events when recovery is feasible. A stopped investigation must retain its unresolved claims and cannot be reported as a completed scientific result.

Primary sources grounding the model and RL/search precedent are the [official model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) and [Search-R1](https://arxiv.org/abs/2503.09516v5). They do not establish this curriculum's efficacy, selective forgetting, brain equivalence, or superiority over larger models. Those claims require their own evidence; this protocol proposes tests and reports no gains.

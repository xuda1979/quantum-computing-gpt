# Crux finding, evidence use, and compact research state

Revision 1.0 · 11 November 2026 · Companion to the [governing plan](../QWEN38_27B_ADAPTER_RD_PLAN.md).

The objective is a Qwen3.8-27B adapter that identifies consequential unknowns, runs useful investigations, and completes quantum research tasks efficiently. The active route remains RL-only SAPO from the official post-trained checkpoint. This protocol specifies proposed experiments; it does not implement an interactive trainer or report capability gains. The separate [planning contract](../../configs/qwen38/cognitive_contract.example.json) is deliberately unable to authorize training.

## 1. Decisions on the feedback

| Proposal | Decision and reason | Evidence required before adoption |
| --- | --- | --- |
| Remove factual FFN subspaces | Defer destructive weight edits. Factual knowledge, concepts and reasoning are not known to occupy separable subspaces; a probe score does not establish such a separation. Lower recall is not a success criterion. | A separate recoverable compression study, structural parameter accounting, real memory/latency measurements and retained quantum/math/physics/general-code performance |
| Specialize hybrid blocks | Test asymmetric adapters after enumerating actual module paths, shapes and supported gradients. Freezing FFNs limits those updates; it cannot prevent other trainable weights from memorizing. | Both block families represented, zero-effect initialization, probability/cache parity, matched parameter-budget and cost controls |
| Crux-SAPO | Begin with observable decisions and offline diagnostics. Add bounded episode shaping only in a separate arm after validating the scorer. | Independently checked useful probes, unchanged terminal verifier, independent development improvement and total-cost accounting |
| Mutated worlds | Admit coherent sandbox worlds and paired tests. Start with a proposed 50% intervention exposure only in its own treatment arm. | Environment/document agreement, reference and mutant tests, ancestry isolation, ordinary-world retention |
| Toy falsification | Add an explicit bounded action, with 2–3-qubit examples initially. A small example can disprove a conjecture; passing it does not prove arbitrary-size correctness. | Executed receipt, valid interpretation, subsequent full-contract checks |
| Evidence folding | Test compact, source-linked working state while retaining immutable receipts. Rebuild the actual model context and recurrent state after validated folding. | Replay parity, contradiction/constraint preservation, recoverable source access, matched summary/full-context controls |
| Frontier curriculum | Study learnability and useful-probe signals with uncertainty, exploration and measured cost. Do not replace the existing controller with an unvalidated variance product. | Selection probabilities and all candidate costs, held-out development gains, coverage and retention |
| Distill to 4B–8B or repurpose MTP | Defer. A student changes the required base; imitation of teacher traces changes the RL-only route. Token-prediction heads are not calibrated uncertainty/value heads. | A separately scoped study and objective; actual head availability, calibration, gradient and load tests |

No expected reduction to 12B–15B, human equivalence, few-hundred-step emergence, submillisecond probe latency, or guaranteed IBM pass rate is asserted. SVD truncation changes deployed size only if factors are actually stored and executed in a supported representation; low numerical rank alone does not reduce dense tensor storage.

## 2. Observable crux and action contract

A crux is a task-dependent uncertainty whose resolution changes a justified next action or removes a consequential obligation. There may be several cruxes, or none. Do not force a single bottleneck onto every problem. Record short decision summaries, testable predictions and evidence links; do not solicit teacher reasoning transcripts as training targets.

Extend the research task/episode schema, not the legacy one-qubit Task class, with a versioned public projection:

~~~json
{
  "protocol_version": "qresearch-cognitive-v1",
  "crux_id": "c1",
  "claim_ids": ["channel-family"],
  "question": "Does the channel change the excited-state population?",
  "alternatives": ["population-changing", "population-preserving"],
  "next_action_by_outcome": {
    "population-changing": "fit_population_decay",
    "population-preserving": "test_coherence_decay"
  },
  "action": {
    "type": "PROBE_TOY_CASE",
    "crux_id": "c1",
    "input_artifact_id": "probe-source-1"
  }
}
~~~

This is an illustration, not an admitted task. Production records additionally bind task/root/episode IDs, pre-action context hash, feasible outcome IDs, tool/schema version, generated token spans, receipts, budget and artifact hashes. Proposed alternatives and branch plans are policy claims until independently checked. They are not verifier labels.

Use these public action names consistently:

| Action | Purpose and routing rule | Existing/planned mapping |
| --- | --- | --- |
| DEDUCE | Compute from supplied premises; no unnecessary external call | Recorded local derivation or checked calculation |
| SEARCH / READ_SOURCE | Resolve accessible facts, SDK contracts or source conflicts | Versioned retrieval / repository search |
| PROBE_TOY_CASE | Execute a bounded counterexample or discriminating experiment | Sandbox RUN_CODE or SIMULATE with a separate resource profile |
| CLARIFY | Obtain irreducible intent/preferences or required information unavailable through authorized tools | Existing planned ASK alias, backed by a deterministic simulated respondent during training |
| FOLD_EVIDENCE | Propose compact state with source/receipt dependencies | Validated context transformation, not a final answer |
| REVISE_CLAIM / FINISH | Update dependencies or submit the checked artifact | Existing planned research actions |

Require a public reason tied to the current obligation, an expected observation and a stopping/branch condition for information-acquisition actions. Reject malformed calls through the same environment API as other invalid actions. Do not add mandatory tags to native benchmark prompts. A structured field may be rendered as a crux, pivot or fold block, but formatting itself earns no reward.

Estimate action value as expected reduction in task-relevant decision error minus resource cost. Freeze normalized resource scales and weights before the arm; retain raw generated/processed tokens, simulator/device seconds, wall time and user burden separately. A planning preference for deduction/search before a live human question is sensible, but there is no universal strict latency/cost ordering. A cheap simulation may precede search; a costly search may be unnecessary. Model estimates are hypotheses, not measured VoI.

Penalize unnecessary clarification only when the required fact was actually accessible within the recorded permissions and budget and a checked alternative could resolve it. Missing tools, inaccessible sources and unresolved user intent are different cases. Fully specified tasks can finish directly. Training never waits on an actual human: every CLARIFY action has a harness-owned answer source, cost and response rule.

## 3. Coherent world interventions

Implement a separate environment transformation layer after trusted root creation and before public/private projection. Do not rewrite existing smoke data or mutate genuine installed SDKs globally. Each intervention record contains parent root and ancestry IDs, transform/version/seed, world hash, public-document hash, simulator/API hash, private oracle hash, exposure flag and admission receipts.

Initial intervention families are sandbox API aliases/signatures, hardware connectivity and allowed operations, declared noise tables, bit ordering, and syndrome label/sign conventions. Label invented APIs as local contracts. Change every dependent document, tool implementation and reference consistently. Preserve quantum mechanics and mathematical identities; an arbitrary sign change in a physical law is not an API perturbation. Check physical validity of generated channels/noise and identifiability under the permitted observations.

For each eligible root build a canonical world, a decisive intervention and an irrelevant edit. The canonical world also requires consulting external state where appropriate; otherwise the treatment would conflate new facts with new tool requirements. Authoritative local documents are available through the same authorized tools in both conditions. Private seeds, oracle paths and manifest fields cannot leak the hidden answer through the public prompt.

Assign source/ancestry components to partitions before creating siblings. All canonical/mutated/irrelevant siblings stay in the same partition. Hold out transformation families, compositions and source repositories as well as random seeds. A new SDK name alone is not structural generalization. Mutation-treatment and control see the same number of eligible roots and matched total cost; log which instances differ.

Start the isolated mutation arm at a proposed fraction 0.50; use development-only alternatives 0/0.25/0.50 if budget permits. It is not a global default. Terminal correctness remains the first reward. Do not add an automatic -2 penalty for answering without a search: a correct inference from sufficient supplied information is legitimate, and an absent search is not proof of hallucination. Unsupported claims and wrong-world artifacts fail their independent task checks.

## 4. Crux diagnostics and conservative reward experiment

Compute diagnostics in a private scoring service. The actor never sees gold answers, private tests, counterfactual reference completions or future observations. The sampler can use permitted training-only scalar receipts, not private content. Reference/scorer calls, counterfactual branches and unsuccessful probes count toward the experiment budget. Begin with at most two diagnostic counterfactual pairs per eligible episode, selected by a frozen rule; report coverage and missingness.

**Pivotal sensitivity.** Freeze a finite catalogue A of feasible next-action categories, including finish/clarify where valid. At the same pre-query prefix and checkpoint, obtain normalized distributions P+ and P- over A under two independently feasible outcomes. Define S = JS(P+, P-)/ln(2), using natural logs and identical action support. It lies in [0,1]. Define how constrained catalogue probabilities are obtained, or use preregistered independent samples with a fixed parser, smoothing and uncertainty intervals. Catalogue diagnostics are not the unconstrained actor's behavior log probabilities.

High S alone is neither necessary nor sufficient for a useful question. Nuisance observations can provoke unjustified plan changes; a useful verification can confirm the same plan. Check the branch plans and relation to the task obligation independently. Infeasible outcomes, impossible histories, malformed categorical outputs and missing support yield an unmeasurable diagnostic, not an invented zero. Preserve all such records.

**Reference-likelihood proxy.** For a task with an independently validated canonical answer y*, calculate the frozen reference model's mean answer-token log probability with and without the observation:

~~~text
L_ref(y | h) = sum_j log pi_ref(y_j | h, y_<j) / len(y)
C = L_ref(y* | h_prequery + q + o) - L_ref(y* | h_prequery + q + neutral_slot)
~~~

Use a preregistered, semantically empty replacement and matched context/position policy; report residual length/format effects. Start from the prefix before the query in both branches. Do not delete an earlier observation from a completed transcript while retaining later actions that depended on it. Score only the answer tokens, and retain the canonicalization/equivalence rules. Multiple valid programs, reference mistakes and answer-style bias can make this proxy unsuitable; mark it unavailable or use a separately validated executable-continuation comparison. It is **not mutual information or a causal information-gain estimator**. A fixed reference can misvalue genuinely useful evidence. Negative C does not establish that reasoning was wasted.

**Default and optional shaping.** Keep binary terminal SAPO unchanged for R0/R1 and the first cognitive comparisons. Record diagnostic S, C, branch-validity and costs without changing gradients. Only after scorer calibration, an independent reward-hacking review and fresh development checks may a child crux_reward arm use:

~~~text
z_i = frozen_score(valid_probes, evidence_effect, charged_cost) in [0,1]
R_i = correct_i + correct_i * epsilon * z_i,  0 <= epsilon <= 0.05
~~~

Freeze the exact scorer, normalization constants, nonnegative component weights, cost weights and eligibility rules in the child measurement contract before launch. A valid episode with no useful probe has z=0. Missing/corrupt scoring evidence is unresolved; retain it and quarantine the complete group if the declared reward cannot be computed. Compare against the same task/action protocol with outcome-only reward. Successes score [1,1.05], failures 0, so a cheap incorrect answer cannot outrank a verified solution. This is bounded **episode-level** shaping, not the proposed token-local Crux-SAPO algorithm.

Recompute group mean/std and SAPO temperature choice from the resulting episode advantage; detach rewards/statistics. Never multiply a signed advantage by a signed crux score: two negative numbers would reward a failed trajectory. No extra importance ratio is added to the published SAPO surrogate. A token-local estimator would be a further experiment requiring token/span attribution, masks, normalization, unbiasedness limits and independent gradient tests. Do not punish long reasoning merely for length or low reference likelihood; compare completion at explicit total resource budgets.

## 5. Bounded toy experiments

The initial PROBE_TOY_CASE profile proposes at most three qubits, a small allowed library set, deterministic seeds, a two-second CPU execution timeout and recorded memory/output limits resolved by preflight. These are resource ceilings, not a latency prediction. Compile/import/startup and timeout overhead count. Numerical/shot uncertainty and tolerance belong to the task contract. Use a different declared profile for symbolic or larger stabilizer checks.

Require the candidate to name the conjecture, probe input, predicted alternatives and the inference permitted by each result. Execute the source, hash input/output and independently check interpretation. Reject fabricated receipts and repeated no-information probes. A counterexample can refute a universal claim; agreement on N=2 or N=3 leaves the arbitrary-N claim unproved. Before FINISH require the original task's full-size validation, justified proof, or explicitly scoped limitation.

Worked case: in the reviewed amplitude-damping versus phase-damping family, prepare the excited state and measure its population. At identifiable nonzero strengths the result separates population decay from preserved populations; a later coherence test or parameter fit depends on that observation. A separate superposition input checks the fitted channel. At zero strength the mechanisms may be indistinguishable, so the environment must exclude that boundary or accept a justified limitation. This ties crux, probe and revision to a testable scientific decision rather than a long narrative.

## 6. Folding evidence without losing the world

Maintain immutable full receipts outside the active context. A fold contains supported propositions with source/receipt IDs, scope and preconditions, unresolved contradictions, claim dependencies, active cruxes and the next obligation. Protect the original task, mandatory constraints, remaining budget and artifact IDs separately. The proposed 150-token fold cap applies to the compact evidence payload, not to those protected fields or total processed input. Count tokens with the pinned model tokenizer, including visible pointers.

After one to three acquisition actions, or earlier context pressure, the actor may propose a fold. Do not require a fold on every short/direct task. Validate syntax, pointer resolution, source hashes, constraint/contradiction coverage and consistency with receipts before eviction. Mechanical checks cannot prove semantic sufficiency; test mandatory facts with harness-owned obligations and fresh continuation/held-out queries. Actor-written folds remain untrusted evidence claims. If the cap cannot preserve required evidence, retain relevant raw spans and refetch, use a separately declared larger profile, or terminate under the budget contract. Never silently drop a contradiction to fit 150 tokens.

For an admitted fold:

1. Commit the pre-fold context, fold tokens, dependencies, validation result and selected next-context token IDs.
2. Rebuild the next call from protected state + admitted evidence + necessary recent messages. Evict only the selected raw spans from active input; the receipt store remains retrievable.
3. Discard old attention KV caches **and recurrent/DeltaNet states**, then prefill the rebuilt context. Otherwise evicted observations can survive inside hidden state.
4. Record the actual new model input and reproduce those exact per-turn contexts during SAPO training. Fold tokens get actor loss at the turn where generated; later inserted state is context with actor mask zero.
5. On restore, reconstruct the same context/cache boundary from committed receipts. Fail replay parity or unverifiable provenance explicitly.

Compare crux-guided folding with equal-budget generic summaries and a full-context/truncation control under identical raw observation permissions. Report peak active context, total processed input including repeated prefills/refetches, generated tokens, failures, success and wall/device cost. A smaller context window alone is not an efficiency result; folding can increase reprocessing cost.

## 7. Frontier sampling without zeroing difficult skills

The proposed product pass@k * (1-pass@1) * Var(crux_reward) is a hypothesis, not a learnability estimator. Pass@k depends on k and correlated attempts; raw variance can emphasize noisy scorers. Zero observed successes in a small sample does not prove an unreachable task. A saturated crux metric can also have zero variance on useful tasks.

Keep the existing domain/process controller and within-cell R1 rule as controls. In the isolated frontier_sampling arm, use prior-policy training receipts only, a fixed k and frozen lookback/lag. Keep 20% undercovered-family and at least 10% uniform eligible-root allocation. Within the 70% frontier component propose:

~~~text
priority = (p_hat*(1-p_hat)*(1+eta*u_hat) + 0.1/sqrt(n+1))
           / max(relative_complete_group_cost, 0.25)
~~~

Here p_hat uses the existing Beta(1,1) training-success estimator, u_hat in [0,1] is independently verified useful-probe frequency shrunk toward a neutral prior, and eta=1 is a proposed frozen starting value. If probe diagnostics are unavailable, retain the existing frontier rule and label the missing component; do not fabricate low usefulness. Freeze the shrinkage prior/sample minimum before the arm. Report pass@k and crux-score variance as diagnostics first; testing their product requires a separately frozen rule.

Log selection probabilities before dispatch, all groups and exclusions, effective family exposure, probe/scorer uncertainty, policy lag and total generation/verification cost. Diagnose repeated zero-success families and offer checked prerequisite tasks without discarding those skills. Existing process caps, domain bounds and standalone union cap remain binding. No final benchmark/sealed-test feedback enters the sampler. A selected training distribution is a new curriculum distribution, not an unbiased estimate of the old one.

## 8. Hybrid specialization and deferred size reduction

The official [configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json), inspected on 11 November 2026, declares 64 language blocks, 48 linear-attention and 16 full-attention entries, intermediate size 17408 and one MTP hidden layer. Its model type is qwen3_5; do not infer a framework class or module path from the marketing name. Archive the actual immutable revision at preflight. Configuration presence does not prove the training runtime loads or exposes every head.

Keep the baseline's frozen FFNs and rank-32 adapters in both supported language block families. Test unequal ranks/gate targets only after discovering real named_modules, tensor shapes and autograd support. Compare equal trainable-parameter budgets as well as measured end-to-end cost. Verify every intended target, no visual/output/auxiliary updates, and zero-effect initialization. Rank 4 on FFNs is still plasticity and does not imply fact-free learning.

Treat FFN SVD/Fisher/activation sensitivity as diagnostic hypotheses. Any later compression branch needs disjoint memory and supplied-premise reasoning probes, held-out families, rank/retention curves, error controls, a recoverable original checkpoint and real factorized runtime benchmarks. Contrastive probe sensitivity does not certify causal separation of factual and reasoning circuits. Structural edits no longer constitute an ordinary adapter for the unchanged official base and need distinct artifact IDs.

Use the 27B model as a task proposer or private reference within current boundaries if useful. Do not imitate its folded traces or logits in the active RL route. A 4B–8B student and a separately trained VoI/value head are deferred studies; neither is needed to complete the requested 27B adapter.

## 9. One-factor experiments and decision gates

All proposed arms are disabled in the planning example. Begin only after the existing P0–P4 trainer/data/runtime lifecycle gates. Select a matched interactive parent with the same task pool, tools, context budgets, module inventory, terminal scorer and initialization. Freeze one changed factor, paired roots/seeds, total actor plus diagnostic budget, development endpoints and stop rules before dispatch.

| Arm key | Only intended change | Control / decisive check |
| --- | --- | --- |
| crux_observability | Public crux/pivot decision fields | Same actor/tasks/tools without mandatory fields; completion and cost, not tag frequency |
| world_mutation | Reviewed canonical/intervened training exposure | Same parent training exposure at mutation fraction zero; paired fresh-world evaluation |
| toy_probe | Bounded toy action interface | Same sandbox operations and budgets through generic code/simulation; useful falsification and full-contract correctness |
| evidence_folding | Crux-guided admitted context folding | Equal-budget generic summary; full-context diagnostic separately; replay and retention checks |
| crux_reward | Bounded success-only episode bonus | Same observability/actions/scorer calls with binary terminal reward |
| frontier_sampling | Training-only useful-probe priority | Same admitted pool/shares/diagnostics with existing R1 rule |
| hybrid_targets | Inventory-derived asymmetric adapter allocation | Same data/reward/sampler with baseline adapter allocation |

For a folding-only change, freeze any crux fields identically in both arms; for a reward-only change collect the same diagnostics in both arms. Likewise a curriculum comparison cannot also introduce mutated data. Account for every shared prerequisite in the parent contract. Combined treatments require a named follow-up after individual effects are understood.

Use matched actual total-cost ceilings and report realized doses; counterfactual scorer calls are not free. Confirm promising factors with the existing independent-seed/root-cluster protocol. Keep release co-primary thresholds and retention guards unchanged. The new metrics below diagnose mechanisms; they do not replace external quantum/research success.

The sidecar's eight-million actor-token pilot ceiling is aggregate across treatment, matched control and diagnostic/reference generation, within the parent's existing ceiling. Count processed input, tools and wall/device cost separately under frozen caps. This is a proposed suballocation boundary, not new compute or spending authorization.

| Diagnostic | Frozen definition and limitation |
| --- | --- |
| Mutation gap | D=Acc_canonical-Acc_mutated on paired roots; also show absolute mutated accuracy and treatment-control differences in both conditions. D near zero with both accuracies zero is not success. |
| First-probe crux hit | Among all preregistered answerable episodes requiring acquisition with independently adjudicated decisive probe sets, count whether the first acquisition action targets such a set. Missing/invalid/no probe is a miss; ambiguous ground truth is unmeasurable with reconciled counts. Show joint hit-and-final-success rate. |
| Evidence sensitivity | Appropriate response to decisive interventions and invariance to irrelevant edits, with independently checked final artifacts |
| Folding fidelity | Mandatory-constraint/contradiction preservation, pointer recovery and fresh continuation correctness; exact context replay separately |
| Cost/quality curve | Verified success at fixed total processed-input and generated-token budgets, with peak active context, all tools/scorer calls and measured wall/device cost |

Accuracy per 1,000 active tokens may be an additional descriptive ratio with its precise denominator, but cannot be a standalone release gate. Default cognitive gains remain UNKNOWN until these arms run on the real target with authenticated artifacts.

## 10. Implementation handoff

Extend production research contracts and llm_boundary with public/private action projections; add a separate environment transformation registry beside generator/quantum; add fold/cache-boundary support to the new interactive rollout service; and integrate diagnostics through a private scorer. Do not relabel the legacy one-shot rollout or supervised LoRA scripts as these services. The [runbook](FUTURE_CODE_RUNBOOK.md) lists dependencies and production-path acceptance tests.

The only executable addition in this update is planning-contract validation and its rejection tests. The [SAPO paper](https://arxiv.org/html/2511.20347v2) defines the optimizer; this document's crux records, diagnostics, folding policy and reward study are project proposals, not established properties of SAPO or Qwen.

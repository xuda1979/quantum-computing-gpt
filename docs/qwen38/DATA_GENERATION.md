# Data generation, admission, and interactive environments

Applies to [the Qwen3.8-27B governing plan](../QWEN38_27B_ADAPTER_RD_PLAN.md), revision 6.0. This is the implementation specification for the data factory. The existing one-qubit JSONL format is only a fixture; do not present it as the schema below.

## 1. Storage boundaries and record identities

Create separate stores with explicit read permissions:

| Store | Contents | Actor access |
| --- | --- | --- |
| Public training context | Prompt, allowed files, published documentation/method records, tool schema, declared budget | Readable within the episode contract |
| Environment private state | Hidden mechanism, parameters, initial state, oracle inputs, allowed observations | Only through authorized tools |
| Verifier private state | References, private tests, scoring rules, audit fixtures | Never mounted in actor sandbox or retrieval |
| Evaluation vault | Sealed tasks, answers, scorer assets and previous test receipts | Only evaluation service; no training writer |
| Evidence store | Immutable action/observation/program receipts, hashes and verdicts | Actor sees only explicitly released feedback for its current episode |
| Training manifest | Admitted task IDs, ancestry, hashes and partition | Trainer/sampler sees eligibility; actor sees only public projection |

No shared filesystem root, search index, package cache, environment variable, traceback, filename, or network endpoint may expose evaluator assets to the actor. Tools execute candidate code in an isolated process/container with its own resource limits. Candidate code cannot import the scorer process, inspect test definitions, or modify its verdict. Log and test this isolation.

Stable identities:

- `source_id`: original paper, repository, authored contract, or licensed dataset revision.
- `root_id`: one substantive scientific problem/hidden world, fixed before wording variants.
- `ancestry_component_id`: connected component of linked sources/roots/derivatives that cannot cross splits.
- `family_id`: algorithm, mechanism, theorem, witness schema, or workflow family.
- `environment_id@version`: executable transition and observation semantics.
- `task_id`: public prompt variant plus exact private instance and contract.
- `group_id`: one task, initial environment state, behavior checkpoint and G independently sampled episodes.
- `episode_id`, `action_id`, `receipt_id`: unique immutable attempt/action/evidence identities.

Compute content hashes using SHA-256 on canonical UTF-8 JSON with sorted keys, fixed separators, explicit array ordering, and no NaN/Infinity. Hash referenced files by bytes. IDs remain stable when formatting changes; a semantic change gets a new version and hash.

## 2. Source acquisition and split construction

Use authored mathematical constructions, official Qiskit/Cirq/PennyLane documentation, reproducible scientific repositories, and licensed research material. For each source record URL, exact revision/download date, license/terms, permitted use, local hash, citation, and review status. Public availability alone does not establish training rights. A teacher provider's applicable output-use terms belong in its source record. Do not ingest credentials or private personal material.

Register official benchmark repositories and their solutions/tests as **evaluation-only**. Quarantine their derivatives, copied examples, generated paraphrases, prior evaluation discussions, and benchmark-derived repair traces from training and actor retrieval. Upstream pretraining exposure remains unknown; do not claim it was removed.

Build an ancestry graph before generating variants. Union roots sharing a source problem, paper target, repository task, translated solution, nontrivial solution/witness template, or direct derivation. Assign entire components using a saved seeded allocation. Use coarse family labels additionally for explicit held-out-family tracks; shared generic primitives need not merge every task into one component.

Partitions:

1. `train`: only roots available to question generation and online sampling.
2. `dev_tune`: curriculum, hyperparameter and checkpoint selection.
3. `dev_calibration`: confidence mapping and statistical/power design, separate from tuning roots.
4. `sealed_test`: fixed private final evaluation roots, inaccessible to training.
5. `public_benchmark`: pinned native suites and separately named tool adaptations.
6. `quarantine`: ambiguous provenance, leakage, semantic mismatch, invalid/unresolved reference.

The provisional 300 development roots are split 200 tune/100 calibration. This is a planning count; power requirements can increase it before sealing. The 120 diagnostic roots belong to `train` and do not contribute evaluation evidence. After reading a sealed failure for debugging, retire that evaluation version before a new adaptive experiment. Preserve its historical score; use a newly authored/sealed set for a fresh final claim.

Run exact text/code hashing, normalized token/AST comparisons, source and ancestry overlap, semantic-neighbor screening, and manual adjudication of flagged cases. Embedding similarity is a review signal, not a correctness label. Check leakage through documentation examples, teacher prompts, generated method records, memory, logs, caches, and tool fixtures. Save counts and reasons.

## 3. Capability coverage and environment registry

Implement at least two independently reviewed environments per major domain for the first pilot; expand by measured coverage and error taxonomy. Twenty variants of one generator do not satisfy this requirement. Each environment implements:

```python
public_spec = describe(contract_version)
state, observation = reset(instance_id, private_seed)
next_state, observation, tool_receipt = step(state, validated_action)
verdict, reward_or_null, evidence = verify(state, submitted_artifacts)
```

The verifier is an independent service/process; the pseudocode does not grant the actor access to `state`, `private_seed`, or `verify`. Reset reproduces the initial world for a group. Network responses use a versioned snapshot or recorded, hash-checked corpus for reproducible training. Live retrieval is a separately labeled environment version.

| Domain | Initial environment families | Required oracle and failure probes |
| --- | --- | --- |
| Circuit/software semantics | State preparation; composition/control; measurement and bit order; parameter binding; transpiler constraints; SDK migration | Independent matrix/state/distribution reference on tractable cases; wrong controls, reversed order, absent imports, obsolete API, ignored constraints |
| Quantum algorithms | Phase estimation; amplitude amplification; small Hamiltonian simulation; variational eigensolvers; resource estimation | Analytic small cases plus independent implementation; wrong eigenphase/sign, invalid assumptions, false speedup/resource claims |
| Noise/open systems | Channel identification; Kraus/Choi construction; Lindblad evolution; mitigation under declared noise | Correct channel/dynamics reference; non-CP/non-TP maps, incorrect rates, unjustified extrapolation, wrong limiting behavior |
| QEC | Small stabilizer codes; syndrome construction; decoding; correlated-noise diagnosis | Exhaustive small error sets or trusted independent stabilizer simulation; logical/stabilizer confusion, invalid syndrome, train/eval noise leakage |
| Mathematics | Spectral/tensor identities; probability and estimation; optimization/gradients; perturbation/complexity assumptions | Exact/symbolic oracle or reviewed proof; counterexamples, boundary/degenerate cases, incorrect quantifiers and unproved claims |
| Physics/scientific numerics | Dynamics, approximations, numerical stability, conservation laws, finite-size/convergence studies | Analytic limits plus independent converged computation; dimensional error, insufficient convergence, cancellation, sampling uncertainty |
| Scientific software | Repair a quantum repository; reproduce a figure/table; diagnose two inconsistent results | Frozen repository/data and hidden acceptance tests; superficial edits, data contamination, fabricated measurements, unrecorded manual repair |
| Research control | Hidden-mechanism experiments; conflicting source/version evidence; choose the next question; allocate a limited measurement budget | Known latent world or independently adjudicated obligations; confirmation-only tests, ignored contradictions, needless calls, premature certainty |
| Tool/instruction retention | Valid file edits, structured calls, scoped constraints, recovery after a simulated tool fault | Machine-checked action/artifact contracts; hallucinated tools, malformed calls, wrong paths, false completion |

For early statevector checks use roughly 1–8 qubits. Larger Clifford cases can use stabilizer simulation. Larger general circuits need a separate measured simulator/memory budget; density matrices scale differently from statevectors. Numeric tolerance, global phase, observable equivalence, shots, and constraints are part of each contract, not global defaults.

Global-phase invariance is valid only for the stated context. A replacement valid up to global phase in an isolated circuit may change a coherently controlled operation. Include paired contexts and independent witnesses. Likewise, matching one input state or a few observables does not prove full unitary/channel equivalence. Numerical samples cannot certify a theorem unless the contract explicitly asks for empirical evidence.

## 4. Mixtures and coverage accounting

Assign exactly one primary domain and one primary process category, with additional overlapping skill tags. Retain six information-need strata: directly solvable; one missing fact; dependent gaps; conflict/false premise; research-to-code; genuinely insufficient information. Report counts and success by all three axes, plus family, SDK/version, size, difficulty, horizon and oracle type.

The initial 400-group planning window has this feasible joint count matrix:

| Primary domain | Hypothesis/test | Gaps/conflict | Assumptions/revision | Attention/budget | Foundations | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Quantum code | 48 | 30 | 24 | 12 | 6 | 120 |
| Quantum science | 40 | 25 | 20 | 10 | 5 | 100 |
| Agentic research | 24 | 15 | 12 | 6 | 3 | 60 |
| Mathematics/physics | 24 | 15 | 12 | 6 | 3 | 60 |
| General software | 16 | 10 | 8 | 4 | 2 | 40 |
| Instruction/tools | 8 | 5 | 4 | 2 | 1 | 20 |
| Total | 160 | 100 | 80 | 40 | 20 | 400 |

For shorter windows freeze integer quotas first, round the foundations count down, and document unavoidable deviations in other margins. The 32-group canary uses no standalone/foundations groups. Do not drop generated failures to repair a quota. Control future dispatches instead.

Independently audit `standalone_forward_drill`. The union of that flag and foundations must be ≤5% of consumed groups **and** trained actor tokens. Maintain generation and loss-token accounts separately; invalid groups cannot be used to manufacture a favorable denominator. Schedule non-foundation groups first. A conservative token allowance for additional capped material is `floor(non_capped_tokens / 19) - capped_tokens`, checked in each relevant account. Dispatch a capped group only if its declared maximum can fit; otherwise defer it or add eligible non-capped work within the original budget. Never cut an already-scored answer after generation to meet a quota. A negative allowance or breach pauses future training for reconciliation.

Fix a maximum of eight fresh rollout groups per root across one model run initially. Count all semantic variants toward their ancestor cap. This is an exposure ceiling, not a requirement to reuse each root eight times. Three-seed confirmation runs retain separate exposure ledgers and report total source reuse. Increase ceilings only in an explicit development experiment.

## 5. Trusted construction before LLM augmentation

For each environment:

1. Write the contract, private parameter generator, public projection and observation semantics.
2. Produce exact solutions or independently checked references on a bounded instance range.
3. Create positive controls, minimally wrong mutants, adversarial outputs, invalid instances, and infrastructure-fault fixtures.
4. Verify that references pass and **every required mutant** is rejected for the correct reason. A surviving mutant blocks the affected contract until fixed or narrowly rescoped.
5. Generate a small train-only root set with fixed seeds and explicit assumptions. Reject empty, inconsistent, impossible-within-budget, ambiguous, or already-leaking tasks.
6. Allow an LLM to propose wording, new hypotheses, repairs, or new roots. Reconstruct the trusted contract independently; ignore proposal-supplied labels, IDs, split, expected score and permissions.
7. Check semantic equivalence between wording and contract. Free-form questions require recorded semantic adjudication; unresolved proposals remain quarantined. Reviewed deterministic renderers can be reused without reviewing each wording mechanically.
8. Audit lineage and duplicates, then sign admission with contract/reference/test hashes and reviewer/verifier identity.

An executing reference and the student's solution must not share the same untested algorithmic helper as the sole oracle. Use exact algebra, an independent implementation/engine, or adjudicated calculations. Cross-engine agreement can still share a conceptual error; document the independence level. Sanity checks such as norm/trace preservation supplement a specification-specific oracle.

Admission lifecycle: `PROPOSED → CONTRACT_BOUND → REFERENCE_CHECKED → MUTANTS_CHECKED → SEMANTICS_REVIEWED → SPLIT_AUDITED → ADMITTED`. Failure becomes `REJECTED` with a reason; incomplete evidence becomes `REVIEW_REQUIRED`. Only `ADMITTED` train records enter policy updates. Changing an environment or verifier invalidates affected cached admission evidence until rechecked.

## 6. Teacher/proposer instructions

Use this template only with training-eligible source/root IDs:

```text
Role: untrusted scientific task proposer for quantum research training.
Inputs: registered environment schema; approved source spans; allowed primary
domain/process cells; missing coverage cells; difficulty band; tool/episode budget.
Propose at most 4 substantive tasks. For each return:
public problem, explicit assumptions, required artifact, available evidence/tools,
reason it requires the requested skill, proposed lineage, parameter ranges,
candidate reference method, likely wrong approaches, and contract ambiguities.
For paired variants, state one decisive premise change and one irrelevant edit.
Do not quote benchmark tasks/solutions. Do not choose the trusted partition,
assert a positive label, supply hidden tests to the actor, or claim novelty.
Your answer is a proposal; independent admission determines whether it is used.
```

Use a separate reference worker with the bound contract and approved material. Run proposed code/proofs through independent checks. Disagreement between teachers triggers adjudication rather than majority-vote truth. Log provider/model revision, prompts, costs, filters and licenses. A stronger teacher is useful where available; deterministic constructions still permit progress when teacher access is absent.

Add tasks from the **training** failure taxonomy: imports/API, physical model, math assumptions, planning, evidence interpretation, numerical stability, recovery and budget misuse. Do not turn sealed benchmark failures into training prompts. If development feedback drives task design, record that exposure and retain a separate final test.

## 7. Public/private task record

The following is a structural illustration, not an admitted task or a real SHA manifest:

```json
{
  "schema_version": "qresearch-task-v1",
  "task_id": "train-channel-identification-0001-v1",
  "root_id": "channel-root-0001",
  "ancestry_component_id": "authored-channel-source-01",
  "source_ids": ["authored-channel-source-01"],
  "family_id": "single-qubit-channel-identification",
  "split": "train",
  "domain": "quantum_science",
  "process": "hypothesis_falsification",
  "information_need": "dependent_gaps",
  "standalone_forward_drill": false,
  "environment": {"id": "channel_lab", "version": "1"},
  "public": {
    "prompt": "Identify the accessible noise process, estimate its strength, and test your prediction on a new input state.",
    "tool_set_id": "channel-lab-public-v1",
    "artifact_contract": "channel-estimate-with-executed-check-v1",
    "episode_profile": "short"
  },
  "private_ref": "vault://train/channel-root-0001",
  "admission_ref": "evidence://admission/channel-root-0001-v1",
  "status": "ADMITTED"
}
```

Production records additionally require hashes of source, environment image, public context, parameter instance, reference, tests and admission evidence. Private URI values must not appear in the actor's projection; they are opaque to the sampler and accessible only by the authorized service. Define JSON Schema in P0, reject unknown fields and invalid enums, and test serialization round trips. Production schema must distinguish this planning illustration from validated records.

## 8. Interactive rollout and evidence

Every group begins from the same task/initial world. Generate G independent actor trajectories using pinned policy weights and per-episode sampling seeds. The environment seed is private and fixed per instance; it is not resampled to make failed candidates pass. Keep noise and statistical comparison procedures explicit.

The observable actor state contains the task, relevant public source spans, a claim ledger, known/unknown assumptions, checked artifacts, remaining budget, and tool history. Use actions such as `READ_SOURCE`, `SEARCH`, `RUN_CODE`, `SIMULATE`, `CHECK_PROOF`, `REVISE_CLAIM`, `WRITE_ARTIFACT`, `ASK`, and `FINISH`, each mapped to an actual tool schema. A short checked derivation can be an action. Do not require invented rival hypotheses or needless retrieval for an already settled direct task.

Claim states are `UNTESTED`, `SUPPORTED`, `REFUTED`, `CONFLICTING`, or `UNRESOLVED`, with evidence and dependency IDs. Revision invalidates downstream claims that depended on the refuted premise. External memory stores source spans, method preconditions, experiment receipts, and open obligations, rather than unqualified solved-task scripts.

`ASK` is a simulated, bounded information-acquisition action during training; it must have a defined answer source and cost. Do not make automatic training depend on an unanswered live human question. Answerable-task abstention fails; justified insufficiency can pass only where that is the task's true completion contract.

Each step receipt records exact input/output bytes, tool/schema/image versions, action and artifact IDs, start/end times, return status, resource usage, and hashes. Each episode additionally records exact model-input token IDs, generated token IDs, per-token behavior log probabilities, actor masks, termination type, checkpoint ID, all verdict components and total cost. See the training specification for masking and grouping rules.

Never ask the actor to write its own final reward. Store candidate artifacts before scoring. Training feedback may reveal a bounded failure category or public counterexample; hidden ground truth and private tests stay private. Feedback after a failed attempt is part of a new documented environment transition or training task version; do not silently change the original evaluation prompt.

## 9. Worked hidden-channel environment

Start with a reviewed finite family containing amplitude damping and phase damping, bounded strengths away from indistinguishable boundary cases, and a declared experimental interface. The harness draws a mechanism/strength privately. Expose state preparation, measurement basis and shot count, not the answer. A noiseless exact-observation version is the first canary; finite-shot versions require an uncertainty-aware scorer.

The actor proposes alternatives, selects discriminating states/measurements, requests observations, fits parameters, constructs a channel implementation, and predicts an unseen-state result. The private verifier checks mechanism/parameter claims within declared tolerance, the submitted channel against an independent reference, and genuinely fresh predictions. Verify identifiability and attainable accuracy under the shot budget before admission.

Pair it with a changed-mechanism instance where the right investigation/conclusion must change and an irrelevant wording edit where behavior should remain correct. Add an insufficient-information version where the allowed measurements cannot identify the mechanism; its successful result is a warranted limitation plus the needed additional experiment. All siblings remain in one partition.

For the same root, an SFT research branch could collect a verified teacher interaction; the active RL route learns only from the current policy's interactions and rewards. Exact phase/channel solutions are private oracle material, not actor targets.

## 10. Statistical/numerical adjudication

For deterministic exact tasks, return a checked PASS/FAIL with the exact contract. For floating-point tasks freeze dtype, tolerances, conditioning bounds, comparison norm, units, convergence tests and independent reference accuracy. Refuse instances where numerical error could reverse the label without additional computation.

For stochastic tasks freeze shot/seed policy and error control before comparison. Use fresh held-out seeds for success certification. If adaptively increasing shots, use a valid sequential method or a finite prespecified schedule with multiplicity allocation; ordinary fixed-sample intervals repeatedly peeked at are insufficient. If evidence cannot establish the declared success/failure rule, return VOID/UNKNOWN, allocate additional evidence within budget, or quarantine. Do not round ambiguity into success or treat it as a student failure.

The initial SAPO canary uses deterministic/exactly adjudicable rewards. Statistical reward intervals and adaptive verifier allocation are separate experiments and cannot be silently substituted for point rewards.

## 11. Optional verified SFT dataset study

This section answers how to create supervised data. It **does not enable SFT** in the RL-only run contract. Any study uses a separately named arm with its own lineage and cost report.

1. Sample only admitted train roots under the same coverage/ancestry restrictions.
2. Obtain candidate responses from a trusted mathematical construction, executable reference, stronger teacher, or student search. Preserve unsuccessful costs.
3. Execute every claimed tool call. Reject fabricated observations, future-information leakage, unsupported intermediate claims, reference/test access, and incomplete artifacts.
4. Verify the final result independently, including private randomized tests and necessary proof review. Correct output does not establish that every supplied derivation is valid.
5. Retain substantively different successful approaches and verified recoveries, bounded per root. Avoid selecting only long, fluent, easy or near-duplicate demonstrations.
6. Export coding responses and recorded multi-turn trajectories separately. Include concise evidence-supported justifications; teacher hidden reasoning is not required. Preserve real tool observations as context.
7. Apply assistant-only label masks. System/user/tool observations and padding do not become prediction targets. Verify chat-template boundaries, EOS behavior, packing isolation and no truncation.
8. Use one initial pass, LoRA targets matched to the RL arm, and a development-only learning-rate grid. Epoch count is not evidence against memorization. Compare at matched total data-generation/training cost and also report actor-training tokens.

No verified solution means no positive SFT example. A partial result can become a newly scoped, reviewed task, never a mislabeled complete solution. SFT→RL is an additional arm only if the SFT-only comparison and remaining budget justify it; it is not a mandatory warm-up.

If the separate SFT-C arm is explicitly enabled, its initial bounded profile is at most 1,024 verified examples/episodes and 8,000,000 generated target tokens, one pass, global batch 32 episodes, microbatch chosen by actual memory, LoRA rank/targets matched to RL, BF16 where supported, AdamW with the same betas/epsilon/clipping and initial LR 1e-5 (development grid 3e-6/1e-5/3e-5). Optimize token-normalized cross-entropy over assistant target tokens only. Disable cross-episode packing initially; exceedance of sequence limits requires a newly scoped task or a correctly recorded per-turn training representation, not truncation of a verified target. Freeze the exact schedule and budget before the arm; compare total teacher/search/verification and training cost, task support and retention, not just supervised loss. These settings require a distinct study contract and do not alter the active SAPO schema.

## 12. Data acceptance report

Write `data_report.json` with source/partition/family/skill counts; rejection reasons; overlap checks; reference/mutant outcomes; reviewed semantic samples; private-boundary tests; token/length distributions; estimated generation/verification cost; and unresolved items. Reconcile `proposed = admitted + rejected + review_required` and `admitted = train + dev_tune + dev_calibration + sealed_test + public_benchmark` over unique IDs, with quarantine counted before admission. State explicitly when an item is a view rather than an independent record.

Fail admission on a partition conflict, a surviving required mutant, a missing reference/hash, or actor access to private data. Freeze the accepted manifest and its hash before any rollout update. New training data can enter only through a new versioned manifest and a child/continuation contract that preserves the sealed evaluation boundary.

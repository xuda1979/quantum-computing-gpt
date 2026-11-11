# Future Code implementation and execution runbook

Follow [the governing plan](../QWEN38_27B_ADAPTER_RD_PLAN.md), [data](DATA_GENERATION.md), [training](SAPO_TRAINING.md), [adaptive mixture](ADAPTIVE_MIXTURE.md), [cognitive protocol](COGNITIVE_CORE_PROTOCOL.md), and [evaluation](EVALUATION_RELEASE.md). The target is exactly `Qwen/Qwen3.8-27B`. This update adds a planning validator, not a production SAPO trainer. Complete implementation gates before issuing the planned commands below.

## 1. Durable control state

Create `.ai-loop/` in the actual working checkout, with `state.json`, append-only `events.jsonl`, `decisions.jsonl`, task records, experiment records and incident records. Store heavyweight immutable data/evidence/checkpoints in the resolved durable artifact root. State contains:

```text
objective/version; repository commit; base model/revision; active stage/task;
contract hash; task DAG and dependencies; PASS/FAIL/VOID/UNKNOWN gates;
job/rank identities; current/parent checkpoint; actor/behavior/ref identities;
data/scorer/measurement hashes; sampler/mixture/exposure/consumption cursors;
budget spent/remaining; last actual progress; open obligations; next gate.
```

Do not derive completion solely from narrative reports. Reconcile state with real files, process/scheduler state, complete checkpoints and authenticated receipts after every restart. Use one writer per mutable subsystem and atomic file replacement/transactions. A RUNNING marker without a live job is stale, not evidence of ongoing work.

## 2. Implementation task graph

| ID | Dependencies | Write scope / required output | Acceptance evidence |
| --- | --- | --- | --- |
| T01 | None | Contract resolver, artifact/state layouts, typed schemas | Reject wrong model, placeholders in resolved mode, changed hashes, invalid budgets/mixtures and unknown fields |
| T02 | T01 | Source/root registry, ancestry splits, data manifests | Duplicate/ancestry/benchmark leakage injection is rejected; counts reconcile |
| T03 | T01,T02 | Independent environment and verifier API; two reviewed families per initial domain | Reference positives and required mutants; private-state access denied; deterministic reset and numerical contract |
| T04 | T03 | Task admission and proposer/reference workers | Untrusted labels/split overrides ignored; ambiguous wording quarantined; valid new task admitted with receipts |
| T05 | T01,T03 | Qwen3.8 runtime/processor adapter, module inventory, synchronous sampler | Actual target hardware probability/cache parity; expected trainable paths; exact actor token receipts |
| T06 | T01,T05 | SAPO/reward/group reductions and trainer | Independent finite difference, distributed/mask/length/group tests; production loss imported |
| T07 | T01,T02,T03,T05 | Frozen evaluator, benchmark registry and authenticated output ledger | Reference/mutant checks; stale/wrong adapter and missing outputs detected; native/adapted distinction |
| T08 | T02,T04,T06,T07 | Fixed/frontier samplers and optional dev-driven mixture controller | Fixed control; no sealed input; constraints/projection/exposure/cost tests; action logs |
| T09 | T05,T06,T08 | Checkpoint transaction, scheduler integration and recovery | Real save/kill/reload second-step comparison; no duplicate group update; disconnect job detection |
| T10 | T03–T09 | Data diagnostic, untouched base, real lifecycle and canary | P1–P4 artifacts/health gates with actual checkpoint/device receipts |
| T11 | T10 | Short-horizon/interactive/confirmation experiment execution | Frozen arms, measured dose/cost, guardrails, dev-only selection |
| T12 | T11 | Sealed evaluation, report, numerical release package | Complete statistics/denominators, fresh-load smoke, manifest identity, honest aspiration status |

Build narrow slices in dependency order. Data and evaluator engineering can proceed while hardware/download dependencies are blocked. Do not wait for thousands of tasks before testing real token/reward/checkpoint wiring. The missing trainer is a work item, not a reason to run the legacy SFT script.

Optional cognitive implementation tasks follow the same gates. T13–T19 extend production research interfaces; do not overload the legacy one-qubit fixture contracts or reinterpret old scores. Offline data/scorer fixtures can be built early, but target-model treatments wait for P0–P4 and a matched interactive parent.

| ID | Dependencies | Write scope / required output | Production-path acceptance evidence |
| --- | --- | --- | --- |
| T13 | T01,T03,T05 | Research contracts and public/private llm boundary: crux/action/pivot records, cost-aware routing, CLARIFY alias | Direct tasks need no probe; multiple cruxes valid; invalid actions rejected; simulated clarification deterministic; private labels excluded |
| T14 | T02,T03,T04 | Separate world transformation registry beside generator/quantum | Document/tool/oracle agreement; physical-validity and identifiability checks; canonical/mutated/irrelevant siblings cannot cross partitions; wrong-world guesses rejected |
| T15 | T03,T13 | Bounded PROBE_TOY_CASE tool profile and receipts | Timeout/memory/output enforcement; fabricated receipts rejected; counterexample interpretation checked; small-case agreement cannot satisfy unsupported general-N claim |
| T16 | T05,T09,T13 | New interactive rollout context builder and folding transactions | Invalid/missing pointers and lost contradictions block eviction; caches and recurrent state reset; pinned-tokenizer cap; fresh-process exact-context/replay parity; overflow/refetch charged |
| T17 | T07,T13 | Private pivotal-sensitivity / reference-likelihood diagnostic service | Normalized common action support; infeasible outcome rejected; identical prefixes; no downstream orphan text; irrelevant/misleading evidence and answer-form sensitivity controls; actor cannot access gold |
| T18 | T06,T08,T17 | Separate bounded reward and useful-probe curriculum children | One changed factor; independent reward-bound/sign/group/temperature tests; scorer failures quarantine groups; before-dispatch probabilities, exploration, lag and complete cost ledger |
| T19 | T05,T07 | Hybrid module/rank allocation comparison | Inventory-derived targets, matched trainable budgets, gradient/probability/cache checks, base/visual/auxiliary freeze and real retention/cost evidence |

For T16, actor folds get loss only where originally generated; later reinserted state is context. For T18, retain whole-episode SAPO credit in the first reward study; token-local Crux-SAPO needs a separate estimator and new acceptance tests. Compression, student distillation and MTP value heads are not dependencies for this program.

## 3. Planned command surface: implement before use

Provide a single future module `quantum_posttrain.rd` with subcommands below. It is **not present in the reviewed baseline**. Every subcommand must support `--help`, a resolved/planning config distinction, JSON output, stable exit codes, and explicit artifact paths. Refuse an unsupported phase rather than returning synthetic success.

| Planned command | Input | Required output |
| --- | --- | --- |
| `resolve` | Planning JSON, allocation/runtime discovery | Resolved candidate contract, unresolved list, immutable identity locks; no training launch |
| `data build` | Train-eligible sources/environment contracts | Proposed roots/tasks, references and admission receipts |
| `data audit` | Candidate manifests | Split/contamination/reference/mutant report and frozen accepted manifest |
| `runtime preflight` | Target model and runtime contract | Module/gradient/sampling/cache/topology/cost evidence |
| `eval baseline` | Frozen dev manifest and untouched model | Authenticated dev samples and failure taxonomy |
| `train lifecycle` | Tiny real profile | Two-step uninterrupted/resumed comparison and numerical adapter |
| `train canary` | Accepted train roots and real runtime | Eight updates, complete receipts, health and early dev report |
| `train run` | Frozen experiment, arm and stage | Rollout/update/evidence logs, checkpoints, budget/curriculum state |
| `eval run` | Frozen measurement and candidate | Per-task receipts plus full statistical/cost report |
| `release package` | Accepted artifact/evidence IDs | Explicit file list, checksums, fresh-load tested numerical package |
| `status` / `resume` | Durable run identity | Reconciled actual job/checkpoint state or safe continuation |

Suggested CLI shapes after T01–T09 are implemented and tested:

```bash
python -m quantum_posttrain.rd resolve --config configs/qwen38/rd_contract.example.json --output runs/qwen38/contract.candidate.json
python -m quantum_posttrain.rd data build --config runs/qwen38/contract.candidate.json --output runs/qwen38/data-candidate
python -m quantum_posttrain.rd data audit --config runs/qwen38/contract.candidate.json --input runs/qwen38/data-candidate --output runs/qwen38/data-audit
python -m quantum_posttrain.rd runtime preflight --config runs/qwen38/contract.candidate.json --output runs/qwen38/preflight
python -m quantum_posttrain.rd eval baseline --config runs/qwen38/contract.resolved.json --split dev_tune --output runs/qwen38/base-dev
python -m quantum_posttrain.rd train lifecycle --config runs/qwen38/contract.resolved.json --output runs/qwen38/lifecycle
python -m quantum_posttrain.rd train canary --config runs/qwen38/contract.resolved.json --output runs/qwen38/canary
python -m quantum_posttrain.rd train run --config runs/qwen38/r0.contract.json --arm R0 --output runs/qwen38/R0
python -m quantum_posttrain.rd eval run --config runs/qwen38/measurement.frozen.json --candidate runs/qwen38/candidate.json --output runs/qwen38/final-evaluation
python -m quantum_posttrain.rd release package --config runs/qwen38/release.contract.json --output runs/qwen38/release
```

`contract.candidate.json` can remain planning while data/preflight work resolves artifacts. T01's resolver must emit `contract.resolved.json` only when all required identity, budget, implementation and compatibility gates are satisfied; later arm/measurement/release contracts derive from it with explicit parent hashes. Bootstrap gate work has its own bounded allocation and cannot recursively require a successful full-training contract. Record actual filenames emitted; never handwrite PASS to satisfy a resolver.

Existing commands available now for **CPU fixture validation only**:

```bash
python -m pip install -e '.[test]'
python scripts/validate_cognitive_contract.py
python -m unittest discover -s tests -v
python scripts/validate_research_ideas.py --trials 10000
python scripts/validate_larger_scale.py --roots 10000 --variants-per-root 10 --output runs/scale-validation
```

These commands do not train Qwen3.8 or certify its runtime. Do not call `train_qwen_lora.py` for the active RL route. Do not advertise the old `rollout_qwen.py` as the new benchmark service.

The validator is implemented now. It validates both baseline and cognitive planning schemas plus cross-contract sums, one-factor identity, budgets and protected-state requirements. It rejects duplicate/nonfinite JSON, unknown fields, objective substitutions, multiple enabled factors, absent lineage/control, unsafe folding, final-test feedback and signed score multiplication. `--require-resolved` must fail; even syntactically valid hashes authenticate no artifacts.

The [sidecar example](../../configs/qwen38/cognitive_contract.example.json) uses an aggregate eight-million actor-token planning ceiling across the proposed treatment and its matched control, within the parent pilot ceiling. Diagnostic/reference generation is charged to that aggregate account, with all processed input, tools, wall/device time recorded separately. It is not additional spending permission. A production child resolver must bind the exact task/action/scorer/context versions, one-factor settings, parent/measurement hashes and a real suballocation; it cannot launch a changed objective through the unchanged binary-reward v1 schema.

## 4. Automatic stage procedure

1. Read the governing plan, inspect current implementation and existing `.ai-loop` state, and reconcile prior artifacts/jobs. Record the current user objective and authorized resource envelope.
2. Resolve source/model/runtime identities and implement missing interfaces. Run static/schema tests and focused contract tests first. Pin dependencies from a proven environment, not floating `latest` packages.
3. Build the smallest reviewed multi-domain diagnostic and test scorer isolation/mutants. Produce train/dev split manifests before policy updates. Prepare independent development probes.
4. Profile actual target-model forward/generation/backward and save/load; prove actor probability/cache parity. Compute realistic total-cost projections from measured throughput.
5. Freeze the real lifecycle/canary contract within the authorized allocation. Run the two-update save/resume control, then the 32-root canary. Preserve every generated candidate, invalid group and cost.
6. If healthy, run fixed-mixture R0. Evaluate on dev only. Diagnose unsupported tasks, bad rewards, truncation, gradients and API mismatches before scaling. Keep the untouched base and known-good checkpoints.
7. Run R1 with the same objective/pool/shares. Run R1-MIX only under its frozen dev-feedback rule. Fit skills/prerequisites from development evidence and keep a fixed-mixture control. New admitted data creates a versioned continuation; sealed sets remain untouched.
8. Add interactive horizon/memory/delegation factors progressively. Keep matched interactive controls. Apply retention and budget stop rules. Use three seeds for a stable method claim.
9. Freeze selected candidate(s), scorers, complete task IDs and statistical analysis. Run final tests through the isolated evaluation service, including native/adapted public tracks and fresh private roots.
10. Package only authenticated artifacts. Write the actual improvement/retention/IBM/frontier decisions and missing cells. Fresh-load and hash-verify the package before handing it off.

If an aspiration is unmet, retain the best valid artifact and publish the measured boundary. Plan a child experiment from authorized **development** diagnostics. Never change the goal denominator or continue training on a final test while reusing its generalization claim.

## 5. Remote execution and incident playbooks

Launch through the available scheduler or persistent service and save job/rank IDs. Use structured heartbeats and step events. Poll often at boot/checkpoint/error boundaries, then back off during healthy phases. A process being alive without step progress is a possible stall; inspect logs/rank collectives and resource state before deciding.

| Observation | Immediate action | Resume condition |
| --- | --- | --- |
| Browser/SSH session lost | Query scheduler/service and latest event; do not start duplicate job | Existing job healthy, or last complete checkpoint reconciled |
| Download/API transient | One unchanged retry, then test connectivity/auth/cache/mirror eligibility | Bytes/revision verified; provider identity unchanged |
| OOM | Preserve checkpoint/logs; reduce microbatch or concurrency under a child runtime profile | Same objective/group reduction, target hardware canary repeated |
| Nonfinite loss or ratio mismatch | Stop updates; save diagnostic cohort and first causal evidence | Independent reproduction and fixed cause; loss/parity gates pass |
| Zero task signal | Audit scorer, sample support, task assumptions and lengths | Valid learnable training slice and nonzero task advantages demonstrated |
| Test/reference/adapter mismatch | Void affected measurement; quarantine cached verdicts | Correct artifact/instrument re-authenticated and candidates re-evaluated |
| Mixture oscillation/retention breach | Pause, preserve controller trace and last accepted checkpoint | Fixed-control comparison or declared recovery child passes dev guards |
| Inconclusive stochastic verification | Spend the declared additional evidence budget or mark unresolved | Valid certification under frozen error-control rule |
| Budget/storage exhausted | Commit partial state, report achieved dose and remaining obligations | New actual allocation or lower-cost validated child profile |

Capture incident ID, exact command/contract, code/runtime/data/model hashes, first failing action/cohort, logs around first cause, process/resource state, last complete checkpoint, reproduction, hypothesis, fix, tests and recovery decision. Avoid repeated blind retries or installing random dependency versions into a running experiment.

## 6. Acceptance checklist for implementation handoff

- The official Qwen3.8-27B revision and actual architecture are authenticated; no MoE-era settings silently remain.
- All missing planned interfaces have real implementations and production-path tests.
- Data ancestry, benchmark isolation, private filesystem/network boundaries and candidate identity are enforced.
- Independent positives/mutants certify every active deterministic verifier; numerical uncertainty remains explicit.
- SAPO derivative, reference separation, actor masks, distributed scaling and probability reconstruction pass independent checks.
- Target-model adapter movement and real checkpoint/resume are measured; all consumed groups have at-most-once semantics.
- Question balancing uses permitted development inputs, bounded changes, coverage/exposure caps and logged controller state.
- Any enabled cognitive treatment has independently checked crux/probe labels, coherent world contracts, private diagnostic isolation, exact fold/cache replay and a matched one-factor control. Missing mechanism evidence blocks that mechanism's claim.
- Complete native/adapted and short/long denominators, uncertainty, costs and missing cells are reported.
- A numerical adapter and tested inference configuration are delivered with reproducible lineage and honest claim status.

The implementation is complete only when these checks refer to real artifacts. This plan by itself passes none of the target-model training or capability gates.

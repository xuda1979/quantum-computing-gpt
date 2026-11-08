# Evaluation-driven balancing of training question types

Companion to [the governing plan](../QWEN38_27B_ADAPTER_RD_PLAN.md). This section implements the user's requirement to rebalance questions according to evaluation evidence. It is a proposed **outer-loop controller over training tasks**, separate from SAPO's parameter update and within-skill difficulty sampling. Its benefit must be tested against the fixed-mixture control.

## 1. Why the weakest benchmark is not a sampling rule

A low aggregate score can reflect wrong APIs, weak mathematics, missing source access, invalid evaluation, or tasks outside current learnability. It does not uniquely identify a useful training question type. A high score can hide weak rare skills or near-duplicate tasks. Increasing one bucket can also harm retention or transfer to several other buckets.

The relevant precedents are [Skill-It](https://arxiv.org/abs/2307.14430), which uses skill dependencies and evaluation difficulty to select training skills, and [DoReMi](https://arxiv.org/abs/2305.10429), which learns domain weights from a proxy/reference objective for pretraining. They motivate studying data-mixture control; DoReMi is not itself a benchmark-score feedback algorithm, and neither paper validates this project's quantum SAPO controller.

Optimize a predeclared multi-skill objective under resource and retention constraints, not the currently lowest leaderboard number. Separate **what to train**, **how hard the instances should be**, and **how many candidates to sample**.

## 2. Three evaluation roles and a read boundary

| Stream | May choose mixture/checkpoint? | Use |
| --- | --- | --- |
| `dev_tune` skill diagnostics and genuine official development splits | Yes, under this frozen controller | Rebalancing, debugging, checkpoint selection |
| `dev_calibration` | Only for prespecified calibration/power procedures | Uncertainty mapping and final-analysis design; not routine controller feedback |
| `sealed_test` and public suites designated native final test | No | Final unbiased-to-project-adaptation comparison under the frozen candidate |

If a public suite has no development partition, construct independent skill-matched development tasks. Do not repeatedly consume its final test and still describe it as a pristine holdout. Public scores already observed historically can motivate broad hypotheses; disclose that history and rely on fresh sealed tasks for generalization claims. Never feed test failures, solutions, tests, or retrieved answer pages into the task proposer.

The controller receives aggregate development skill records, uncertainty, counts, cost and failure categories, not full final-test outputs. Its input schema rejects `sealed_test` and `public_benchmark` result IDs. Baseline final-test outputs may be computed by an isolated evaluator and sealed, but cannot be inspected by training/controller code before candidate freeze.

## 3. Map evaluation failures to training skills

Maintain a versioned many-to-many map from evaluation items to skills and training environment cells. Split the same original task and all its variants together. Initial skill taxonomy:

| Evaluation symptom | Skills to diagnose | Candidate training environments |
| --- | --- | --- |
| Qiskit Standard good, Hard weak | Imports, API discovery, version interpretation, standalone function construction | Documentation-assisted migration and repair; independently authored API tasks |
| Quantum circuit runs but is wrong | Control/global phase, bit order, state/channel semantics, assumptions | Counterexamples, paired contexts, randomized semantic checks |
| Quantum science/math weak | Spectral/tensor reasoning, probability, optimization, valid proof steps | Checked small instances and prerequisite investigations with relevant method records |
| Physics calculation inconsistent | Modeling assumptions, units, dynamics, approximations, numerical convergence | Hidden mechanism identification and independently checked numerical experiments |
| SciCode/reproduction weak | Scientific concept-to-code translation, data handling, numerical validation | Fresh repository/reproduction tasks with hidden artifacts and convergence checks |
| Agent completion weak despite good single-turn code | Planning, source selection, observation interpretation, recovery, memory | Multi-turn dependency, contradiction, fault recovery and bounded question-selection tasks |
| Retention decline | Forgotten general code/math/tool competence | Previously declared retention environments and fixed-reference regularization study |

A diagnosis must be supported by real receipts. Malformed benchmark setup or missing tools first goes to infrastructure repair, not to a bigger training bucket. Use oracle-source/tool controls to separate retrieval from interpretation. Avoid changing a mixture based solely on an LLM's label of a failure.

Store map `M[j,k]` from training cell j to evaluation skill k. Initially it is a reviewed prerequisite/coverage hypothesis with nonnegative row-normalized weights. Where affordable, estimate transfer through short one-factor training interventions from the same checkpoint and independent dev probes. Such interventions must count their cost. Simultaneous progress after mixed training is not causal proof that every sampled cell helped.

## 4. Evidence for each control window

Canary uses fixed quotas. Thereafter evaluate at most once every eight completed rollout cohorts, with a declared minimum data dose between evaluations. Freeze controller probe IDs/seed schedule and record repeated use. Periodically add fresh **development** roots through reviewed versions to audit dev overfitting; never quietly move sealed roots into development.

For each skill k record one first-attempt outcome per independent development root for the controller, sample size n_k, paired change from the prior checkpoint, an uncertainty interval, invalid/missing rate, and failure taxonomy. Stochastic multi-attempt scores can be additional diagnostics but cannot replace the declared first-attempt measure midway.

Each root is weighted once; correlated variants do not inflate n_k. Use cluster-aware uncertainty for source-linked roots. With fewer than 30 independent roots, mark the skill low-confidence and expand diagnostics rather than aggressively reallocating. With >5% VOID/UNKNOWN, repair or complete evaluation before using a new weakness signal. Missing results keep the prior allocation and create an evaluation obligation.

For the first implementation require every mandatory coarse skill group to pass these count/measurement gates before changing the domain mixture. If any fails, freeze the entire previous domain vector and collect/repair development evidence within budget. Fine-grained exploratory slices are excluded from controller inputs until separately admitted as sufficiently measured skills. This removes ambiguity about reallocating away from an unmeasured capability.

Before the first adaptive window freeze a phase target tau_k, skill importance lambda_k, retention margin, and evaluation cost cap. A default phase target is `min(1, base_dev_score_k + 0.10)`; this is a curriculum target, not a promised benchmark gain. Lambda weights are reviewed according to the mission, rather than proportional to benchmark size. Keep quantum coding/research co-primary and math/physics/agent/retention constraints explicit.

Initial group importance weights are quantum coding 0.30, quantum research 0.25, mathematics 0.15, physics 0.10, agentic research 0.15, and general retention 0.05. Divide each group's weight equally among its currently measured constituent skills, unless a different split is frozen before the experiment. The 200-root dev_tune starter allocates 40/40/30/30/30/30 independent roots respectively; rare subskills remain descriptive until adequately sampled. These evaluation weights are distinct from training domain shares and the final co-primary statistical weights.

## 5. Initial controller rule

This is a transparent starting heuristic; compare it to fixed weights and simpler deficit-only weighting. Let s_k be measured dev accuracy, reliability `rho_k=n_k/(n_k+50)`, normalized deficit `d_k=max(0,tau_k-s_k)/max(tau_k,0.1)`, and credible forgetting `f_k=max(0,lower(previous_best_k)-upper(current_k)-0.02)`. Use consistent recorded intervals; a noisy point decline alone is not a confirmed regression.

Compute a training-cell need:

\[
a_j=\sum_k\lambda_k M_{jk}\rho_k(d_k+2f_k).
\]

Maintain a separate estimated learnability/transfer benefit and relative total cost for each cell. Initially benefit is neutral, not invented from a proxy model. For cells with repeated zero-success support, route effort to reviewed prerequisites/scaffolded investigations and diagnosis rather than allocating unbounded hard rollouts. When causal transfer evidence exists, use a nonnegative, uncertainty-shrunk benefit estimate, with at least 10% exploration retained. A smaller proxy model can propose rankings only after their ordering transfers on a bounded Qwen3.8 pilot.

Set a proposal proportional to need × usable benefit / normalized cost, mix it with 20% initial coverage allocation, and smooth: `v = 0.8*w_old + 0.2*w_proposal`. If all valid needs are zero or evidence is insufficient, retain w_old and prioritize fresh diagnostics. The within-cell frontier sampler still handles difficulty; do not confuse low dev score with a command to sample only impossible training roots.

Use the training specification's fixed-allocation, median end-to-end group-wall-time normalization for the initial cost denominator, floored at 0.25. Charge source/teacher construction separately and include it in the experiment's total-cost comparison. Resolve and freeze numerical skill-importance weights and any transfer-benefit estimator before enabling this arm; absent transfer evidence uses the neutral multiplier 1.0. At least 10% within-cell uniform exploration remains active. Low-confidence or missing skills retain their previous need/allocation contribution and trigger more diagnostics rather than receiving fabricated zero error.

Initial allowed domain bounds:

| Domain | Initial share | Lower | Upper |
| --- | ---: | ---: | ---: |
| Quantum code | 0.30 | 0.20 | 0.45 |
| Quantum science | 0.25 | 0.15 | 0.40 |
| Agentic research | 0.15 | 0.10 | 0.30 |
| Mathematics/physics | 0.15 | 0.10 | 0.30 |
| General software | 0.10 | 0.05 | 0.20 |
| Instruction/tools | 0.05 | 0.03 | 0.10 |

Intersect those bounds with `w_old ±0.05` per window. Project v onto the bounded simplex, not normalize-then-clip: solve `w_j=clip(v_j+c, lower_j, upper_j)` for c by monotone bisection so weights sum to 1. Check feasibility first (`sum(lower)≤1≤sum(upper)`). Limit total L1 change to 0.10 by interpolating between the feasible old and projected weights; this moves at most five percentage points of probability mass per window. Recheck sums, bounds and numerical tolerance.

Keep the process-column allocation and the ≤5% union cap from the data specification unchanged. Construct a feasible domain×process allocation with explicit integer apportionment (e.g. bounded min-cost flow with fixed column counts), then apply exposure and generated/loss-token caps. Empty or unreviewed cells cannot be filled with mislabeled data. If constraints cannot be satisfied, retain the last feasible mixture, record the infeasibility, and request new **training-only** admitted roots.

New rules/bounds/targets/maps require a child experiment. Weight changes produced by this already frozen rule are expected within-run controller state and are logged, not silently edited configuration. R1 retains fixed domain shares; name the adaptive-domain treatment `R1-MIX` and hold the SAPO objective, pool, within-cell sampler and other factors fixed for its comparison.

## 6. Measure whether rebalancing helped

Before each update save the prior mixture/controller state, input dev record hashes, candidate proposal, projection trace, final quotas, predicted benefit/cost, diagnostic reasons and remaining budget. After the next fixed training dose evaluate the same skill probes plus fresh development controls. Preserve every window, including negative outcomes; a coincident score rise is not sufficient causal attribution.

Compare R1-MIX with fixed-mixture R1 from the same starting checkpoint and matched total cost. Also compare a deficit-only controller if resources allow. Report macro skill success, co-primary scores, worst-skill retention, family transfer, effective task advantages and total cost. Check whether allocation merely improves repeated dev items. Stable gains require independent seeds and the frozen final evaluation in the release specification.

If retention credibly breaches its margin, pause the affected candidate and diagnose before another expensive window. Roll back to the last accepted development checkpoint for a child recovery experiment with preserved lineage; do not erase the failed window. If a new mixture has no useful gain for three scheduled windows, restore the fixed-mixture control as the next baseline and investigate data quality, prerequisites, reward and optimization. Do not chase a single benchmark indefinitely.

## 7. Worked balancing example

Suppose development diagnostics show strong supplied-import quantum coding, poor independent API discovery, correct arithmetic but weak channel reasoning, and declining general Python retention. Confirm this with enough independent roots and valid infrastructure. Increase the reviewed API/semantic/channel-investigation cells, preserve the general-code floor and schedule retention recovery. Apply bounded shifts; do not convert the entire dataset to IBM-like questions.

If fresh math training improves quantum coding in an isolated pilot, update the reviewed transfer map with that evidence. If hard physics tasks yield no successes, first supply prerequisite contexts or smaller experiments. If a score is low because a simulator package is missing, repair the environment. The next evaluation determines whether the action helped; the sealed IBM suite remains outside this loop.

# SAPO training and target-runtime specification

Companion to [the governing Qwen3.8-27B plan](../QWEN38_27B_ADAPTER_RD_PLAN.md). All settings here are explicit starting hypotheses, not measured optima. The production trainer and environment integration must first be implemented under the [runbook](FUTURE_CODE_RUNBOOK.md).

## 1. Base, processor, and runtime resolution

Resolve `Qwen/Qwen3.8-27B` through the official repository to an immutable commit, then download a snapshot and record file hashes. Load the configuration and supported model/processor classes from the selected framework. Inspect actual architecture, module names, special tokens, template, context controls and cache implementation. Do not rename the old model flag and assume compatibility.

Freeze the official weights, visual encoder/projector, embeddings/output head, and any auxiliary prediction heads initially. Train only explicit language-model LoRA targets. Inventory the hybrid recurrent blocks and attention blocks separately. Start with supported projection modules in **both** language block families; inspect actual linear/module types and enumerate exact paths. Never silently adapt only one block family because a suffix matcher misses the other. If one family lacks training support, classify that as a separate reduced-scope experiment rather than claiming the intended architecture was adapted.

Initial adapter proposal: rank 32, alpha 64, dropout 0, no trainable biases. Exclude embeddings, output head and visual/auxiliary modules. Save exact selected paths/shapes/dtypes/counts and verify every intended family is represented. FFN adapters and ranks 16/64 are later one-factor development comparisons. Validate zero-effect initialization against the untouched base before the first update.

Select the runtime from the actual assigned hardware:

- CUDA: evaluate a pinned TRL or verl path with SAPO and supported Qwen3.8 model kernels; use a native framework forward/generation reference before a fast rollout engine.
- Ascend 910B/910C or successors: lock driver, firmware, CANN, torch, torch-npu, distributed/HCCL, trainer, rollout engine and PEFT versions. Verify recurrent-state autograd, distributed sharding, adapter attachment, and save/resume on this exact checkpoint. Another Qwen model in the official support matrix is insufficient evidence.
- CPU: run data/loss/schema tests and small-model lifecycle fixtures. CPU success cannot pass target-model/device gates.

Choose precision, placement and sharding through measured preflight. BF16 is the initial target where correctly supported. Do not use `device_map=auto` as a distributed training strategy. Count actor, behavior snapshot, frozen reference, rollout replicas, adapters, optimizer, activations, recurrent state, caches, communication buffers, and simulator processes. Shared base weights with separately selected frozen/trainable adapters are allowed only after attachment and concurrency-isolation tests. No full fine-tuning, quantized training, multi-token speculative generation, or asynchronous rollout in the first causal experiment.

Persist an image digest, dependency lock with hashes, accelerator topology/rank map, launch environment, model/processor/template hashes, kernel selections, and adapter inventory. Test the actual scheduled multi-rank topology. Inference support is distinct from backward, checkpoint and multi-turn support.

## 2. Policy and token contract

Distinguish three policies:

1. `pi_theta`: current trainable adapter.
2. `pi_old`: immutable behavior checkpoint used to generate one rollout cohort.
3. `pi_ref`: untouched official checkpoint, frozen throughout the run.

Synchronize/acknowledge behavior weights on every rollout worker before dispatch. Record a weight hash and generation number with each group. Begin with synchronous cohorts and no stale-policy reuse. Do not mix greedy, teacher, repaired, or previous-checkpoint candidates into a fresh on-policy group.

Use the pinned processor's official chat template. Preserve the same thinking/history behavior in actor generation and training reconstruction. The generation parser can extract a public action, but the receipt retains **all generated actor tokens**, including reasoning tokens, even if not shown to tools. Do not rescore a cleaned/rewritten response in place of the sampled response.

Training sampler initially uses temperature 1.0, top-p 1.0, top-k disabled, no repetition/presence penalties, and no beam search or hard grammar filtering. This full-support choice makes the sampling law and raw model probabilities coincide; it is a project training choice, not the model card's recommended benchmark decoding. If any transform/mask is later used, define and record the actual behavior distribution and compatible ratios. Never store raw logits as though they were probabilities after a different sampling transform.

For each turn retain exact conditioning token IDs, generated token IDs, generation configuration, token log probabilities, attention/position metadata and any relevant recurrent-state contract. Actor masks are 1 only on generated policy tokens, including intended stop tokens; they are 0 on system/user/context/tool observations and padding. A public assistant role marker added by the template is context unless it was actually generated. Packing cannot join different episodes or leak their state.

An episode can have several independently reconstructed turn contexts. Compute each actor token probability conditioned on the exact observation history seen at generation time. Sum actor-token losses over turns, then normalize once by that episode's total actor-token count. Averaging each turn separately would change the objective. Truncating hidden history in training while rollout used it is prohibited.

Implement a versioned context policy before long episodes: reserve the current task/constraints, remaining budget, active claim/dependency ledger, unresolved contradictions and artifact IDs; then add relevant source/method spans and recent tool exchanges. Reserve at most 2,048 generated tokens per short decision and 4,096 per long decision, within the aggregate episode caps. Retrieve older receipts by ID when needed. Deterministic selection and any generated summary must be recorded; actor-written summaries count as generated actor tokens. Never remove a mandatory constraint silently. If the protected state plus generation headroom cannot fit, use a declared compaction action or terminate with the contract's budget status. Generation and training consume exactly the same selected context. Reset scratch and episodic memory between roots; reusable method memory excludes solved-test artifacts.

For the optional [evidence-folding arm](COGNITIVE_CORE_PROTOCOL.md), retain all immutable receipts, validate source pointers/constraints/contradictions and record exact post-fold token IDs before eviction. Discard both attention KV caches and recurrent/DeltaNet state and prefill the selected context; otherwise the supposedly evicted observation can remain in hidden state. Fold tokens are actor tokens where originally generated, and context tokens when inserted into later turns. Test fresh-process reconstruction at each fold boundary.

For hybrid recurrent models, prove reset between episodes, full-forward versus incremental-cache agreement, padding/packing isolation, and correct gradient propagation. Never reuse another episode's recurrent state. Long-context support on a model card does not establish affordable or correct long-context RL.

## 3. Exact SAPO objective

For each task q, generate G independent complete episodes from `pi_old` with the same initial world. The initial canary uses G=8. Let `R_i` be the terminal point reward, `m_it` the actor mask, and `T_i=sum_t m_it > 0`. Compute population group standard deviation (ddof=0):

\[
\bar R=G^{-1}\sum_iR_i,\quad s_R=\sqrt{G^{-1}\sum_i(R_i-\bar R)^2},\quad
\hat A_i=\operatorname{stopgrad}\left[(R_i-\bar R)/(s_R+10^{-6})\right].
\]

Detach rewards, group statistics, behavior and reference log probabilities. With the exact recorded actor history h_it:

\[
r_{it}(\theta)=\exp(\log\pi_\theta(a_{it}|h_{it})-\log\pi_{old}(a_{it}|h_{it})),\qquad
f_\tau(r)=\frac{4}{\tau}\sigma(\tau(r-1)).
\]

Set tau_pos=1.0 for positive advantages and tau_neg=1.05 otherwise. These reproduce the published convention, not a reciprocal-temperature convention. SAPO policy loss for a complete group is:

\[
L_{policy}(q)=-\frac1G\sum_i\frac1{T_i}\sum_t m_{it}f_{\tau_i}(r_{it})\hat A_i.
\]

Average equally over eligible complete groups in the optimizer minibatch. Do not introduce another factor of r: `f(r) * r * A` is a different loss. Do not detach f; its derivative provides the policy gradient. For scalar checking, `df/dr = 4*sigmoid(z)*(1-sigmoid(z))`, z=tau*(r-1), and `df/dr=1` at r=1. The surrogate's absolute value is not a success metric and can include advantage-dependent constants. [SAPO paper](https://arxiv.org/html/2511.20347v2).

Initial reference regularization uses beta=0.01 and the sampled-token proxy:

\[
d_{it}=\log\pi_{ref}(a_{it}|h_{it})-\log\pi_\theta(a_{it}|h_{it}),\qquad
k_{it}=\operatorname{expm1}(d_{it})-d_{it},
\]
\[
L=L_{policy}+\beta\,\operatorname{mean}_{q,i}\left[T_i^{-1}\sum_t m_{it}k_{it}\right].
\]

This is a declared empirical regularization surrogate on sampled behavior tokens, not an exact full-vocabulary KL certificate. Keep its implementation and reduction fixed across compared arms. A zero-beta control is a separate factor. Do not confuse the fixed reference with pi_old. Use float32 for log probabilities, ratios, statistics, reductions and stable sigmoid/expm1 evaluation. Pause on nonfinite ratios/proxies; never silently clamp extreme ratios into an apparently healthy run.

Before model training, independently test finite differences/autograd for positive, negative and zero advantage; on-policy derivative; constant groups; masks; lengths; padding/packing; permuted candidates; variable worker partitions; gradient accumulation; reference beta=0; and a deliberately corrupted log-probability receipt. A production-loss import must be exercised, not a copied helper tested against itself.

## 4. Reward contract and validity

Initial reward is deterministic terminal correctness: `R=1` for a verified complete result and `R=0` for a candidate-caused failure. For an insufficient-information contract, a warranted limitation plus the specified evidence can be a complete result. For an answerable task, unsupported abstention fails. Candidate syntax errors, invalid actions, declared-budget exhaustion, misleading claims and incomplete artifacts fail under the task contract.

Verifier/infrastructure failures return VOID/UNKNOWN and `reward=null`. Retain receipts and spending. Re-adjudicate the same immutable candidate if possible; never regenerate only the failed candidate until it looks favorable. A group with unresolved candidate measurement is ineligible for an update. After bounded recovery, quarantine the full group, preserve the reason, and dispatch another group under the declared sampling rule. Track the exclusion rate by skill, length and hardware; candidate failures cannot be mislabeled infrastructure faults.

Do not reward imports, formatting alone, explanation length, number of hypotheses/citations, confident language, or self-judged novelty. Initially disable semantic LLM-judge reward, discovery bonuses, process shaping, anti-rote penalties, and interval-valued advantages. Add them only through independent scorer validation and a separate matched experiment.

An optional later efficiency reward is `R=correct*(1+0.05*(1-normalized_cost))` with cost clipped to [0,1] under the declared episode budget. All verified successes remain above all failures. Compare it to the binary-reward control at matched budgets. Cost variation among failed episodes is not positive evidence of solving ability.

All-pass/all-fail groups have zero task advantage only when their **total rewards are constant**. The regularizer may still move parameters. Record those updates separately; they do not establish useful task learning. Do not remove an individual bad candidate from its comparison group.

## 5. Cohorts, batching, and learning rates

Initial LoRA optimizer: AdamW, LR 1e-6, betas (0.9,0.999), eps 1e-8, weight decay 0; global gradient clipping 1.0. Linear warm-up for the first min(10, ceil(0.05*planned_optimizer_updates)) updates, then constant LR. Planned update count is resolved before launch. Development-only LR alternatives are 3e-7 and 3e-6. Dropout stays disabled for rollout/training probability parity; `eval()` does not disable gradients by itself.

| Profile | Groups/cohort | Episodes/group | Optimizer minibatches/cohort | Maximum cohorts | Purpose |
| --- | ---: | ---: | ---: | ---: | --- |
| Tiny real lifecycle | 2 | 4 | 1 | 2 | Save after update 1, compare uninterrupted/resumed update 2 |
| Target canary | 4 | 8 | 1 | 8 | 32 roots, 256 complete episodes, eight updates |
| Short pilot | 32 | 8 | 4 | 64 | Up to 2,048 groups, 16,384 episodes, 256 optimizer updates |
| Research continuation | 16 | 8 | 4 | 128 | Longer episodes; larger cohorts only after profiling |

Each group is used in exactly one optimizer minibatch, with no additional epochs or old-trajectory replay. Pi_old is frozen at cohort start; earlier minibatch updates can make later minibatches off-policy, which is where gating differences matter. A single optimizer step immediately after fresh on-policy generation cannot by itself demonstrate a SAPO-versus-clipping advantage.

For the short pilot, each optimizer minibatch contains 8 groups × 8 episodes = 64 episodes globally. If DP world size is 8 and microbatch is one episode/rank, gradient accumulation is 8; recompute for the actual world size. Preserve group reward statistics even when candidate episodes are distributed across ranks. Correct the distributed loss scale using global denominators; do not average unequal local means. Variable-length episodes may be split into turn sequences, but the declared episode/group weights remain unchanged.

The max-cohort counts are ceilings, not a promise to finish within resources. Also cap pilot generated actor tokens at 8,000,000 and research tokens at 32,000,000 initially, shared across compared work under the allocation contract. Stop at the first reached wall/device/token/storage budget and report the achieved dose. If arms receive different realized doses, report cost-normalized curves; do not call them a matched fixed-dose comparison.

Lifecycle/canary/initial pilot use training seed 17. Seeds 17/29/43 are the later confirmation design, not an instruction to triple every preliminary job. Freeze per-arm spending before dispatch; the first two-arm pilot can split its eight-million-token ceiling equally into four million per arm, including all candidate generation. Additional arms require a new explicit allocation within the aggregate envelope. Evaluation and teacher/source construction costs remain charged in their declared accounts.

## 6. Difficulty selection and evaluation-driven balance

Keep two distinct controls:

- **Within-skill difficulty:** use training-only success/cost receipts to select learnable roots.
- **Across-skill mixture:** use authorized development evaluations under the [adaptive-mixture specification](ADAPTIVE_MIXTURE.md). Sealed/native-test results cannot drive this controller.

R0 uses fixed domain/process shares and stratified random roots. R1 uses the same shares and, within each feasible cell, 70% frontier-weighted selection, 20% undercovered-family selection, and 10% uniform eligible-root exploration. Estimate training success using a Beta(1,1) prior over at most the latest eight groups and eight cohort versions; record exact counts, uncertainty and checkpoint lag. A starting frontier priority is `[p_hat*(1-p_hat)+0.1/sqrt(n+1)] / max(relative_cost,0.25)`. Normalize inside the stratum and respect exposure/token caps. This is a sampling heuristic, not a theorem.

For this initial rule, relative cost is median end-to-end wall seconds per complete group divided by the pooled median under the same worker/device allocation. Include rollout and verification; use a declared warm-up estimate before enough measurements exist and label it projected. Report device-hours and other resource costs separately. A different cost metric or allocation changes the controller contract; do not add dollars, tokens and seconds without a declared conversion.

Keep all eligible groups in training, including constant-reward groups; the initial controller predicts before dispatch rather than rejecting groups based on observed success. DAPO-style mixed-outcome-only rejection is an optional separate arm with its selection law, acceptance rate and total wasted generation reported. Do not assert it estimates the unchanged original distribution.

If success support remains absent, check contracts, API/source availability, parsing and oracle validity before increasing rollout count. Then reduce horizon/size, expose a permitted method/API reference, or train prerequisite investigations. Record every scaffold. Evaluate on unscaffolded dev tasks as well; hints must not contain final answers, private tests, or teacher traces to imitate. Never add SFT silently.

## 7. Interactive stages and resource profiles

Begin with short repair/assumption/measurement tasks within process categories 1–4. Extend one factor at a time: more dependent gaps, longer action horizons, conflicting evidence, memory pressure, question selection, and finally bounded delegation. Preserve short-task efficiency and foundation retention throughout.

| Aggregate episode limit | Short research | Long workflow |
| --- | ---: | ---: |
| Tool/external calls | 8 | 32 |
| Generated actor/worker tokens | 8,192 | 32,768 |
| Observation tokens | 16,384 | 65,536 |
| Processed input tokens, including rereads | 131,072 | 524,288 |
| Active context plus generation headroom/call | 16,384 | 32,768 |
| Wall seconds | 180 | 1,800 |
| Concurrent workers | 2 | 3 |
| Delegation depth | 1 | 1 |

These are planning caps; actual feasibility is established by canary. Native benchmark budgets override these only in the separately scored native track. All branches, failed calls and repeated context count. Initially train the coordinator only with pinned frozen workers. Worker output is an environment observation and gets no coordinator actor loss; its generation cost still counts. All-role adaptation is a new experiment with per-role policy identities and masks. Use whole-episode terminal advantage initially; turn-local/branch-credit methods require their own estimator and tests.

## 8. Health gates, stopping and recovery

Before updates, require fresh-rollout/full-forward log-probability parity with initial BF16 tolerances p99 absolute error ≤0.05 and max ≤0.5 on actor tokens, plus deterministic generation agreement on selected short probes. Calibrate tighter tolerances where feasible. A failing tolerance is a blocked compatibility gate, not a reason to loosen it after looking at results; any revised tolerance needs an independent numerical justification and a new contract.

Initial pause thresholds: any nonfinite loss/gradient/proxy; zero actor tokens; behavior/version mismatch; scored/train-token mismatch; data leakage; corrupt receipts; three consecutive optimizer steps with zero adapter movement despite nonzero task advantages; mean sampled-reference proxy >0.10 for two updates; or >1% of actor tokens with absolute log importance ratio >1.0 before a minibatch. Log full distributions. These are conservative starting operational thresholds, not guarantees of stability.

For task-learning health, if fewer than 10% of the latest 32 valid groups have nonzero task advantages, diagnose learnability, coverage and reward validity before increasing spending. A tiny diagnostic with fewer groups reports the raw count. If independent dev success worsens beyond the retention rule, preserve a checkpoint and diagnose. Three scheduled development evaluations without practical gain trigger a coverage/reward/runtime/optimization investigation rather than automatic longer training.

Write checkpoints atomically with adapter, optimizer/scheduler/scaler, model/runtime contract, RNG states on every rank, sampler/adaptive-mixture state, exposure ledger, consumed-group IDs, rollout weight version, budget ledger and parent checkpoint. Flush a manifest only after shards are complete and hashed. Checkpoint after each canary cohort and every eight pilot cohorts or 15 minutes, whichever comes first at a safe boundary.

On a crash, inspect the actual job and committed state. Discard incomplete checkpoint transactions; retain evidence. Commit optimizer state and consumed-group cursor consistently so a resumed job cannot apply a group twice. Recompute/re-adjudicate only from immutable receipts and verified hashes. Repeat the interrupted step only from the last complete pre-step state. A hardware/backend/model/schema change requires a new compatibility gate and lineage record.

## 9. Required training report

Report planned/consumed/updated/quarantined groups; nonzero-advantage groups; actual optimizer updates and actor tokens; raw successes; reward and independent-dev curves; gradients/adapter movement; probability/KL/entropy diagnostics; failures/truncation; source/family/mixture/exposure distributions; runtime/device hours; teacher/verifier/tool/evaluation cost; and resume events. Distinguish measured, missing and projected quantities. Lower training loss or higher reward without frozen dev improvement is not sufficient for promotion.

## 10. Proposed crux and hybrid experiments

Follow the [cognitive protocol](COGNITIVE_CORE_PROTOCOL.md) for independently valid counterfactual observations, finite-catalogue normalized JSD, private length-normalized reference-likelihood diagnostics, action routing, fold replay and one-factor controls. The likelihood diagnostic is not mutual information. Neither a large divergence nor a negative likelihood change alone establishes probe quality or wasted reasoning.

Keep binary terminal rewards and the exact loss above unchanged in the initial arms. A later reward-only child may use `R_i=correct_i + correct_i*epsilon*z_i`, with independently frozen `z_i` in [0,1] and epsilon at most 0.05. Failures remain zero, successes remain [1,1.05]. Recompute whole-group statistics and select temperatures from the resulting detached advantages. Do not multiply signed scores and advantages, introduce token-local credit implicitly, or add another importance ratio. Corrupt/missing required reward evidence leaves the group unresolved; it cannot become a favorable zero-cost sample.

The v1 baseline run schema intentionally cannot launch this changed reward. The cognitive sidecar describes a proposed study, including matched control and parent/measurement hashes; a tested production resolver and new resolved child contract must express the exact scorer and training objective before any launch. Counterfactual/reference and fold/refetch costs share the aggregate study budget with both compared arms.

For asymmetric hybrid targets, inventory real modules and supported gradients rather than hardcoding the feedback's gate names or layer count. Compare with the baseline allocation at matched trainable-parameter budgets and total cost. FFNs remain frozen in the baseline; this is not a guarantee against memorization. Keep compression, smaller students and MTP value heads out of the active adapter run.

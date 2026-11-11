# Validation of the Qwen3.8 R&D plan updates

## 11 November 2026: cognitive protocol and executable planning checks

Source baseline: `468bcabef3a9f72bbef43c5ddd0f981fb74998ef`. Scope: revised R&D specifications, a disabled-by-default cognitive planning sidecar, its schema/validator and CI tests. Existing training, quantum verifiers, generators and stored task partitions are unchanged. No target-model training, device compatibility, numerical Qwen adapter or new capability score is claimed.

Executed locally:

```bash
python scripts/validate_cognitive_contract.py
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python scripts/validate_research_ideas.py --trials 10000
PYTHONPATH=src python scripts/validate_larger_scale.py --roots 10000 --variants-per-root 10 --output runs/cognitive-plan-scale-validation
git diff --check
```

Results:

- Planning validator: `PLANNING_VALID`, `training_authorized=false`, `identity_evidence=shape_only`. It checks both schemas and cross-contract invariants. `--require-resolved` exits 2; this tool cannot authenticate artifacts or authorize a launch.
- All 19 unit tests pass, including eight new cognitive-contract test methods. The new cases cover the disabled default and all seven isolated child plans; multiple changed factors; missing control/lineage; objective/model substitutions; unsafe folding; final-test input; signed score multiplication; deferred compression/student/MTP flags; unknown fields; parent budget/share tampering; duplicate JSON keys; NaN/Infinity and overflowed numbers; malformed input; and the planning-only CLI.
- Existing mathematical property runner passes 10,000 advantage-bound and 10,000 surrogate-bound trials, the 257-point phase grid and its additional fixtures. This is unchanged fixture mathematics, not an implemented crux reward or policy-learning result.
- Existing scale runner passes generation, root-disjoint splitting, exact verification and serialize/reload for 100,000 synthetic records from 10,000 roots. Counts are 79,930 train, 9,910 dev and 10,160 test. These temporary synthetic outputs do not replace stored research partitions.
- Active local Markdown paths resolve against the checkout and authenticated upstream manifest, including the unchanged historical Word file. The pinned test extra is `jsonschema==4.26.0`; CI installs it and runs the planning validator before all tests.
- Independent review of the actual changed/new specifications, schema, validator and tests found no blocking issue. The verifier separately ran the eight cognitive-contract tests and planning CLI successfully; target-model gates remain unmeasured.

The [cognitive protocol](COGNITIVE_CORE_PROTOCOL.md) separates first-probe hit, absolute mutated accuracy, folding fidelity and full-cost completion from release success. Planning schema validation does not implement or certify any of those measurements. Target-model gradient/cache/reset/resume, coherent transformed environments, private diagnostic isolation, actual cognitive training arms and benchmark results remain UNKNOWN until their production gates run.

## 8 November 2026: previous update record

Date: 8 November 2026. Source baseline: `16259e3c7867e625c7b18e385e3c4e71710198f6`.

Scope: documentation and planning configuration. The production Qwen3.8 SAPO trainer, research environment service, adaptive controller and benchmark adapters remain implementation tasks. No target-model training, device compatibility, new capability score or numerical Qwen adapter is claimed by this PR.

### Existing executable regression checks

Executed locally from the unchanged repository Python implementation:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python scripts/validate_research_ideas.py --trials 10000
PYTHONPATH=src python scripts/validate_larger_scale.py --roots 10000 --variants-per-root 10 --output runs/pr-plan-scale-validation
```

Results: 11 unit tests passed; the mathematical property runner passed 10,000 advantage-bound trials, 10,000 surrogate-bound trials, a 257-point phase grid and its additional fixtures; the scale runner passed generation, root-disjoint splitting, exact verification and serialization/reload for 100,000 synthetic records from 10,000 roots. These are small-matrix/synthetic harness checks, not language-model efficacy tests.

### New specification/configuration checks

- Draft 2020-12 JSON Schema is valid; the planning example validates.
- Negative mutations reject the wrong base, a falsely resolved planning file, enabled SFT/teacher imitation, final-test feedback, sealed controller inputs, adaptive R0, excessive foundations/domain jumps, negative budgets, unknown fields, an incompatible sampler and inconsistent R1 selection. Research stage requires sealed/calibration manifest identities.
- A synthetic filled-in contract tests resolved **shape** only. Schema validation authenticates no file, hardware allocation, model checkpoint or measurement. The future resolver must verify those against real artifacts.
- Domain/process shares, evaluation skill weights, within-cell weights, initial bounds, the 400-group joint table and canary/pilot batch identities reconcile. The canary is 32 groups and 256 episodes; the pilot optimizer minibatch is 64 episodes globally.
- Independently calculated central finite differences agree with the documented SAPO scalar derivative at five positive ratios for both temperatures. This checks the written formula, not an implemented trainer.
- The documented bounded-simplex/L1 projection procedure preserves sum, bounds and movement limits over 5,000 randomized control windows. This checks the proposed procedure, not a deployed controller or evidence of better learning.
- Local Markdown path links in the active and newly archived documents resolve; the unchanged historical Word file is checked against the source repository manifest.
- `git diff --check` passes. Historical Qwen3.6 plans are marked and retained, while active entry points identify the official Qwen3.8-27B target.

Reproduce basic schema validation with an environment containing `jsonschema`:

```bash
python -c 'import json,jsonschema; from pathlib import Path; p=Path("configs/qwen38"); jsonschema.Draft202012Validator.check_schema(json.loads((p/"rd_contract.schema.json").read_text())); jsonschema.validate(json.loads((p/"rd_contract.example.json").read_text()),json.loads((p/"rd_contract.schema.json").read_text()))'
```

The [runbook](FUTURE_CODE_RUNBOOK.md) specifies the independent production-path tests still required before any expensive training. In particular, a planning/schema pass cannot substitute for target-model gradient, probability, hybrid-cache, resume or frozen-benchmark evidence.

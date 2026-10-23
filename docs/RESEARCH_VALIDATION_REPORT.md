# Research and Scale Validation Report

**Run date:** 2026-10-23  
**Seed:** `20261023`  
**Purpose:** exercise the implemented harness and test mathematical properties from the research specification. This is not an end-to-end language-model training result.

## Results

| Check | Result |
| --- | --- |
| Unit suite | 11 tests passed, including the CPU adapter train/save/reload and all 16 verified rollouts |
| Advantage interval property | 10,000 randomized reward boxes passed; all corners were also checked for groups up to size 5; point intervals recovered centered advantages |
| Clipped-surrogate bound | 10,000 randomized groups passed, including token probability ratios above the clipping interval; observed surrogate discrepancy stayed within the stated bound |
| Bound decomposition and allocation limit | The `B = sum(k_j * width_j)` identity, nested-interval certified-mass monotonicity, and equal-length `lambda=0` width-reduction-per-cost limit passed |
| Coherent-control witness | 257 phases from 0 to 2π matched the analytic process fidelity, witness-state overlap, and control-X expectation |
| Corpus stress | 100,000 records from 10,000 roots and 10 variants per root passed exact equivalence, root-disjoint split, JSONL serialization/reload, and contract checks |
| Corpus split | 79,930 train records / 9,910 dev records / 10,160 test records, assigned by root |
| Corpus runtime and memory | 8.7 seconds end-to-end; approximately 91,040 KB peak resident memory in this environment |
| Static and command checks | `compileall` passed; smoke, generator, and validation command help parsed successfully |

The corpus used deterministic one-qubit `I/X/Z/H` circuits of up to 32 gates with known identity pairs inserted. It occupied 40,230,020 bytes across the three JSONL partitions. This is a moderate harness-scale plumbing test, not a benchmark of Qiskit, Lean, multi-qubit statevector simulation, asynchronous workers, or a production distributed verifier.

## What the research checks establish

The randomized checks support implementation consistency for the scalar interval formulas: under the supplied reward intervals, sampled true centered advantages stayed in the derived ranges, nonzero conservative coefficients retained their sign and did not exceed the true magnitude, and sampled clipped-surrogate discrepancies did not exceed the proposed bound. They also check that the bound decomposition is algebraically consistent, that refinement did not reduce the tested certified coefficient mass, and that the allocation formula reduces to width reduction per cost in the equal-length `lambda=0` case.

The coherent-control example confirms a contract distinction: `I` and `exp(iφ)I` have unit standalone phase-insensitive process fidelity, while their controlled extensions differ according to `cos²(φ/2)`. This validates the small-matrix witness calculation, not a model’s ability to discover phase bugs.

## Limits and next experiment

- No Qwen checkpoint, PyTorch/Transformers/PEFT stack, or suitable accelerator was available here. The NumPy smoke adapter is only an orchestration test; it does not validate Qwen LoRA training, GRPO, long-context memory, or model quality.
- The property tests assume the supplied intervals contain the true verifier rewards. They do not establish interval coverage, calibration under adaptive sampling, or validity of any probabilistic verifier.
- The allocation checks verify the proposed equations and the required degeneration control. They do not show that `lambda > 0` is better than uniform, uncertainty-only, or cheapest-first scheduling on real verifier traces.
- The coherent-phase witness is analytically constructed. Witness transfer and zero-tool distillation require model generations, matched training controls, and a root-disjoint held-out evaluation; they remain unmeasured.
- The synthetic corpus has one qubit and one small gate alphabet. Realistic scale testing still needs measured task distributions, multi-qubit simulation, timeout/cancellation behavior, concurrency, queue backpressure, and resource accounting on the target hardware.

For a publishable efficacy claim, run a frozen candidate-pool replay with fully resolved rewards and matched verifier budgets for the acquisition baselines, then run the preregistered model comparisons on held-out roots and witness schemas. Report the complete denominator, cost, coverage, accuracy, and uncertainty—not only accepted examples.

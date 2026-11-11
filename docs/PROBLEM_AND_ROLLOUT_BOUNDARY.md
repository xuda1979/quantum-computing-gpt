# Problem generation and adapter rollout

Use the harness as the source of truth for contracts, root identities, split assignment, reference labels, and final verification. Start with deterministic generators and curated, licensed roots. The generator records provenance and builds variants only through declared transformations. An LLM can propose wording, candidate circuits, repairs, or new root ideas; it cannot author its own trusted contract or grant itself a label.

For a proposal tied to a registered root, the trusted harness reconstructs the task contract, ignores proposal-supplied labels and split fields, and independently checks the proposed answer. A free-form LLM-written problem statement remains `REVIEW_REQUIRED` until a reviewer confirms that it expresses the registered contract and the dataset audit finds no leakage or duplicates. Only adjudicated tasks enter a training manifest.

The adapter supplies candidate responses during rollout. The harness supplies the prompt distribution and judges each response. In an RL run, sample complete prompt groups from a frozen training pool, generate each answer with the pinned behavior-policy adapter, and attach verifier evidence to those exact candidates. Never use test tasks for rollout training or let an LLM judge replace an executable oracle when one exists.

| Stage | Owner | Output | Can assign a positive training label? |
| --- | --- | --- | --- |
| Contract and root registry | Harness / reviewed source | Pinned task specification | No |
| Problem and mutation generation | Harness; optional LLM proposals | Candidate tasks with lineage | No |
| Adjudication | Trusted verifier and, for free-form wording, human review | Accepted task and verified target | Yes, after all gates pass |
| Adapter rollout | Pinned policy model | Candidate answers and behavior log probabilities | No |
| Reward and evaluation | Trusted verifier | PASS / FAIL / VOID / UNKNOWN and evidence | Yes for eligible training groups; sealed test remains evaluation-only |

The included exact one-qubit generator is a pipeline fixture, not a research benchmark. Its 16 examples test code paths and file contracts. They are too small and too related to support generalization claims or meaningful Qwen training results.

## Proposed cognitive environment boundary

Apply the [current cognitive protocol](qwen38/COGNITIVE_CORE_PROTOCOL.md) in a separate versioned research environment. Its action/probe/fold fields do not extend the legacy one-qubit Task format implicitly. Untrusted crux statements, branch predictions and folded evidence cannot assign their own relevance, validity or reward. The verifier owns decisive-probe labels and gold-reference diagnostics; only permitted scalar feedback can cross into training.

World transformations update a sandbox's public documentation, tools, hidden state and independent oracle consistently. Group canonical/mutated siblings by ancestry before splitting. Exclude benchmark-derived solutions, private seeds, oracle paths and diagnostic counterfactual answers from actor search. A correct direct answer from supplied premises remains valid; lack of a tool call is not itself hallucination.

Keep immutable full receipts when folding active context. Validate provenance, mandatory constraints and contradictions before eviction, then reset both attention caches and recurrent state and prefill the recorded new context. Tiny probe success does not satisfy a larger scientific contract without its required proof or full-size checks.

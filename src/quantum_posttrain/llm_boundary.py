"""Policy boundary for optional LLM-authored task proposals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .contracts import Task
from .quantum import canonical_circuit, equivalent_up_to_global_phase


@dataclass(frozen=True)
class ProposalDecision:
    status: str
    task: Task | None
    reason: str


def adjudicate_proposal(
    proposal: dict,
    trusted_roots: dict[str, tuple[str, ...]],
    verifier: Callable = equivalent_up_to_global_phase,
) -> ProposalDecision:
    """Verify a candidate answer against a harness-owned root and contract.

    LLM-supplied target labels, split, family, and contract fields are ignored.
    Free-form problem text remains REVIEW_REQUIRED because semantic equivalence
    to the registered prompt cannot be established by this tiny exact oracle.
    """
    if not isinstance(proposal, dict):
        return ProposalDecision("REJECTED", None, "proposal must be an object")
    root_id = proposal.get("root_id")
    if not isinstance(root_id, str) or root_id not in trusted_roots:
        return ProposalDecision("REJECTED", None, "root is not in the trusted registry")
    try:
        source = tuple(trusted_roots[root_id])
        candidate = tuple(proposal["candidate_gates"])
        if not verifier(source, candidate):
            return ProposalDecision("REJECTED", None, "candidate answer fails trusted verification")
        target = canonical_circuit(source)
    except (KeyError, TypeError, ValueError) as exc:
        return ProposalDecision("REJECTED", None, f"malformed proposal: {exc}")
    task = Task(
        task_id=f"llm-proposal-{root_id}",
        root_id=root_id,
        family_id=f"verified-{root_id}",
        split="train",
        source_gates=source,
        target_gates=target,
    )
    if proposal.get("problem_text"):
        return ProposalDecision(
            "REVIEW_REQUIRED", task,
            "answer verified; human semantic review and deduplication required before training on LLM wording",
        )
    return ProposalDecision("VERIFIED", task, "answer verified under the registered harness contract")

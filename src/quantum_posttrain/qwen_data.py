"""Prompt formatting shared by SFT, rollout, and trusted evaluation."""

from __future__ import annotations

import json

from .contracts import Task


def render_prompt(task: Task) -> str:
    gates = ", ".join(task.source_gates)
    return (
        "Simplify this one-qubit circuit while preserving its unitary up to global phase. "
        "Allowed gates are I, X, Z, H. Return only a JSON object with one key, gates, "
        "whose value is a list of gate names.\n"
        f"Input circuit: [{gates}]\nOutput JSON:"
    )


def render_target(task: Task) -> str:
    return json.dumps({"gates": list(task.target_gates)}, separators=(",", ":"))


def parse_candidate(text: str) -> tuple[str, ...]:
    """Parse the exact output contract; do not repair malformed model outputs."""
    value = json.loads(text.strip())
    if not isinstance(value, dict) or set(value) != {"gates"}:
        raise ValueError("output must be a JSON object with exactly the 'gates' key")
    gates = value["gates"]
    if not isinstance(gates, list) or any(not isinstance(g, str) for g in gates):
        raise ValueError("gates must be a list of strings")
    if any(g not in {"I", "X", "Z", "H"} for g in gates):
        raise ValueError("candidate used a gate outside the pinned contract")
    return tuple(gates)

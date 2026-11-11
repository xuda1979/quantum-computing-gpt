#!/usr/bin/env python3
"""Validate planning shape and invariants; this command never authorizes training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "qwen38"


def load_json_text(text: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number: {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON object key: {key}")
            value[key] = item
        return value

    value = json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_object)
    # JSON exponents such as 1e999 can overflow float without invoking
    # parse_constant; reject these before schema or budget comparisons.
    if _nonfinite(value):
        raise ValueError("non-finite JSON number")
    return value


def _schema_errors(value: Any, schema_path: Path, name: str) -> list[str]:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise RuntimeError("Install the pinned test dependencies with pip install -e '.[test]'") from exc
    schema = load_json_text(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    errors = sorted(Draft202012Validator(schema).iter_errors(value),
                    key=lambda error: str(list(error.path)))
    return [f"{name}:{'/'.join(str(part) for part in error.path)}: {error.message}"
            for error in errors]


def _nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_nonfinite(item) for item in value.values())
    if isinstance(value, list):
        return any(_nonfinite(item) for item in value)
    return False


def validate_contract(cognitive: Any, rd: Any) -> list[str]:
    """Check both schemas and cross-contract constraints, without resolving identities.

    A syntactically valid SHA is only a proposed identity here. File existence,
    hashes, compatibility, measurement validity and implementation readiness
    remain outside this planning check.
    """
    if _nonfinite(cognitive) or _nonfinite(rd):
        return ["contracts contain non-finite numbers"]
    errors = _schema_errors(cognitive, CONFIG / "cognitive_contract.schema.json", "cognitive")
    errors += _schema_errors(rd, CONFIG / "rd_contract.schema.json", "rd")
    if errors:
        return errors

    enabled = [name for name, active in cognitive["arms"].items() if active]
    selected = cognitive["experiment"]["arm"]
    if enabled != ([] if selected == "disabled" else [selected]):
        errors.append("exactly the selected single factor must be enabled")
    if cognitive["crux_reward"]["enabled"] != (selected == "crux_reward"):
        errors.append("crux reward must be enabled only for its own single-factor arm")
    if selected == "disabled" and any(value is not None for key, value in
                                      cognitive["experiment"].items() if key != "arm"):
        errors.append("disabled baseline must not declare a child control or lineage")

    budget = cognitive["budgets"]
    if budget["pilot_actor_token_cap"] > rd["budgets"]["pilot_actor_token_cap"]:
        errors.append("cognitive pilot token cap exceeds parent planning budget")
    if budget["episode_actor_token_cap"] > budget["pilot_actor_token_cap"]:
        errors.append("episode actor budget exceeds pilot token budget")
    if cognitive["evidence_folding"]["token_cap"] >= budget["active_context_token_cap"]:
        errors.append("fold token cap must be below active-context planning cap; "
                      "runtime must measure protected context and generation fit")
    for name, shares in (
        ("domain_shares", rd["data"]["domain_shares"]),
        ("process_shares", rd["data"]["process_shares"]),
        ("evaluation_skill_importance", rd["curriculum"]["evaluation_skill_importance"]),
        ("within_cell_mixture", rd["curriculum"]["within_cell_mixture"]),
    ):
        if not math.isclose(sum(shares.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
            errors.append(f"parent {name} must sum to one")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=CONFIG / "cognitive_contract.example.json")
    parser.add_argument("--rd-contract", type=Path, default=CONFIG / "rd_contract.example.json")
    parser.add_argument("--require-resolved", action="store_true",
                        help="fail: this validator supports planning only")
    args = parser.parse_args()
    try:
        cognitive = load_json_text(args.contract.read_text(encoding="utf-8"))
        rd = load_json_text(args.rd_contract.read_text(encoding="utf-8"))
        errors = validate_contract(cognitive, rd)
        if args.require_resolved:
            errors.append("planning-only sidecar cannot resolve or authorize a training run")
    except (OSError, ValueError, RuntimeError) as exc:
        errors = [str(exc)]
    report = {
        "status": "INVALID" if errors else "PLANNING_VALID",
        "training_authorized": False,
        "identity_evidence": "shape_only",
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

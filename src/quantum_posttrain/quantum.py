"""Exact tiny-circuit oracle for smoke tests; not a general simulator."""

from __future__ import annotations

import numpy as np

GATES: dict[str, np.ndarray] = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
    "H": np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2),
}
GATE_NAMES = tuple(GATES)


def unitary(gates: tuple[str, ...] | list[str]) -> np.ndarray:
    result = np.eye(2, dtype=np.complex128)
    for name in gates:
        if name not in GATES:
            raise ValueError(f"unsupported gate: {name}")
        result = GATES[name] @ result
    return result


def equivalent_up_to_global_phase(
    left: tuple[str, ...] | list[str], right: tuple[str, ...] | list[str],
    atol: float = 1e-10,
) -> bool:
    a, b = unitary(left), unitary(right)
    overlap = np.vdot(b.reshape(-1), a.reshape(-1))
    if abs(overlap) < atol:
        return False
    phase = overlap / abs(overlap)
    return bool(np.allclose(a, phase * b, atol=atol, rtol=0.0))


def canonical_circuit(gates: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Return the shortest lexicographic representative among circuits of <=2 gates."""
    candidates = [()]
    candidates.extend((g,) for g in GATE_NAMES)
    candidates.extend((a, b) for a in GATE_NAMES for b in GATE_NAMES)
    matches = [candidate for candidate in candidates
               if equivalent_up_to_global_phase(gates, candidate)]
    if not matches:
        raise ValueError("no canonical representative in the smoke gate budget")
    return min(matches, key=lambda item: (len(item), item))

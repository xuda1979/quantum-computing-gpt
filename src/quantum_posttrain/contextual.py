"""Small exact witnesses for contract changes under coherent control."""

from __future__ import annotations

import math

import numpy as np


def process_fidelity(left: np.ndarray, right: np.ndarray) -> float:
    """Phase-insensitive unitary process fidelity for equal square matrices."""
    if left.ndim != 2 or left.shape[0] != left.shape[1] or right.shape != left.shape:
        raise ValueError("unitaries must be square matrices with equal shape")
    dimension = left.shape[0]
    return float(abs(np.trace(left.conj().T @ right) / dimension) ** 2)


def controlled_unitary(unitary: np.ndarray) -> np.ndarray:
    """Build the control-first block matrix diag(I, U)."""
    if unitary.ndim != 2 or unitary.shape[0] != unitary.shape[1]:
        raise ValueError("unitary must be square")
    dimension = unitary.shape[0]
    identity = np.eye(dimension, dtype=np.complex128)
    zero = np.zeros_like(identity)
    return np.block([[identity, zero], [zero, unitary]])


def coherent_control_phase_witness(phase: float) -> dict[str, float]:
    """Compare I and exp(i phase) I, then expose their controlled difference."""
    if not math.isfinite(phase):
        raise ValueError("phase must be finite")
    identity = np.eye(2, dtype=np.complex128)
    phase_shifted_identity = np.exp(1j * phase) * identity
    controlled_reference = controlled_unitary(identity)
    controlled_candidate = controlled_unitary(phase_shifted_identity)

    plus = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    zero = np.array([1.0, 0.0], dtype=np.complex128)
    initial = np.kron(plus, zero)
    reference_output = controlled_reference @ initial
    candidate_output = controlled_candidate @ initial
    overlap = abs(np.vdot(reference_output, candidate_output)) ** 2

    pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    x_control = np.kron(pauli_x, identity)
    reference_x = np.vdot(reference_output, x_control @ reference_output).real
    candidate_x = np.vdot(candidate_output, x_control @ candidate_output).real
    return {
        "standalone_process_fidelity": process_fidelity(identity, phase_shifted_identity),
        "controlled_process_fidelity": process_fidelity(
            controlled_reference, controlled_candidate
        ),
        "witness_state_overlap": float(overlap),
        "reference_control_x": float(reference_x),
        "candidate_control_x": float(candidate_x),
    }

"""Deterministic randomized checks for the proposed research mechanisms.

These checks validate mathematical statements and harness behavior; they do
not establish that training improves a language model.
"""

from __future__ import annotations

import math
import random

from .allocation import (
    acquisition_priority,
    advantage_bounds,
    bound_decomposition_weights,
    certified_coefficient_mass,
    clipped_group_surrogate,
    conservative_coefficients,
    surrogate_difference_bound,
)
from .contextual import coherent_control_phase_witness


def _actual_advantages(rewards: list[float]) -> list[float]:
    mean = sum(rewards) / len(rewards)
    return [reward - mean for reward in rewards]


def _check_advantage_bounds(rng: random.Random, trials: int) -> None:
    for _ in range(trials):
        group_size = rng.randint(2, 8)
        intervals = []
        for _ in range(group_size):
            first, second = sorted((rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)))
            intervals.append((first, second))
        bounds = advantage_bounds(intervals)
        conservative = conservative_coefficients(intervals)

        # Check actual rewards, plus every box corner for smaller groups.
        rewards = [rng.uniform(low, high) for low, high in intervals]
        actual = _actual_advantages(rewards)
        for i, (advantage, (low, high), coefficient) in enumerate(
            zip(actual, bounds, conservative)
        ):
            if advantage < low - 1e-12 or advantage > high + 1e-12:
                raise AssertionError(f"advantage {i} escaped its interval")
            if coefficient and (coefficient * advantage < -1e-12
                                or abs(coefficient) > abs(advantage) + 1e-12):
                raise AssertionError(f"conservative coefficient {i} is not certified")

        if group_size <= 5:
            for mask in range(1 << group_size):
                corner = [intervals[i][(mask >> i) & 1] for i in range(group_size)]
                for i, advantage in enumerate(_actual_advantages(corner)):
                    low, high = bounds[i]
                    if advantage < low - 1e-12 or advantage > high + 1e-12:
                        raise AssertionError("a reward-box corner escaped its bound")

    point_intervals = [(rng.uniform(-1.0, 1.0),) * 2 for _ in range(4)]
    point_rewards = [low for low, _ in point_intervals]
    if any(abs(a - b) > 1e-12 for a, b in zip(
        conservative_coefficients(point_intervals), _actual_advantages(point_rewards)
    )):
        raise AssertionError("point intervals failed to recover centered advantages")


def _check_surrogate_bound(rng: random.Random, trials: int) -> None:
    for _ in range(trials):
        group_size = rng.randint(2, 8)
        intervals = []
        for _ in range(group_size):
            low = rng.uniform(-1.0, 0.5)
            high = rng.uniform(low, 1.0)
            intervals.append((low, high))
        rewards = [rng.uniform(low, high) for low, high in intervals]
        true_advantages = _actual_advantages(rewards)
        conservative = conservative_coefficients(intervals)

        ratios, masks = [], []
        completion_cap = 24.0
        for _ in range(group_size):
            token_count = rng.randint(1, 16)
            ratios.append([
                0.0 if rng.random() < 0.03 else
                rng.uniform(0.0, 1.5) if rng.random() < 0.8 else
                rng.uniform(1.5, 25.0)
                for _ in range(token_count)
            ])
            masks.append([rng.random() < 0.85 for _ in range(token_count)])

        epsilon = 0.2
        observed = abs(
            clipped_group_surrogate(ratios, true_advantages, masks, epsilon, completion_cap)
            - clipped_group_surrogate(ratios, conservative, masks, epsilon, completion_cap)
        )
        bound = surrogate_difference_bound(
            intervals, ratios, masks, epsilon, completion_cap
        )
        if observed > bound + 1e-10:
            raise AssertionError(f"surrogate error {observed} exceeded bound {bound}")

        token_weights = []
        for ratio_row, mask_row in zip(ratios, masks):
            token_weights.append(sum(
                max(ratio, min(max(ratio, 1.0 - epsilon), 1.0 + epsilon))
                for ratio, mask in zip(ratio_row, mask_row) if mask
            ) / completion_cap)
        widths = [high - low for low, high in intervals]
        decomposed = sum(k * width for k, width in zip(
            bound_decomposition_weights(token_weights), widths
        ))
        if abs(bound - decomposed) > 1e-10 * max(1.0, bound):
            raise AssertionError("surrogate bound decomposition changed its value")


def _check_equal_length_degeneration() -> None:
    group_size = 6
    intervals = [(-0.8 + i * 0.02, 0.9 - i * 0.01) for i in range(group_size)]
    common_ratios = [0.75, 1.0, 1.1, 1.5, 3.0]
    common_mask = [True, True, True, True, True]
    weights = [common_ratios[:] for _ in range(group_size)]
    masks = [common_mask[:] for _ in range(group_size)]
    reductions = [0.1 + i * 0.17 for i in range(group_size)]
    costs = [0.5 + i * 0.31 for i in range(group_size)]
    priorities = [acquisition_priority(
        intervals, [1.0] * group_size, i, reductions[i], 0.0, costs[i], lambda_mass=0.0
    ) for i in range(group_size)]
    expected = [reductions[i] / costs[i] for i in range(group_size)]
    scale = priorities[0] / expected[0]
    if any(abs(priority - scale * value) > 1e-12
           for priority, value in zip(priorities, expected)):
        raise AssertionError("lambda=0 allocator did not reduce to width-per-cost")

    # Equal token rows imply equal s_i and therefore equal decomposition weights.
    eps = 0.2
    token_weights = [sum(max(r, min(max(r, 1-eps), 1+eps)) for r in row) / 24
                     for row in weights]
    k = bound_decomposition_weights(token_weights)
    if max(k) - min(k) > 1e-12:
        raise AssertionError("equal-length/equal-policy group has unequal k weights")


def _check_nested_interval_mass() -> None:
    coarse = [(-1.0, 1.0), (-0.8, 0.9), (-0.7, 0.6), (-0.3, 0.8)]
    narrow = [(-0.4, 0.5), (-0.2, 0.3), (-0.5, 0.2), (0.1, 0.4)]
    weights = [0.7, 1.1, 0.3, 2.0]
    if certified_coefficient_mass(narrow, weights) + 1e-12 < certified_coefficient_mass(
        coarse, weights
    ):
        raise AssertionError("nested interval refinement reduced certified mass")


def _check_contextual_phase_grid(points: int) -> None:
    for index in range(points):
        phase = 2.0 * math.pi * index / max(1, points - 1)
        values = coherent_control_phase_witness(phase)
        expected = math.cos(phase / 2.0) ** 2
        if abs(values["standalone_process_fidelity"] - 1.0) > 1e-12:
            raise AssertionError("global phase changed standalone process fidelity")
        for key in ("controlled_process_fidelity", "witness_state_overlap"):
            if abs(values[key] - expected) > 1e-12:
                raise AssertionError(f"{key} does not detect the coherent phase")
        if abs(values["candidate_control_x"] - math.cos(phase)) > 1e-12:
            raise AssertionError("control-X witness does not match its analytic value")


def run_research_validation(trials: int = 10_000, seed: int = 20261023) -> dict:
    """Run deterministic randomized checks of the proposed scalar mechanisms."""
    if trials < 1:
        raise ValueError("trials must be positive")
    rng = random.Random(seed)
    _check_advantage_bounds(rng, trials)
    _check_surrogate_bound(rng, trials)
    _check_equal_length_degeneration()
    _check_nested_interval_mass()
    phase_points = 257
    _check_contextual_phase_grid(phase_points)
    return {
        "status": "PASS",
        "seed": seed,
        "advantage_bound_trials": trials,
        "surrogate_bound_trials": trials,
        "phase_grid_points": phase_points,
        "additional_checks": [
            "all reward-box corners for group sizes 2 through 5",
            "point-interval recovery",
            "surrogate-bound coefficient decomposition",
            "lambda=0 equal-length degeneration",
            "nested-interval certified-mass monotonicity",
        ],
        "scope": "mathematical and small-matrix properties only; no learned-policy efficacy claim",
    }

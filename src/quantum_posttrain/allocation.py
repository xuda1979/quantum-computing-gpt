"""Executable checks for interval-based GRPO reward bounds and acquisition scores.

These functions cover the scalar surrogate analysis in the research specification.
They do not bound parameter-gradient error or prove policy improvement.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

Interval = tuple[float, float]


def _intervals(values: Sequence[Interval]) -> list[Interval]:
    if len(values) == 0:
        raise ValueError("at least one reward interval is required")
    result = []
    for index, pair in enumerate(values):
        if len(pair) != 2:
            raise ValueError(f"interval {index} must have two endpoints")
        low, high = float(pair[0]), float(pair[1])
        if not math.isfinite(low) or not math.isfinite(high) or low > high:
            raise ValueError(f"interval {index} must be finite and ordered")
        result.append((low, high))
    return result


def advantage_bounds(intervals: Sequence[Interval]) -> list[Interval]:
    """Exact mean-centered advantage intervals for a rectangular reward set."""
    bounds = _intervals(intervals)
    group_size = len(bounds)
    lows = [pair[0] for pair in bounds]
    highs = [pair[1] for pair in bounds]
    sum_lows, sum_highs = sum(lows), sum(highs)
    output = []
    for low, high in bounds:
        lower = (1.0 - 1.0 / group_size) * low - (sum_highs - high) / group_size
        upper = (1.0 - 1.0 / group_size) * high - (sum_lows - low) / group_size
        output.append((lower, upper))
    return output


def conservative_coefficients(intervals: Sequence[Interval]) -> list[float]:
    """Return the nearest-to-zero coefficient only when its sign is certified."""
    result = []
    for low, high in advantage_bounds(intervals):
        result.append(low if low > 0.0 else high if high < 0.0 else 0.0)
    return result


def _token_rows(
    ratios: Sequence[Sequence[float]], masks: Sequence[Sequence[bool]], group_size: int,
    epsilon: float, completion_cap: float,
) -> tuple[list[list[float]], list[list[bool]]]:
    if not 0.0 <= epsilon < 1.0:
        raise ValueError("epsilon must be in [0, 1)")
    if not math.isfinite(completion_cap) or completion_cap <= 0.0:
        raise ValueError("completion_cap must be positive and finite")
    if len(ratios) != group_size or len(masks) != group_size:
        raise ValueError("ratio and mask rows must match the reward group size")
    clean_ratios, clean_masks = [], []
    for i, (ratio_row, mask_row) in enumerate(zip(ratios, masks)):
        if len(ratio_row) != len(mask_row):
            raise ValueError(f"ratio and mask token counts differ in row {i}")
        values = [float(ratio) for ratio in ratio_row]
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError(f"ratios in row {i} must be finite and nonnegative")
        clean_ratios.append(values)
        clean_masks.append([bool(mask) for mask in mask_row])
    return clean_ratios, clean_masks


def clipped_group_surrogate(
    ratios: Sequence[Sequence[float]], advantages: Sequence[float],
    masks: Sequence[Sequence[bool]], epsilon: float = 0.2,
    completion_cap: float = 1.0,
) -> float:
    """Masked clipped surrogate averaged over group size and completion cap."""
    if len(advantages) == 0:
        raise ValueError("at least one advantage is required")
    clean_ratios, clean_masks = _token_rows(
        ratios, masks, len(advantages), epsilon, completion_cap
    )
    total = 0.0
    for ratio_row, mask_row, advantage in zip(clean_ratios, clean_masks, advantages):
        a = float(advantage)
        if not math.isfinite(a):
            raise ValueError("advantages must be finite")
        for ratio, mask in zip(ratio_row, mask_row):
            if mask:
                clipped = min(max(ratio, 1.0 - epsilon), 1.0 + epsilon)
                total += min(ratio * a, clipped * a)
    return total / (len(advantages) * completion_cap)


def surrogate_difference_bound(
    intervals: Sequence[Interval], ratios: Sequence[Sequence[float]],
    masks: Sequence[Sequence[bool]], epsilon: float = 0.2,
    completion_cap: float = 1.0,
) -> float:
    """Compute B = (1/G) sum_i s_i (U_i-L_i) from Proposition 2."""
    reward_intervals = _intervals(intervals)
    bounds = advantage_bounds(reward_intervals)
    clean_ratios, clean_masks = _token_rows(
        ratios, masks, len(reward_intervals), epsilon, completion_cap
    )
    total = 0.0
    for (low, high), ratio_row, mask_row in zip(bounds, clean_ratios, clean_masks):
        s_i = sum(
            max(ratio, min(max(ratio, 1.0 - epsilon), 1.0 + epsilon))
            for ratio, mask in zip(ratio_row, mask_row) if mask
        ) / completion_cap
        total += s_i * (high - low)
    return total / len(reward_intervals)


def bound_decomposition_weights(token_weights: Sequence[float]) -> list[float]:
    """Return k_j such that B equals sum_j k_j times reward interval width."""
    values = [float(value) for value in token_weights]
    if not values or any(not math.isfinite(v) or v < 0.0 for v in values):
        raise ValueError("token weights must be nonempty, finite, and nonnegative")
    group_size = len(values)
    total = sum(values)
    return [((group_size - 1) * value + total - value) / group_size**2
            for value in values]


def certified_coefficient_mass(
    intervals: Sequence[Interval], token_weights: Sequence[float],
) -> float:
    bounds = _intervals(intervals)
    weights = [float(value) for value in token_weights]
    if len(weights) != len(bounds):
        raise ValueError("one token weight is required per candidate")
    if any(not math.isfinite(w) or w < 0.0 for w in weights):
        raise ValueError("token weights must be finite and nonnegative")
    coeffs = conservative_coefficients(bounds)
    return sum(weight * abs(coeff) for weight, coeff in zip(weights, coeffs)) / len(bounds)


def acquisition_priority(
    intervals: Sequence[Interval], token_weights: Sequence[float], candidate: int,
    expected_width_reduction: float, expected_mass_gain: float, expected_cost: float,
    lambda_mass: float = 1.0,
) -> float:
    """Evaluate the proposed priority; expectations must come from development traces."""
    bounds = _intervals(intervals)
    weights = [float(value) for value in token_weights]
    if len(weights) != len(bounds) or not 0 <= candidate < len(bounds):
        raise ValueError("candidate index or token weights do not match the group")
    if any(not math.isfinite(w) or w < 0.0 for w in weights):
        raise ValueError("token weights must be finite and nonnegative")
    terms = (expected_width_reduction, expected_mass_gain, expected_cost, lambda_mass)
    if any(not math.isfinite(float(value)) for value in terms):
        raise ValueError("acquisition inputs must be finite")
    if expected_width_reduction < 0.0 or expected_mass_gain < 0.0:
        raise ValueError("expected gains must be nonnegative")
    if expected_cost <= 0.0 or lambda_mass < 0.0:
        raise ValueError("cost must be positive and lambda_mass nonnegative")
    k = bound_decomposition_weights(weights)[candidate]
    return (k * expected_width_reduction + lambda_mass * expected_mass_gain) / expected_cost

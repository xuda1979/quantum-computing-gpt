"""Deterministic root-disjoint data splitting."""

from __future__ import annotations

import hashlib
from dataclasses import replace

from .contracts import Task


def assign_root_splits(tasks: list[Task], seed: str = "qposttrain-split-v1") -> list[Task]:
    roots = sorted({task.root_id for task in tasks})
    assignment = {}
    for root in roots:
        value = int.from_bytes(hashlib.sha256(f"{seed}:{root}".encode()).digest()[:8], "big")
        fraction = value / 2**64
        assignment[root] = "train" if fraction < 0.8 else "dev" if fraction < 0.9 else "test"
    return [replace(task, split=assignment[task.root_id]) for task in tasks]


def assert_root_disjoint(*partitions: list[Task]) -> None:
    seen: dict[str, int] = {}
    for index, tasks in enumerate(partitions):
        for task in tasks:
            previous = seen.setdefault(task.root_id, index)
            if previous != index:
                raise ValueError(f"root {task.root_id} appears in multiple splits")

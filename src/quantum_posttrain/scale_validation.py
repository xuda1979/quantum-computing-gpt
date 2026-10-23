"""Synthetic long-circuit data load for exercising the harness at scale.

This is infrastructure validation only. It is deliberately separate from the
16-task learning smoke test and makes no claim about model generalization.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

from .contracts import Task
from .generator import write_jsonl
from .quantum import equivalent_up_to_global_phase
from .splits import assign_root_splits, assert_root_disjoint

_SELF_INVERSE_GATES = ("H", "X", "Z")
_GATES = ("I", "X", "Z", "H")


def generate_scaled_tasks(
    roots: int = 10_000, variants_per_root: int = 10, seed: int = 20261023,
) -> list[Task]:
    """Generate long circuits by inserting known identity pairs into each root."""
    if roots < 3 or variants_per_root < 1:
        raise ValueError("at least 3 roots and 1 variant per root are required")
    rng = random.Random(seed)
    tasks: list[Task] = []
    for root_index in range(roots):
        root_id = f"scale-root-{root_index:07d}"
        base_length = 8 + root_index % 5
        target = tuple(rng.choice(_GATES) for _ in range(base_length))
        family = f"random-circuit-length-{base_length}"
        for variant_index in range(variants_per_root):
            source = list(target)
            for pair_index in range(variant_index + 1):
                gate = _SELF_INVERSE_GATES[(root_index + variant_index + pair_index) % 3]
                insertion = rng.randrange(len(source) + 1)
                source[insertion:insertion] = [gate, gate]
            tasks.append(Task(
                task_id=f"{root_id}-variant-{variant_index:03d}",
                root_id=root_id,
                family_id=f"{root_id}:{family}",
                split="train",
                source_gates=tuple(source),
                target_gates=target,
            ))
    return tasks


def run_scale_validation(
    output_dir: Path, roots: int = 10_000, variants_per_root: int = 10,
    seed: int = 20261023,
) -> dict:
    """Generate, split, serialize, reload, and exactly verify a synthetic corpus."""
    started = time.perf_counter()
    tasks = generate_scaled_tasks(roots, variants_per_root, seed)
    generated_seconds = time.perf_counter() - started

    split_started = time.perf_counter()
    tasks = assign_root_splits(tasks, seed=f"scale-validation-v1:{seed}")
    partitions = {name: [task for task in tasks if task.split == name]
                  for name in ("train", "dev", "test")}
    if any(not partition for partition in partitions.values()):
        raise AssertionError("scale fixture did not populate every split")
    assert_root_disjoint(*(partitions[name] for name in ("train", "dev", "test")))
    split_seconds = time.perf_counter() - split_started

    verify_started = time.perf_counter()
    for task in tasks:
        if not equivalent_up_to_global_phase(task.source_gates, task.target_gates):
            raise AssertionError(f"invalid synthetic label: {task.task_id}")
    verification_seconds = time.perf_counter() - verify_started

    output_dir.mkdir(parents=True, exist_ok=True)
    write_started = time.perf_counter()
    paths = {}
    for name, partition in partitions.items():
        path = output_dir / f"{name}.jsonl"
        write_jsonl(path, partition)
        paths[name] = path
    write_seconds = time.perf_counter() - write_started

    reload_started = time.perf_counter()
    reloaded_counts = Counter()
    hashes = {}
    bytes_by_split = {}
    for name, path in paths.items():
        digest = hashlib.sha256()
        count = 0
        with path.open("rb") as raw:
            for line in raw:
                digest.update(line)
                record = json.loads(line)
                task = Task.from_dict(record)
                if task.split != name:
                    raise AssertionError(f"split mismatch in {path.name}")
                count += 1
        reloaded_counts[name] = count
        hashes[name] = digest.hexdigest()
        bytes_by_split[name] = path.stat().st_size
    reload_seconds = time.perf_counter() - reload_started

    expected = Counter({name: len(partition) for name, partition in partitions.items()})
    if reloaded_counts != expected or sum(reloaded_counts.values()) != len(tasks):
        raise AssertionError("serialized record counts changed during reload")
    root_counts = {name: len({task.root_id for task in partition})
                   for name, partition in partitions.items()}
    return {
        "status": "PASS",
        "scope": "synthetic one-qubit identity insertions; harness scale only",
        "seed": seed,
        "roots": roots,
        "variants_per_root": variants_per_root,
        "records": len(tasks),
        "max_circuit_gates": max(len(task.source_gates) for task in tasks),
        "records_by_split": dict(reloaded_counts),
        "roots_by_split": root_counts,
        "bytes_by_split": bytes_by_split,
        "sha256_by_split": hashes,
        "seconds": {
            "generate": generated_seconds,
            "split_and_disjoint_check": split_seconds,
            "exact_verify": verification_seconds,
            "serialize": write_seconds,
            "reload_and_contract_check": reload_seconds,
            "total": time.perf_counter() - started,
        },
    }

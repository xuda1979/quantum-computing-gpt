"""Deterministic harness-owned task generation for the exact smoke domain."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

from .contracts import Task
from .quantum import GATE_NAMES, canonical_circuit, equivalent_up_to_global_phase
from .splits import assign_root_splits


def generate_tasks(split: str | None = "train") -> list[Task]:
    if split not in {"train", "dev", "test", None}:
        raise ValueError("split must be train, dev, or test")
    tasks: list[Task] = []
    for index, source in enumerate(product(GATE_NAMES, repeat=2)):
        target = canonical_circuit(source)
        if not equivalent_up_to_global_phase(source, target):
            raise AssertionError("generator oracle produced a bad target")
        family = f"pair-{source[0]}-{source[1]}"
        tasks.append(Task(
            task_id=f"smoke-{index:02d}",
            root_id=f"root-{index:02d}",
            family_id=family,
            # Build a valid provisional record; root-level splitting assigns
            # the final partition after every task has a stable root ID.
            split=split or "train",
            source_gates=tuple(source),
            target_gates=target,
        ))
    if split is None:
        return assign_root_splits(tasks)
    return tasks


def write_jsonl(path: Path, tasks: list[Task]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task.to_dict(), sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/smoke/tasks.jsonl"))
    parser.add_argument("--split", choices=("train", "dev", "test", "all"), default="train")
    args = parser.parse_args()
    tasks = generate_tasks(None if args.split == "all" else args.split)
    write_jsonl(args.output, tasks)
    print(json.dumps({"written": len(tasks), "path": str(args.output),
                      "split_counts": {name: sum(t.split == name for t in tasks)
                                       for name in ("train", "dev", "test")}}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create a root-disjoint exact smoke dataset and split manifest."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from quantum_posttrain.generator import generate_tasks, write_jsonl
from quantum_posttrain.splits import assert_root_disjoint


def main() -> None:
    root = Path("data/experiment")
    tasks = generate_tasks(None)
    partitions = {name: [task for task in tasks if task.split == name]
                  for name in ("train", "dev", "test")}
    if any(not values for values in partitions.values()):
        raise SystemExit("too few independent roots to populate all splits")
    assert_root_disjoint(*partitions.values())
    for name, values in partitions.items():
        write_jsonl(root / f"{name}.jsonl", values)
    manifest = {
        "schema_version": 1,
        "split_method": "sha256_root_assignment_v1",
        "seed": "qposttrain-split-v1",
        "counts": dict(Counter(task.split for task in tasks)),
        "root_ids": {name: sorted({task.root_id for task in values})
                     for name, values in partitions.items()},
    }
    (root / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(root), "counts": manifest["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()

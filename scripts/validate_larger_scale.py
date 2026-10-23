#!/usr/bin/env python3
"""Generate and verify a deterministic synthetic corpus at harness scale."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantum_posttrain.scale_validation import run_scale_validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", type=int, default=10_000)
    parser.add_argument("--variants-per-root", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--output", type=Path, default=Path("runs/scale-validation"))
    args = parser.parse_args()
    result = run_scale_validation(
        args.output, roots=args.roots,
        variants_per_root=args.variants_per_root, seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

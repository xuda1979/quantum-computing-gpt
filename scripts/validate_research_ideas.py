#!/usr/bin/env python3
"""Run deterministic randomized checks of the research-spec mechanisms."""

from __future__ import annotations

import argparse
import json

from quantum_posttrain.research_validation import run_research_validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    print(json.dumps(run_research_validation(args.trials, args.seed), indent=2))


if __name__ == "__main__":
    main()

"""Tiny ordered-2DM smoke search.

This is not a decision procedure. It is a bounded experiment driver for the
first conjecture loop: generate very small 2DM-like safety instances whose
controllable decision actions are scheduled one-at-a-time, run the existing
bounded memoryless checker, and summarize whether any bounded counterexample
appears.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.ats_brute_solver import find_memoryless_safety_strategy
from src.experiments.ats_generator import generate_tiny_game


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a tiny ordered-2DM bounded safety search")
    parser.add_argument("--instances", type=int, default=50)
    parser.add_argument("--max-processes", type=int, default=3)
    parser.add_argument("--max-local-states", type=int, default=3)
    parser.add_argument("--objective", default="safety", choices=["safety"])
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_processes < 2:
        raise SystemExit("--max-processes must be at least 2")
    if args.max_local_states < 2:
        raise SystemExit("--max-local-states must be at least 2")

    summaries = []
    counterexamples = []
    for seed in range(args.instances):
        process_count = 2 + (seed % max(1, args.max_processes - 1))
        states_per_process = 2 + (seed % max(1, args.max_local_states - 1))
        game = generate_tiny_game(
            kind="2DM",
            process_count=min(process_count, 4),
            states_per_process=min(states_per_process, 3),
            seed=seed,
        )
        result = find_memoryless_safety_strategy(game, depth=args.depth)
        item = {
            "seed": seed,
            "game": game.name,
            "processes": len(game.processes),
            "states_per_process": len(game.processes[0].local_states),
            "causal_order_constraint": "controllable decision actions are singleton local actions in interleaving semantics",
            "winning": result.winning,
            "checked_strategies": result.checked_strategies,
            "counterexample": result.counterexample,
        }
        summaries.append(item)
        if result.counterexample:
            counterexamples.append(item)

    payload = {
        "experiment": "tiny_ordered_2dm_search",
        "instances": args.instances,
        "depth": args.depth,
        "objective": args.objective,
        "counterexample_count": len(counterexamples),
        "counterexamples": counterexamples[:20],
        "summaries": summaries,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(f"counterexample_count={len(counterexamples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

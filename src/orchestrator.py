"""Conservative manual pipeline skeleton for research harness tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import db
from .curate import PendingAnalysis, curate_pending
from .experiments.ats_brute_solver import find_memoryless_safety_strategy
from .experiments.ats_generator import generate_tiny_game
from .extract import LLMClient, extract_from_text
from .schemas import ExperimentRun


@dataclass(frozen=True)
class PipelineResult:
    """Compact result from a manual pipeline invocation."""

    mode: str
    cluster_id: int
    summary: str


def run_extraction_on_text(text: str, db_path: str | Path | None = None) -> list[int]:
    """Run extraction on user-supplied text and store only pending entries."""

    return extract_from_text(text, db_path=db_path, client=LLMClient())


def run_curator(db_path: str | Path | None = None) -> list[PendingAnalysis]:
    """Run curation analysis without approving entries."""

    return curate_pending(db_path=db_path)


def generate_examples_for_one_active_conjecture(
    cluster_id: int,
    db_path: str | Path | None = None,
) -> dict:
    """Generate one small ATS-family example for an active conjecture cluster."""

    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        conjectures = [
            conjecture
            for conjecture in db.list_conjectures(connection, cluster_id=cluster_id)
            if conjecture.status == "active"
        ]
    seed = cluster_id if conjectures else 0
    game = generate_tiny_game(kind="ATS", process_count=2, states_per_process=2, seed=seed)
    return game.to_dict()


def brute_check_generated(game_data: dict, depth: int = 5) -> dict:
    """Run the bounded ATS-family brute checker on generated game JSON."""

    from .experiments.ats_models import SafetyGame

    result = find_memoryless_safety_strategy(SafetyGame.from_dict(game_data), depth=depth)
    return {
        "winning": result.winning,
        "checked_strategies": result.checked_strategies,
        "depth": result.depth,
        "strategy": result.strategy,
        "counterexample": result.counterexample,
    }


def store_experiment_result(
    cluster_id: int,
    experiment_type: str,
    input_json: dict,
    output_json: dict,
    result_summary: str,
    db_path: str | Path | None = None,
) -> int:
    """Store a bounded experiment run."""

    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        return db.insert_experiment_run(
            connection,
            ExperimentRun(
                cluster_id=cluster_id,
                experiment_type=experiment_type,
                input_json=input_json,
                output_json=output_json,
                result_summary=result_summary,
            ),
        )


def run_pipeline(cluster_id: int, mode: str, db_path: str | Path | None = None) -> PipelineResult:
    """Run a single bounded manual pipeline step."""

    if mode == "literature":
        analyses = curate_pending(db_path=db_path)
        warning_count = sum(len(analysis.warnings) for analysis in analyses)
        return PipelineResult(
            mode=mode,
            cluster_id=cluster_id,
            summary=f"curated {len(analyses)} pending entries with {warning_count} warnings",
        )
    if mode == "experiments":
        game = generate_examples_for_one_active_conjecture(cluster_id=cluster_id, db_path=db_path)
        output = brute_check_generated(game, depth=5)
        run_id = store_experiment_result(
            cluster_id=cluster_id,
            experiment_type="ats_bounded_safety",
            input_json=game,
            output_json=output,
            result_summary=f"winning={output['winning']} checked={output['checked_strategies']}",
            db_path=db_path,
        )
        return PipelineResult(
            mode=mode,
            cluster_id=cluster_id,
            summary=f"stored experiment run {run_id}",
        )
    raise ValueError("mode must be literature or experiments")

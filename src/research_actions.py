"""Validated research references and trusted bounded experiment helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field

from . import db
from .experiments.ats_brute_solver import find_memoryless_safety_strategy
from .experiments.ats_generator import generate_tiny_game
from .git_utils import get_current_commit_hash
from .schemas import CodeArtifact, ExperimentRun
from .schemas import StrictBase


ReferenceKind = Literal[
    "research_task",
    "research_unit",
    "concept",
    "paper",
    "theorem",
    "reduction",
    "conjecture",
    "open_problem",
    "derived_result",
    "proof_attempt",
    "evidence",
    "experiment_run",
    "code_artifact",
]


class ResearchReference(StrictBase):
    """Reference to an ID supplied in the bounded research context."""

    kind: ReferenceKind
    object_id: int = Field(gt=0)


class BoundedExperimentParameters(StrictBase):
    """Conservative limits that the LLM cannot exceed."""

    artifact_id: int = Field(gt=0)
    kind: Literal["ATS", "CDM", "2DM"] = "ATS"
    process_count: int = Field(default=2, ge=2, le=3)
    states_per_process: int = Field(default=2, ge=2, le=2)
    depth: int = Field(default=5, ge=0, le=5)
    seed: int = Field(default=0, ge=0, le=1_000_000)


@dataclass(frozen=True)
class TrustedExperimentResult:
    """Result returned by the trusted bounded experiment handler."""

    run_id: int
    summary: str
    output: dict[str, object]


def get_trusted_artifact(
    artifact_id: int,
    task_id: int,
    db_path: str | Path | None = None,
) -> CodeArtifact:
    """Validate trust metadata without executing the artifact itself."""

    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        artifact = db.get_code_artifact(connection, artifact_id)
    if artifact is None:
        raise ValueError(f"code artifact {artifact_id} does not exist")
    if artifact.status != "tested":
        raise ValueError(f"code artifact {artifact_id} is not trusted for bounded execution")
    if artifact.artifact_type not in {"checker", "solver"}:
        raise ValueError(f"code artifact {artifact_id} is not a trusted checker or solver")
    if artifact.task_id is not None and artifact.task_id != task_id:
        raise ValueError(f"code artifact {artifact_id} belongs to another research task")
    return artifact


def run_trusted_bounded_experiment(
    parameters: BoundedExperimentParameters,
    task_id: int,
    db_path: str | Path | None = None,
    connection: db.PostgresConnection | None = None,
) -> TrustedExperimentResult:
    """Run the fixed in-process tiny ATS-family safety handler.

    The artifact record supplies provenance only. Its path, entrypoint, notes,
    and any LLM-provided command text are never executed by this handler.
    """

    get_trusted_artifact(parameters.artifact_id, task_id, db_path)
    game = generate_tiny_game(
        kind=parameters.kind,
        process_count=parameters.process_count,
        states_per_process=parameters.states_per_process,
        seed=parameters.seed,
    )
    result = find_memoryless_safety_strategy(game, depth=parameters.depth)
    output: dict[str, object] = {
        "winning": result.winning,
        "checked_strategies": result.checked_strategies,
        "depth": result.depth,
        "strategy": result.strategy,
        "counterexample": result.counterexample,
    }
    summary = f"winning={result.winning} checked={result.checked_strategies} depth={result.depth}"
    record = ExperimentRun(
        artifact_id=parameters.artifact_id,
        task_id=task_id,
        experiment_type="ats_bounded_safety",
        input_json=game.to_dict(),
        output_json=output,
        result_summary=summary,
        command_run="trusted_in_process:ats_bounded_safety",
        git_commit_hash=get_current_commit_hash(),
        notes="Controller-runnable bounded handler; no artifact command was executed.",
    )
    if connection is None:
        with db.get_connection(db_path) as owned_connection:
            db.create_tables(owned_connection)
            run_id = db.insert_experiment_run(owned_connection, record)
    else:
        run_id = db.insert_experiment_run(connection, record)
    return TrustedExperimentResult(run_id=run_id, summary=summary, output=output)

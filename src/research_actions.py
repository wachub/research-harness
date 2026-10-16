"""Validated research references and trusted bounded experiment helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, ValidationError

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


class ATSExperimentInput(StrictBase):
    """Inputs accepted by the existing tiny safety-game handler."""

    kind: Literal["ATS", "CDM", "2DM"] = "ATS"
    process_count: int = Field(default=2, ge=2, le=3)
    states_per_process: int = Field(default=2, ge=2, le=2)
    depth: int = Field(default=5, ge=0, le=5)
    seed: int = Field(default=0, ge=0, le=1_000_000)


class BoundedExperimentParameters(ATSExperimentInput):
    """Legacy ATS-family proposal shape, retained for existing runs and callers."""

    artifact_id: int = Field(gt=0)


class RegisteredExperimentParameters(StrictBase):
    """Select a code-registered handler; parameters are validated by its own model."""

    artifact_id: int = Field(gt=0)
    handler: str = Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=80)
    parameters: dict[str, Any] = Field(default_factory=dict)


ExperimentParameters = BoundedExperimentParameters | RegisteredExperimentParameters


@dataclass(frozen=True)
class ExperimentOutput:
    input_json: dict[str, Any]
    output_json: dict[str, Any]
    summary: str


@dataclass(frozen=True)
class ExperimentHandler:
    input_model: type[BaseModel]
    run: Callable[[BaseModel], ExperimentOutput]
    artifact_types: frozenset[str]
    description: str


_HANDLERS: dict[str, ExperimentHandler] = {}


def register_experiment_handler(name: str, handler: ExperimentHandler) -> None:
    """Register reviewed Python code, never an import path or command from a model."""
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,79}", name) or name in _HANDLERS:
        raise ValueError("experiment handler name is invalid or already registered")
    if (not handler.artifact_types or not isinstance(handler.input_model, type)
            or not issubclass(handler.input_model, StrictBase) or not callable(handler.run)
            or not handler.description.strip()):
        raise ValueError("experiment handler needs a strict input model, runner and artifact types")
    _HANDLERS[name] = handler


def available_experiment_handlers() -> list[dict[str, Any]]:
    """Expose only code-registered names and parameter schemas to the controller."""
    return [{"name": name, "description": handler.description,
             "artifact_types": sorted(handler.artifact_types),
             "parameters_schema": handler.input_model.model_json_schema()}
            for name, handler in sorted(_HANDLERS.items())]


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
    allowed_types: frozenset[str] = frozenset({"checker", "solver"}),
) -> CodeArtifact:
    """Validate trust metadata without executing the artifact itself."""

    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        artifact = db.get_code_artifact(connection, artifact_id)
    if artifact is None:
        raise ValueError(f"code artifact {artifact_id} does not exist")
    if artifact.status != "tested":
        raise ValueError(f"code artifact {artifact_id} is not trusted for bounded execution")
    if artifact.artifact_type not in allowed_types:
        raise ValueError(f"code artifact {artifact_id} has an incompatible type")
    if artifact.task_id is not None and artifact.task_id != task_id:
        raise ValueError(f"code artifact {artifact_id} belongs to another research task")
    return artifact


def _ats_bounded_safety(parameters: BaseModel) -> ExperimentOutput:
    assert isinstance(parameters, ATSExperimentInput)
    game = generate_tiny_game(
        kind=parameters.kind,
        process_count=parameters.process_count,
        states_per_process=parameters.states_per_process,
        seed=parameters.seed,
    )
    result = find_memoryless_safety_strategy(game, depth=parameters.depth)
    output: dict[str, Any] = {
        "winning": result.winning,
        "checked_strategies": result.checked_strategies,
        "depth": result.depth,
        "strategy": result.strategy,
        "counterexample": result.counterexample,
    }
    return ExperimentOutput(game.to_dict(), output,
                            f"winning={result.winning} checked={result.checked_strategies} depth={result.depth}")


register_experiment_handler("ats_bounded_safety", ExperimentHandler(
    input_model=ATSExperimentInput, run=_ats_bounded_safety,
    artifact_types=frozenset({"checker", "solver"}),
    description="Fixed tiny ATS/CDM/2DM memoryless-safety check; no artifact code is executed.",
))


def run_trusted_bounded_experiment(
    parameters: ExperimentParameters,
    task_id: int,
    db_path: str | Path | None = None,
    connection: db.PostgresConnection | None = None,
) -> TrustedExperimentResult:
    """Run a code-registered, input-validated handler with bounded JSON output.

    The artifact record supplies provenance only. Its path, entrypoint, notes,
    and any LLM-provided command text are never executed by this handler.
    """

    if isinstance(parameters, BoundedExperimentParameters):
        name = "ats_bounded_safety"
        raw_inputs = parameters.model_dump(exclude={"artifact_id"})
    else:
        name = parameters.handler
        raw_inputs = parameters.parameters
    handler = _HANDLERS.get(name)
    if handler is None:
        raise ValueError(f"unknown registered experiment handler: {name}")
    get_trusted_artifact(parameters.artifact_id, task_id, db_path, handler.artifact_types)
    try:
        validated = handler.input_model.model_validate(raw_inputs)
        encoded_input = json.dumps(validated.model_dump(mode="json"), allow_nan=False)
    except (ValidationError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid parameters for registered experiment handler: {name}") from exc
    if len(encoded_input) > 20_000:
        raise ValueError("experiment parameters exceed the size limit")
    try:
        outcome = handler.run(validated)
        if (not isinstance(outcome, ExperimentOutput) or not isinstance(outcome.summary, str)
                or not 1 <= len(outcome.summary) <= 2_000
                or not isinstance(outcome.input_json, dict) or not isinstance(outcome.output_json, dict)
                or len(json.dumps(outcome.input_json, allow_nan=False)) > 100_000
                or len(json.dumps(outcome.output_json, allow_nan=False)) > 100_000):
            raise ValueError("invalid or oversized experiment output")
    except Exception as exc:
        raise ValueError(f"registered experiment handler {name} failed ({type(exc).__name__})") from exc
    record = ExperimentRun(
        artifact_id=parameters.artifact_id,
        task_id=task_id,
        experiment_type=name,
        input_json=outcome.input_json,
        output_json=outcome.output_json,
        result_summary=outcome.summary,
        command_run=f"trusted_in_process:{name}",
        git_commit_hash=get_current_commit_hash(),
        notes="Code-registered bounded handler; no artifact command was executed.",
    )
    if connection is None:
        with db.get_connection(db_path) as owned_connection:
            db.create_tables(owned_connection)
            run_id = db.insert_experiment_run(owned_connection, record)
    else:
        run_id = db.insert_experiment_run(connection, record)
    return TrustedExperimentResult(run_id=run_id, summary=outcome.summary, output=outcome.output_json)

"""Strict, finite action vocabulary for the bounded research controller."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from . import db
from .experiments.ats_brute_solver import find_memoryless_safety_strategy
from .experiments.ats_generator import generate_tiny_game
from .git_utils import get_current_commit_hash
from .schemas import CodeArtifact, ExperimentRun
from .schemas import StrictBase


ReferenceKind = Literal[
    "research_cluster",
    "concept",
    "paper",
    "theorem",
    "reduction",
    "conjecture",
    "open_problem",
    "evidence",
    "experiment_run",
    "code_artifact",
]


class ResearchReference(StrictBase):
    """Reference to an ID supplied in the controller's compact context."""

    kind: ReferenceKind
    object_id: int = Field(gt=0)


class EmptyParameters(StrictBase):
    """Explicit empty parameter object for read-only actions."""


class EvidenceReviewParameters(StrictBase):
    evidence_ids: list[int] = Field(default_factory=list, max_length=5)


class Subquestion(StrictBase):
    question: str = Field(min_length=1, max_length=500)
    rationale: str = Field(min_length=1, max_length=1000)
    uncertainty_note: str = Field(min_length=1, max_length=500)


class SubquestionParameters(StrictBase):
    subquestions: list[Subquestion] = Field(min_length=1, max_length=3)


class PendingConjectureParameters(StrictBase):
    statement: str = Field(min_length=1, max_length=1200)
    rationale: str = Field(min_length=1, max_length=1000)
    assumptions: list[str] = Field(default_factory=list, max_length=5)
    uncertainty_note: str = Field(min_length=1, max_length=500)


class PendingOpenProblemParameters(StrictBase):
    question: str = Field(min_length=1, max_length=1200)
    rationale: str = Field(min_length=1, max_length=1000)
    assumptions: list[str] = Field(default_factory=list, max_length=5)
    uncertainty_note: str = Field(min_length=1, max_length=500)


class BoundedExperimentParameters(StrictBase):
    """Conservative limits that the LLM cannot exceed."""

    artifact_id: int = Field(gt=0)
    kind: Literal["ATS", "CDM", "2DM"] = "ATS"
    process_count: int = Field(default=2, ge=2, le=3)
    states_per_process: int = Field(default=2, ge=2, le=2)
    depth: int = Field(default=5, ge=0, le=5)
    seed: int = Field(default=0, ge=0, le=1_000_000)


class ExperimentResultParameters(StrictBase):
    run_id: int = Field(gt=0)


class ReassessPlanParameters(StrictBase):
    summary: str = Field(min_length=1, max_length=1200)
    uncertainty_note: str = Field(min_length=1, max_length=500)


class ApprovalParameters(StrictBase):
    """A real existing operation that is deliberately exposed only to block it."""

    pending_id: int = Field(gt=0)


class BaseAction(StrictBase):
    reason: str = Field(min_length=1, max_length=1200)
    expected_effect: str = Field(min_length=1, max_length=1200)
    references: list[ResearchReference] = Field(default_factory=list, max_length=8)


class InspectStateAction(BaseAction):
    action_type: Literal["inspect_state"]
    parameters: EmptyParameters


class ReviewStoredEvidenceAction(BaseAction):
    action_type: Literal["review_stored_evidence"]
    parameters: EvidenceReviewParameters


class ProposeSubquestionsAction(BaseAction):
    action_type: Literal["propose_subquestions"]
    parameters: SubquestionParameters


class CreatePendingConjectureAction(BaseAction):
    action_type: Literal["create_pending_conjecture"]
    parameters: PendingConjectureParameters


class CreatePendingOpenProblemAction(BaseAction):
    action_type: Literal["create_pending_open_problem"]
    parameters: PendingOpenProblemParameters


class DesignBoundedExperimentAction(BaseAction):
    action_type: Literal["design_bounded_experiment"]
    parameters: BoundedExperimentParameters


class RunTrustedExperimentAction(BaseAction):
    action_type: Literal["run_trusted_experiment"]
    parameters: BoundedExperimentParameters


class InspectExperimentResultAction(BaseAction):
    action_type: Literal["inspect_experiment_result"]
    parameters: ExperimentResultParameters


class ReassessPlanAction(BaseAction):
    action_type: Literal["reassess_plan"]
    parameters: ReassessPlanParameters


class StopAction(BaseAction):
    action_type: Literal["stop"]
    parameters: EmptyParameters


class ApprovePendingAction(BaseAction):
    """Included so a high-authority request is paused and recorded, never run."""

    action_type: Literal["approve_pending"]
    parameters: ApprovalParameters


ControllerAction = Annotated[
    InspectStateAction
    | ReviewStoredEvidenceAction
    | ProposeSubquestionsAction
    | CreatePendingConjectureAction
    | CreatePendingOpenProblemAction
    | DesignBoundedExperimentAction
    | RunTrustedExperimentAction
    | InspectExperimentResultAction
    | ReassessPlanAction
    | StopAction
    | ApprovePendingAction,
    Field(discriminator="action_type"),
]


class ControllerDecision(StrictBase):
    """The LLM must return exactly one typed next action per iteration."""

    action: ControllerAction


@dataclass(frozen=True)
class TrustedExperimentResult:
    """Result returned by the sole controller-runnable experiment handler."""

    run_id: int
    summary: str
    output: dict[str, object]


def get_trusted_artifact(
    artifact_id: int,
    cluster_id: int,
    db_path: str | Path | None = None,
) -> CodeArtifact:
    """Validate controller trust metadata without executing the artifact itself."""

    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        artifact = db.get_code_artifact(connection, artifact_id)
    if artifact is None:
        raise ValueError(f"code artifact {artifact_id} does not exist")
    if artifact.status != "tested":
        raise ValueError(f"code artifact {artifact_id} is not trusted for controller execution")
    if artifact.artifact_type not in {"checker", "solver"}:
        raise ValueError(f"code artifact {artifact_id} is not a trusted checker or solver")
    if artifact.cluster_id is not None and artifact.cluster_id != cluster_id:
        raise ValueError(f"code artifact {artifact_id} belongs to another research cluster")
    return artifact


def run_trusted_bounded_experiment(
    parameters: BoundedExperimentParameters,
    cluster_id: int,
    db_path: str | Path | None = None,
) -> TrustedExperimentResult:
    """Run the fixed in-process tiny ATS-family safety handler.

    The artifact record supplies provenance only. Its path, entrypoint, notes,
    and any LLM-provided command text are never executed by this handler.
    """

    get_trusted_artifact(parameters.artifact_id, cluster_id, db_path)
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
    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        run_id = db.insert_experiment_run(
            connection,
            ExperimentRun(
                artifact_id=parameters.artifact_id,
                cluster_id=cluster_id,
                experiment_type="ats_bounded_safety",
                input_json=game.to_dict(),
                output_json=output,
                result_summary=summary,
                command_run="trusted_in_process:ats_bounded_safety",
                git_commit_hash=get_current_commit_hash(),
                notes="Controller-runnable bounded handler; no artifact command was executed.",
            ),
        )
    return TrustedExperimentResult(run_id=run_id, summary=summary, output=output)

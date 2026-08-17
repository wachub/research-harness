"""Reviewable, LLM-assisted research planning over existing harness state.

The planner has no authority to approve claims, create durable research
objects, or execute experiments. It may only create pending candidate entries
when the caller explicitly requests that action.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from . import db
from .llm import LLMClient, LLMError, LLMMessage, LLMRequest
from .schemas import Conjecture, OpenProblem, PendingEntry, ResearchCluster, StrictBase


StateKind = Literal[
    "research_cluster",
    "concept",
    "paper",
    "theorem",
    "reduction",
    "conjecture",
    "open_problem",
    "evidence",
    "experiment_run",
]


class ResearchGoalRequest(StrictBase):
    """A user-supplied goal and optional cluster scope."""

    goal: str = Field(min_length=1)
    cluster_id: int | None = None


class ExistingStateReference(StrictBase):
    """An LLM-selected reference back to state supplied by the planner."""

    kind: StateKind
    object_id: int
    relevance: str = Field(min_length=1)


class ProposedSubquestion(StrictBase):
    question: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    dependencies_or_assumptions: list[str] = Field(default_factory=list)
    uncertainty_note: str = Field(min_length=1)


class ProposedConjecture(StrictBase):
    statement: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    dependencies_or_assumptions: list[str] = Field(default_factory=list)
    uncertainty_note: str = Field(min_length=1)


class ProposedLiteratureTask(StrictBase):
    task: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    dependencies_or_assumptions: list[str] = Field(default_factory=list)
    uncertainty_note: str = Field(min_length=1)


class ProposedExperiment(StrictBase):
    bounded_design: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    dependencies_or_assumptions: list[str] = Field(default_factory=list)
    uncertainty_note: str = Field(min_length=1)


class ResearchPlan(StrictBase):
    """A constrained plan that contains proposals, not established results."""

    interpreted_goal: str = Field(min_length=1)
    relevant_existing_state: list[ExistingStateReference] = Field(default_factory=list)
    recommended_cluster_id: int | None = None
    cluster_rationale: str | None = None
    proposed_subquestions: list[ProposedSubquestion] = Field(default_factory=list, max_length=5)
    proposed_conjectures: list[ProposedConjecture] = Field(default_factory=list, max_length=3)
    proposed_literature_tasks: list[ProposedLiteratureTask] = Field(default_factory=list, max_length=5)
    proposed_experiments: list[ProposedExperiment] = Field(default_factory=list, max_length=3)
    uncertainty_note: str = Field(min_length=1)


@dataclass(frozen=True)
class ResearchPlanningResult:
    """A plan or a clear unavailable result; neither implies execution."""

    available: bool
    message: str
    plan: ResearchPlan | None = None
    selected_cluster: ResearchCluster | None = None
    state_context: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    provider_metadata: dict[str, Any] = field(default_factory=dict)


PLANNING_INSTRUCTIONS = """
You are a research-planning assistant. Propose only reviewable next directions;
do not state or imply that a theorem, conjecture, open problem, or experiment
result is established. Use only IDs and state supplied in the request when
populating relevant_existing_state. Do not invent citations or evidence.

Return one JSON object matching this shape:
{
  "interpreted_goal": "...",
  "relevant_existing_state": [{"kind": "theorem", "object_id": 1, "relevance": "..."}],
  "recommended_cluster_id": 1,
  "cluster_rationale": "...",
  "proposed_subquestions": [{"question": "...", "rationale": "...", "dependencies_or_assumptions": [], "uncertainty_note": "..."}],
  "proposed_conjectures": [{"statement": "...", "rationale": "...", "dependencies_or_assumptions": [], "uncertainty_note": "..."}],
  "proposed_literature_tasks": [{"task": "...", "rationale": "...", "dependencies_or_assumptions": [], "uncertainty_note": "..."}],
  "proposed_experiments": [{"bounded_design": "...", "rationale": "...", "dependencies_or_assumptions": [], "uncertainty_note": "..."}],
  "uncertainty_note": "..."
}

Use empty arrays when no useful proposal fits. Keep the plan small and label
uncertainty explicitly. Proposed experiments are designs only: do not execute
or claim their outcomes.
""".strip()


def plan_research(
    goal: str,
    cluster_id: int | None = None,
    *,
    use_llm: bool = False,
    db_path: str | Path | None = None,
    client: LLMClient | None = None,
) -> ResearchPlanningResult:
    """Load project state and return a validated, non-executing research plan."""

    request = ResearchGoalRequest(goal=goal, cluster_id=cluster_id)
    selected_cluster, state_context, known_ids = _load_state_context(request.cluster_id, db_path)
    if not use_llm:
        return ResearchPlanningResult(
            available=False,
            message="Research planning requires --llm and a configured remote LLM provider; nothing was written.",
            selected_cluster=selected_cluster,
            state_context=state_context,
        )

    provider_client = client or LLMClient()
    if not provider_client.available:
        return ResearchPlanningResult(
            available=False,
            message="No remote LLM provider is configured; nothing was written.",
            selected_cluster=selected_cluster,
            state_context=state_context,
            provider_metadata=provider_client.metadata(),
        )

    prompt = {
        "instruction": PLANNING_INSTRUCTIONS,
        "goal": request.goal,
        "selected_cluster_id": request.cluster_id,
        "research_state": state_context,
    }
    try:
        plan = provider_client.complete_json(
            LLMRequest(
                messages=(
                    LLMMessage(
                        role="system",
                        content="You produce cautious, reviewable research plans from supplied project state.",
                    ),
                    LLMMessage(role="user", content=json.dumps(prompt, sort_keys=True)),
                ),
                temperature=0.0,
                json_mode=True,
            ),
            ResearchPlan,
        )
        _validate_plan_references(plan, known_ids, request.cluster_id)
    except LLMError as exc:
        return ResearchPlanningResult(
            available=False,
            message=f"Research planning did not produce a valid plan: {exc}",
            selected_cluster=selected_cluster,
            state_context=state_context,
            provider_metadata=provider_client.metadata(),
        )

    return ResearchPlanningResult(
        available=True,
        message="Research plan generated for review; no proposal was persisted or executed.",
        plan=plan,
        selected_cluster=selected_cluster,
        state_context=state_context,
        provider_metadata=provider_client.metadata(),
    )


def save_plan_as_pending(
    result: ResearchPlanningResult,
    goal: str,
    db_path: str | Path | None = None,
) -> list[int]:
    """Persist only compatible plan proposals as existing pending-entry types."""

    if not result.available or result.plan is None:
        raise ValueError("a valid research plan is required before saving pending proposals")
    cluster_id = result.selected_cluster.cluster_id if result.selected_cluster else result.plan.recommended_cluster_id
    warning_prefix = "LLM-proposed research plan; requires human review"
    entries: list[PendingEntry] = []

    for proposal in result.plan.proposed_conjectures:
        candidate = Conjecture(
            statement=proposal.statement,
            cluster_id=cluster_id,
            motivation=proposal.rationale,
            expected_status="unknown",
            confidence="needs_review",
            attack_plan="; ".join(proposal.dependencies_or_assumptions) or None,
            notes=f"Uncertainty: {proposal.uncertainty_note}",
        )
        entries.append(
            PendingEntry(
                entry_type="conjecture_seed",
                payload=candidate.model_dump(),
                source_text=goal,
                warnings=[warning_prefix, f"uncertainty: {proposal.uncertainty_note}"],
            )
        )

    for proposal in result.plan.proposed_subquestions:
        candidate = OpenProblem(
            title=_title(proposal.question),
            statement=proposal.question,
            context=proposal.rationale,
            cluster_id=cluster_id,
            notes=(
                "LLM-proposed research question, not an established open problem. "
                f"Dependencies/assumptions: {'; '.join(proposal.dependencies_or_assumptions) or 'none'}. "
                f"Uncertainty: {proposal.uncertainty_note}"
            ),
        )
        entries.append(
            PendingEntry(
                entry_type="open_problem",
                payload=candidate.model_dump(),
                source_text=goal,
                warnings=[warning_prefix, f"uncertainty: {proposal.uncertainty_note}"],
            )
        )

    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        return [db.insert_pending_entry(connection, entry) for entry in entries]


def _load_state_context(
    cluster_id: int | None,
    db_path: str | Path | None,
) -> tuple[ResearchCluster | None, dict[str, list[dict[str, Any]]], dict[StateKind, set[int]]]:
    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        selected_cluster = db.get_cluster(connection, cluster_id) if cluster_id is not None else None
        if cluster_id is not None and selected_cluster is None:
            raise ValueError(f"cluster {cluster_id} does not exist")
        all_clusters = db.list_clusters(connection)
        clusters = [selected_cluster] if selected_cluster is not None else all_clusters
        concepts = db.list_concepts(connection)
        papers = db.list_papers(connection, cluster_id=cluster_id)
        theorems = db.list_theorems(connection, cluster_id=cluster_id)
        reductions = db.list_reductions(connection, cluster_id=cluster_id)
        conjectures = db.list_conjectures(connection, cluster_id=cluster_id)
        problems = db.list_open_problems(connection, cluster_id=cluster_id)
        experiments = db.list_experiment_runs(connection, cluster_id=cluster_id)
        paper_ids = {paper.id for paper in papers if paper.id is not None}
        evidence = [
            item
            for item in db.list_evidence_spans(connection)
            if cluster_id is None or item.paper_id in paper_ids
        ]

    context = {
        "clusters": [
            {"id": cluster.cluster_id, "name": cluster.name, "description": cluster.description}
            for cluster in clusters
        ],
        "concepts": [
            {"id": concept.concept_id, "name": concept.name, "type": concept.concept_type, "aliases": concept.aliases}
            for concept in concepts
        ],
        "papers": [
            {"id": paper.id, "title": paper.title, "year": paper.year, "venue": paper.venue}
            for paper in papers
        ],
        "theorems": [
            {"id": theorem.id, "title": theorem.title, "statement": theorem.statement, "type": theorem.theorem_type}
            for theorem in theorems
        ],
        "reductions": [
            {"id": reduction.id, "title": reduction.title, "statement": reduction.statement}
            for reduction in reductions
        ],
        "conjectures": [
            {"id": conjecture.id, "title": conjecture.title, "statement": conjecture.statement, "status": conjecture.status}
            for conjecture in conjectures
        ],
        "open_problems": [
            {"id": problem.id, "title": problem.title, "statement": problem.statement, "status": problem.status}
            for problem in problems
        ],
        "evidence": [
            {"id": item.evidence_id, "paper_id": item.paper_id, "summary": item.quote_or_summary}
            for item in evidence
        ],
        "experiment_runs": [
            {
                "id": run.run_id,
                "type": run.experiment_type,
                "summary": run.result_summary,
                "input_path": run.input_path,
                "output_path": run.output_path,
            }
            for run in experiments
        ],
    }
    known_ids: dict[StateKind, set[int]] = {
        "research_cluster": {cluster.cluster_id for cluster in clusters if cluster.cluster_id is not None},
        "concept": {concept.concept_id for concept in concepts if concept.concept_id is not None},
        "paper": {paper.id for paper in papers if paper.id is not None},
        "theorem": {theorem.id for theorem in theorems if theorem.id is not None},
        "reduction": {reduction.id for reduction in reductions if reduction.id is not None},
        "conjecture": {conjecture.id for conjecture in conjectures if conjecture.id is not None},
        "open_problem": {problem.id for problem in problems if problem.id is not None},
        "evidence": {item.evidence_id for item in evidence if item.evidence_id is not None},
        "experiment_run": {run.run_id for run in experiments if run.run_id is not None},
    }
    return selected_cluster, context, known_ids


def _validate_plan_references(
    plan: ResearchPlan,
    known_ids: dict[StateKind, set[int]],
    selected_cluster_id: int | None,
) -> None:
    if selected_cluster_id is not None and plan.recommended_cluster_id not in {None, selected_cluster_id}:
        raise LLMError("LLM plan recommended a cluster different from the explicitly selected cluster")
    if plan.recommended_cluster_id is not None and plan.recommended_cluster_id not in known_ids["research_cluster"]:
        raise LLMError("LLM plan referenced an unknown research cluster")
    for reference in plan.relevant_existing_state:
        if reference.object_id not in known_ids[reference.kind]:
            raise LLMError(f"LLM plan referenced unknown {reference.kind} id {reference.object_id}")


def _title(value: str) -> str:
    return value.replace("\n", " ").strip()[:80].rstrip(" .") or "Proposed research question"

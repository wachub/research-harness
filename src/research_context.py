"""Capped, task-scoped state summaries for controller decisions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import db
from .schemas import ResearchTask, ResearchUnit, ResearchUnitLink


@dataclass(frozen=True)
class ControllerContext:
    """Only the small set of state that may be supplied to the controller LLM."""

    task: ResearchTask
    summary: dict[str, list[dict[str, Any]]]
    known_ids: dict[str, set[int]]
    active_unit: ResearchUnit | None = None
    unit_links: tuple[ResearchUnitLink, ...] = ()


def load_controller_context(
    task_id: int,
    db_path: str | Path | None = None,
    unit_id: int | None = None,
) -> ControllerContext:
    """Load a bounded current-state snapshot for one existing task."""

    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        task = db.get_task(connection, task_id)
        if task is None:
            raise ValueError(f"task {task_id} does not exist")
        units = db.list_research_units(connection, task_id=task_id)
        active_unit = db.get_research_unit(connection, unit_id) if unit_id is not None else db.next_research_unit(connection, task_id)
        if unit_id is not None and (active_unit is None or active_unit.task_id != task_id):
            raise ValueError(f"research unit {unit_id} does not belong to task {task_id}")
        if active_unit is not None and active_unit.status in {"finished", "abandoned"}:
            raise ValueError(f"research unit {active_unit.unit_id} is closed")
        unit_links = tuple(db.list_research_unit_links(connection, active_unit.unit_id)) if active_unit and active_unit.unit_id else ()
        papers = db.list_papers(connection, task_id=task_id)[:8]
        theorems = db.list_theorems(connection, task_id=task_id)[:8]
        reductions = db.list_reductions(connection, task_id=task_id)[:6]
        conjectures = db.list_conjectures(connection, task_id=task_id)[:8]
        problems = db.list_open_problems(connection, task_id=task_id)[:8]
        results = db.list_derived_results(connection, task_id=task_id)[-8:]
        attempts = db.list_proof_attempts(connection, task_id=task_id)[-6:]
        runs = db.list_experiment_runs(connection, task_id=task_id)[-6:]
        concepts = db.list_concepts(connection)[:10]
        artifacts = [
            artifact
            for artifact in db.list_code_artifacts(connection)
            if artifact.task_id in {None, task_id}
        ][:6]
        paper_ids = {paper.id for paper in papers if paper.id is not None}
        evidence = [
            item
            for item in db.list_evidence_spans(connection)
            if item.paper_id in paper_ids
        ][:8]

    visible_units = units[-12:]
    if active_unit is not None and all(item.unit_id != active_unit.unit_id for item in visible_units):
        visible_units = [*visible_units, active_unit]
    summary = {
        "research_units": [
            {"id": item.unit_id, "kind": item.kind, "title": item.title,
             "status": item.status, "parent_unit_id": item.parent_unit_id}
            for item in visible_units
        ],
        "task": [{"id": task.task_id, "name": task.name, "description": _short(task.description)}],
        "concepts": [
            {"id": item.concept_id, "name": item.name, "type": item.concept_type}
            for item in concepts
        ],
        "papers": [
            {"id": item.id, "title": item.title, "year": item.year, "venue": item.venue}
            for item in papers
        ],
        "theorems": [
            {"id": item.id, "title": item.title, "statement": _short(item.statement), "type": item.theorem_type, "confidence": item.confidence, "source_paper_id": item.source_paper_id}
            for item in theorems
        ],
        "reductions": [
            {"id": item.id, "title": item.title, "statement": _short(item.statement)}
            for item in reductions
        ],
        "conjectures": [
            {"id": item.id, "title": item.title, "statement": _short(item.statement), "status": item.status, "confidence": item.confidence}
            for item in conjectures
        ],
        "open_problems": [
            {"id": item.id, "title": item.title, "statement": _short(item.statement), "status": item.status}
            for item in problems
        ],
        "derived_results": [
            {"id": item.id, "title": item.title, "statement": _short(item.statement), "status": item.status, "dependencies": item.dependencies}
            for item in results
        ],
        "proof_attempts": [
            {"id": item.id, "target_type": item.target_type, "target_id": item.target_id,
             "status": item.status, "strategy": _short(item.strategy)}
            for item in attempts
        ],
        "evidence": [
            {"id": item.evidence_id, "paper_id": item.paper_id, "summary": _short(item.quote_or_summary), "confidence": item.confidence}
            for item in evidence
        ],
        "experiment_runs": [
            {"id": item.run_id, "type": item.experiment_type, "summary": _short(item.result_summary)}
            for item in runs
        ],
        "code_artifacts": [
            {
                "id": item.artifact_id,
                "name": item.name,
                "type": item.artifact_type,
                "status": item.status,
            }
            for item in artifacts
        ],
    }
    known_ids = {
        "research_unit": {item.unit_id for item in visible_units if item.unit_id is not None},
        "research_task": {task.task_id} if task.task_id is not None else set(),
        "concept": {item.concept_id for item in concepts if item.concept_id is not None},
        "paper": {item.id for item in papers if item.id is not None},
        "theorem": {item.id for item in theorems if item.id is not None},
        "reduction": {item.id for item in reductions if item.id is not None},
        "conjecture": {item.id for item in conjectures if item.id is not None},
        "open_problem": {item.id for item in problems if item.id is not None},
        "derived_result": {item.id for item in results if item.id is not None},
        "proof_attempt": {item.id for item in attempts if item.id is not None},
        "evidence": {item.evidence_id for item in evidence if item.evidence_id is not None},
        "experiment_run": {item.run_id for item in runs if item.run_id is not None},
        "code_artifact": {item.artifact_id for item in artifacts if item.artifact_id is not None},
    }
    return ControllerContext(
        task=task, summary=summary, known_ids=known_ids,
        active_unit=active_unit, unit_links=unit_links,
    )


def validate_context_reference(context: ControllerContext, kind: str, object_id: int) -> None:
    """Reject unknown or out-of-scope references before handler dispatch."""

    if object_id not in context.known_ids.get(kind, set()):
        raise ValueError(f"unknown or out-of-scope {kind} id {object_id}")


def _short(value: str | None, limit: int = 500) -> str | None:
    if value is None:
        return None
    return value.replace("\n", " ").strip()[:limit]

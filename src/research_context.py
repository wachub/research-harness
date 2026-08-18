"""Capped, cluster-scoped state summaries for controller decisions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import db
from .schemas import ResearchCluster


@dataclass(frozen=True)
class ControllerContext:
    """Only the small set of state that may be supplied to the controller LLM."""

    cluster: ResearchCluster
    summary: dict[str, list[dict[str, Any]]]
    known_ids: dict[str, set[int]]


def load_controller_context(
    cluster_id: int,
    db_path: str | Path | None = None,
) -> ControllerContext:
    """Load a bounded current-state snapshot for one existing cluster."""

    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        cluster = db.get_cluster(connection, cluster_id)
        if cluster is None:
            raise ValueError(f"cluster {cluster_id} does not exist")
        papers = db.list_papers(connection, cluster_id=cluster_id)[:8]
        theorems = db.list_theorems(connection, cluster_id=cluster_id)[:8]
        reductions = db.list_reductions(connection, cluster_id=cluster_id)[:6]
        conjectures = db.list_conjectures(connection, cluster_id=cluster_id)[:8]
        problems = db.list_open_problems(connection, cluster_id=cluster_id)[:8]
        runs = db.list_experiment_runs(connection, cluster_id=cluster_id)[-6:]
        concepts = db.list_concepts(connection)[:10]
        artifacts = [
            artifact
            for artifact in db.list_code_artifacts(connection)
            if artifact.cluster_id in {None, cluster_id}
        ][:6]
        paper_ids = {paper.id for paper in papers if paper.id is not None}
        evidence = [
            item
            for item in db.list_evidence_spans(connection)
            if item.paper_id in paper_ids
        ][:8]

    summary = {
        "cluster": [{"id": cluster.cluster_id, "name": cluster.name, "description": _short(cluster.description)}],
        "concepts": [
            {"id": item.concept_id, "name": item.name, "type": item.concept_type}
            for item in concepts
        ],
        "papers": [
            {"id": item.id, "title": item.title, "year": item.year, "venue": item.venue}
            for item in papers
        ],
        "theorems": [
            {"id": item.id, "title": item.title, "statement": _short(item.statement), "type": item.theorem_type}
            for item in theorems
        ],
        "reductions": [
            {"id": item.id, "title": item.title, "statement": _short(item.statement)}
            for item in reductions
        ],
        "conjectures": [
            {"id": item.id, "title": item.title, "statement": _short(item.statement), "status": item.status}
            for item in conjectures
        ],
        "open_problems": [
            {"id": item.id, "title": item.title, "statement": _short(item.statement), "status": item.status}
            for item in problems
        ],
        "evidence": [
            {"id": item.evidence_id, "paper_id": item.paper_id, "summary": _short(item.quote_or_summary)}
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
        "research_cluster": {cluster.cluster_id} if cluster.cluster_id is not None else set(),
        "concept": {item.concept_id for item in concepts if item.concept_id is not None},
        "paper": {item.id for item in papers if item.id is not None},
        "theorem": {item.id for item in theorems if item.id is not None},
        "reduction": {item.id for item in reductions if item.id is not None},
        "conjecture": {item.id for item in conjectures if item.id is not None},
        "open_problem": {item.id for item in problems if item.id is not None},
        "evidence": {item.evidence_id for item in evidence if item.evidence_id is not None},
        "experiment_run": {item.run_id for item in runs if item.run_id is not None},
        "code_artifact": {item.artifact_id for item in artifacts if item.artifact_id is not None},
    }
    return ControllerContext(cluster=cluster, summary=summary, known_ids=known_ids)


def validate_context_reference(context: ControllerContext, kind: str, object_id: int) -> None:
    """Reject unknown or out-of-scope references before handler dispatch."""

    if object_id not in context.known_ids.get(kind, set()):
        raise ValueError(f"unknown or out-of-scope {kind} id {object_id}")


def _short(value: str | None, limit: int = 500) -> str | None:
    if value is None:
        return None
    return value.replace("\n", " ").strip()[:limit]

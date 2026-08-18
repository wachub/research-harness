"""Read models and safe review adapters used by the local Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import db
from .curate import analyze_pending_entry, approve_pending, flag_pending, reject_pending
from .git_utils import get_current_commit_hash
from .llm import LLMConfiguration


def dashboard_summary(db_path: str | Path | None = None) -> dict[str, Any]:
    """Load compact global metrics and attention items from existing state."""

    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        clusters = db.list_clusters(connection)
        pending = db.list_pending_entries(connection, status=None)
        counts = {
            "research_clusters": len(clusters),
            "active_clusters": sum(cluster.status == "active" for cluster in clusters),
            "papers": len(db.list_papers(connection)),
            "theorems": len(db.list_theorems(connection)),
            "reductions": len(db.list_reductions(connection)),
            "conjectures": len(db.list_conjectures(connection)),
            "open_problems": len(db.list_open_problems(connection)),
            "pending_entries": len(pending),
            "proof_attempts": len(db.list_proof_attempts(connection)),
            "evidence_spans": len(db.list_evidence_spans(connection)),
            "experiment_runs": len(db.list_experiment_runs(connection)),
            "code_artifacts": len(db.list_code_artifacts(connection)),
        }
        recent = _recent_records(connection)
        active_conjectures = [item.model_dump() for item in db.list_conjectures(connection) if item.status == "active"][:8]
        active_problems = [item.model_dump() for item in db.list_open_problems(connection) if item.status == "active"][:8]
    return {
        "counts": counts,
        "recent": recent,
        "pending": pending_review_items(db_path, limit=10),
        "active_conjectures": active_conjectures,
        "active_open_problems": active_problems,
    }


def project_summaries(db_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Return per-cluster counts without placing query logic in the GUI."""

    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        pending = db.list_pending_entries(connection, status=None)
        result = []
        for cluster in db.list_clusters(connection):
            cluster_id = cluster.cluster_id
            result.append(
                {
                    **cluster.model_dump(),
                    "papers": len(db.list_papers(connection, cluster_id=cluster_id)),
                    "theorems": len(db.list_theorems(connection, cluster_id=cluster_id)),
                    "conjectures": len(db.list_conjectures(connection, cluster_id=cluster_id)),
                    "open_problems": len(db.list_open_problems(connection, cluster_id=cluster_id)),
                    "proof_attempts": len(db.list_proof_attempts(connection, cluster_id=cluster_id)),
                    "experiment_runs": len(db.list_experiment_runs(connection, cluster_id=cluster_id)),
                    "pending_proposals": sum(_pending_cluster_id(item) == cluster_id for item in pending),
                }
            )
    return result


def project_detail(cluster_id: int, db_path: str | Path | None = None) -> dict[str, Any]:
    """Collect all existing objects associated with one selected project."""

    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        cluster = db.get_cluster(connection, cluster_id)
        if cluster is None:
            raise ValueError(f"cluster {cluster_id} does not exist")
        papers = db.list_papers(connection, cluster_id=cluster_id)
        paper_ids = {paper.id for paper in papers if paper.id is not None}
        notes = [item for item in db.list_literature_notes(connection) if item.paper_id in paper_ids]
        summaries = [item for item in db.list_literature_summaries(connection) if item.paper_id in paper_ids]
        evidence = [item for item in db.list_evidence_spans(connection) if item.paper_id in paper_ids]
        pending = [item for item in db.list_pending_entries(connection, status=None) if _pending_cluster_id(item) == cluster_id]
        artifacts = [item for item in db.list_code_artifacts(connection) if item.cluster_id in {None, cluster_id}]
        return {
            "cluster": cluster.model_dump(),
            "theorems": _dump(db.list_theorems(connection, cluster_id=cluster_id)),
            "reductions": _dump(db.list_reductions(connection, cluster_id=cluster_id)),
            "derived_results": _dump(db.list_derived_results(connection, cluster_id=cluster_id)),
            "conjectures": _dump(db.list_conjectures(connection, cluster_id=cluster_id)),
            "open_problems": _dump(db.list_open_problems(connection, cluster_id=cluster_id)),
            "proof_attempts": _dump(db.list_proof_attempts(connection, cluster_id=cluster_id)),
            "papers": _dump(papers),
            "evidence_spans": _dump(evidence),
            "literature_notes": _dump(notes),
            "literature_summaries": _dump(summaries),
            "experiment_runs": _dump(db.list_experiment_runs(connection, cluster_id=cluster_id)),
            "code_artifacts": _dump(artifacts),
            "pending_entries": _dump(pending),
        }


def pending_review_items(db_path: str | Path | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    """Read the review queue using the existing duplicate/curation analysis."""

    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        raw_rows = {row["id"]: row for row in db.list_explorer_records(connection, "pending_entries", limit=5_000)}
        result = []
        for entry in db.list_pending_entries(connection, status=None):
            warnings, duplicates = analyze_pending_entry(connection, entry)
            raw = raw_rows.get(entry.id, {})
            result.append(
                {
                    **entry.model_dump(),
                    "created_at": raw.get("created_at"),
                    "reviewed_at": raw.get("reviewed_at"),
                    "duplicates": [duplicate.__dict__ for duplicate in duplicates],
                    "warnings": warnings,
                }
            )
    return result[:limit] if limit is not None else result


def approve_review_item(entry_id: int, db_path: str | Path | None = None) -> dict[str, Any]:
    """Use the exact existing approval path; no GUI-specific approval exists."""

    result = approve_pending(entry_id, db_path=db_path)
    return {"pending_id": result.pending_id, "inserted_table": result.inserted_table, "inserted_id": result.inserted_id}


def reject_review_item(entry_id: int, reason: str | None = None, db_path: str | Path | None = None) -> None:
    reject_pending(entry_id, reason=reason, db_path=db_path)


def flag_review_item(entry_id: int, reason: str, db_path: str | Path | None = None) -> None:
    flag_pending(entry_id, reason=reason, db_path=db_path)


def explorer_records(table_name: str, db_path: str | Path | None = None, limit: int = 500) -> list[dict[str, Any]]:
    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        return db.list_explorer_records(connection, table_name, limit=limit)


def project_timeline(cluster_id: int, db_path: str | Path | None = None) -> list[dict[str, Any]]:
    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        if db.get_cluster(connection, cluster_id) is None:
            raise ValueError(f"cluster {cluster_id} does not exist")
        return db.list_project_timeline(connection, cluster_id)


def experiment_records(cluster_id: int | None = None, db_path: str | Path | None = None) -> list[dict[str, Any]]:
    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        artifacts = {item.artifact_id: item.name for item in db.list_code_artifacts(connection)}
        conjectures = {item.id: item.title for item in db.list_conjectures(connection)}
        clusters = {item.cluster_id: item.name for item in db.list_clusters(connection)}
        return [
            {
                **run.model_dump(),
                "cluster_name": clusters.get(run.cluster_id),
                "conjecture_title": conjectures.get(run.conjecture_id),
                "artifact_name": artifacts.get(run.artifact_id),
            }
            for run in db.list_experiment_runs(connection, cluster_id=cluster_id)
        ]


def read_experiment_output(run_id: int, db_path: str | Path | None = None, max_bytes: int = 200_000) -> str:
    """Read a local result file only when it is a regular project-local file."""

    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        run = db.get_experiment_run(connection, run_id)
    if run is None:
        raise ValueError(f"experiment run {run_id} does not exist")
    candidate = run.output_json.get("stdout_stderr_path") or run.output_path
    if not isinstance(candidate, str) or not candidate:
        return "No local output file is recorded for this experiment."
    path = Path(candidate).resolve()
    root = db.PROJECT_ROOT.resolve()
    if root not in path.parents or not path.is_file():
        return "Recorded output file is unavailable or outside the project workspace."
    return path.read_text(encoding="utf-8", errors="replace")[:max_bytes]


def system_status(db_path: str | Path | None = None) -> dict[str, Any]:
    """Return safe diagnostics only; credentials and raw environment are excluded."""

    path = db.resolve_db_path(db_path)
    configuration = LLMConfiguration.from_environment()
    with db.get_connection(path) as connection:
        db.create_tables(connection)
        pending_count = len(db.list_pending_entries(connection, status="pending"))
        artifacts = db.list_code_artifacts(connection)
    return {
        "llm_provider": configuration.provider,
        "llm_model": configuration.model,
        "remote_llm_available": configuration.remote_enabled,
        "database_path": str(path),
        "database_exists": path.exists(),
        "git_commit": get_current_commit_hash(),
        "pending_review_items": pending_count,
        "tested_code_artifacts": sum(item.status == "tested" for item in artifacts),
        "code_artifacts": len(artifacts),
    }


def _recent_records(connection) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for table in ("papers", "theorems", "conjectures", "open_problems", "pending_entries", "experiment_runs"):
        for row in db.list_explorer_records(connection, table, limit=5):
            records.append({"table": table, "created_at": row.get("created_at"), "record": row})
    return sorted(records, key=lambda item: item["created_at"] or "", reverse=True)[:12]


def _pending_cluster_id(entry) -> int | None:
    value = entry.payload.get("cluster_id")
    return value if isinstance(value, int) else None


def _dump(items) -> list[dict[str, Any]]:
    return [item.model_dump() for item in items]

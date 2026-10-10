"""Read models and safe review adapters used by the local Streamlit dashboard."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import db
from .curate import analyze_pending_entry, approve_pending, flag_pending, reject_pending
from .git_utils import get_current_commit_hash
from .llm import LLMClient, LLMConfiguration
from .research_controller import ResearchController
from .autonomous_research import AutonomousResearch
from .research_planning import assess_literature_need
from .research_policy import ControllerMode


def dashboard_summary(db_path: str | Path | None = None) -> dict[str, Any]:
    """Load compact global metrics and attention items from existing state."""

    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        tasks = db.list_tasks(connection)
        pending = db.list_pending_entries(connection, status=None)
        units = db.list_research_units(connection)
        counts = {
            "research_tasks": len(tasks),
            "research_units": len(units),
            "open_research_units": sum(unit.status in {"proposed", "ready", "active"} for unit in units),
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


def create_research_task(
    name: str, description: str, db_path: str | Path | None = None,
) -> int:
    """Create a fixed objective; subsequent progress belongs in research units."""

    from .schemas import ResearchTask

    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        return db.insert_task(connection, ResearchTask(name=name, description=description))


def project_summaries(db_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Return per-task counts without placing query logic in the GUI."""

    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        pending = db.list_pending_entries(connection, status=None)
        result = []
        for task in db.list_tasks(connection):
            task_id = task.task_id
            result.append(
                {
                    **task.model_dump(),
                    "papers": len(db.list_papers(connection, task_id=task_id)),
                    "research_units": len(db.list_research_units(connection, task_id=task_id)),
                    "theorems": len(db.list_theorems(connection, task_id=task_id)),
                    "conjectures": len(db.list_conjectures(connection, task_id=task_id)),
                    "open_problems": len(db.list_open_problems(connection, task_id=task_id)),
                    "proof_attempts": len(db.list_proof_attempts(connection, task_id=task_id)),
                    "experiment_runs": len(db.list_experiment_runs(connection, task_id=task_id)),
                    "pending_proposals": sum(_pending_task_id(item) == task_id for item in pending),
                }
            )
    return result


def project_detail(task_id: int, db_path: str | Path | None = None) -> dict[str, Any]:
    """Collect all existing objects associated with one selected project."""

    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        task = db.get_task(connection, task_id)
        if task is None:
            raise ValueError(f"task {task_id} does not exist")
        papers = db.list_papers(connection, task_id=task_id)
        paper_ids = {paper.id for paper in papers if paper.id is not None}
        notes = [item for item in db.list_literature_notes(connection) if item.paper_id in paper_ids]
        summaries = [item for item in db.list_literature_summaries(connection) if item.paper_id in paper_ids]
        evidence = [item for item in db.list_evidence_spans(connection) if item.paper_id in paper_ids]
        pending = [item for item in db.list_pending_entries(connection, status=None) if _pending_task_id(item) == task_id]
        artifacts = [item for item in db.list_code_artifacts(connection) if item.task_id in {None, task_id}]
        units = db.list_research_units(connection, task_id=task_id)
        next_unit = db.next_research_unit(connection, task_id)
        return {
            "task": task.model_dump(),
            "research_units": _dump(units),
            "research_unit_links": {
                str(unit.unit_id): _dump(db.list_research_unit_links(connection, unit.unit_id))
                for unit in units if unit.unit_id is not None
            },
            "next_research_unit": next_unit.model_dump() if next_unit else None,
            "theorems": _dump(db.list_theorems(connection, task_id=task_id)),
            "reductions": _dump(db.list_reductions(connection, task_id=task_id)),
            "derived_results": _dump(db.list_derived_results(connection, task_id=task_id)),
            "conjectures": _dump(db.list_conjectures(connection, task_id=task_id)),
            "open_problems": _dump(db.list_open_problems(connection, task_id=task_id)),
            "proof_attempts": _dump(db.list_proof_attempts(connection, task_id=task_id)),
            "papers": _dump(papers),
            "evidence_spans": _dump(evidence),
            "literature_notes": _dump(notes),
            "literature_summaries": _dump(summaries),
            "experiment_runs": _dump(db.list_experiment_runs(connection, task_id=task_id)),
            "code_artifacts": _dump(artifacts),
            "pending_entries": _dump(pending),
        }



def run_research_steps(
    task_id: int, steps: int = 1, unit_id: int | None = None,
    literature_from_unit: bool = False, db_path: str | Path | None = None,
    client: LLMClient | None = None, model_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Create one completed unit per successful autonomous research step."""

    result = AutonomousResearch(client or LLMClient(model_overrides=model_overrides), db_path).run(
        task_id, steps=steps, unit_id=unit_id,
        literature_from_unit=literature_from_unit,
    )
    return {
        "status": result.status, "message": result.message,
        "task_id": result.task_id, "research_unit_ids": list(result.unit_ids),
        "steps_completed": len(result.unit_ids),
        "error_type": result.error_type,
        "error_details": list(result.error_details),
        "diagnostic_id": result.diagnostic_id,
        "diagnostics": result.diagnostics,
        "blocked_research_unit_ids": list(result.blocked_unit_ids),
    }


def continue_research(
    task_id: int, unit_id: int | None = None,
    db_path: str | Path | None = None, client: LLMClient | None = None,
    model_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run one explicit, bounded controller step for the selected task/unit."""

    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        task = db.get_task(connection, task_id)
        if task is None:
            raise ValueError(f"research task {task_id} does not exist")
    goal = task.description or task.name
    controller = ResearchController(client or LLMClient(model_overrides=model_overrides), db_path=db_path)
    result = controller.run(
        goal, task_id, unit_id=unit_id, mode=ControllerMode.AUTONOMOUS, max_steps=1,
    )
    return {
        "status": result.status,
        "message": result.message,
        "research_unit_id": controller.selected_unit_id,
        "steps": [asdict(step) for step in result.steps],
        "log_path": str(result.log_path) if result.log_path else None,
    }


def assess_research_unit_literature(
    task_id: int, unit_id: int,
    db_path: str | Path | None = None, client: LLMClient | None = None,
    model_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Assess existing literature coverage without fetching sources or writing state."""

    result = assess_literature_need(
        task_id, unit_id, db_path=db_path,
        client=client or LLMClient(model_overrides=model_overrides),
    )
    return {
        "available": result.available,
        "message": result.message,
        "assessment": result.assessment.model_dump() if result.assessment else None,
        "provider": result.provider_metadata,
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


def project_timeline(task_id: int, db_path: str | Path | None = None) -> list[dict[str, Any]]:
    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        if db.get_task(connection, task_id) is None:
            raise ValueError(f"task {task_id} does not exist")
        return db.list_project_timeline(connection, task_id)


def experiment_records(task_id: int | None = None, db_path: str | Path | None = None) -> list[dict[str, Any]]:
    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        artifacts = {item.artifact_id: item.name for item in db.list_code_artifacts(connection)}
        conjectures = {item.id: item.title for item in db.list_conjectures(connection)}
        tasks = {item.task_id: item.name for item in db.list_tasks(connection)}
        return [
            {
                **run.model_dump(),
                "task_name": tasks.get(run.task_id),
                "conjecture_title": conjectures.get(run.conjecture_id),
                "artifact_name": artifacts.get(run.artifact_id),
            }
            for run in db.list_experiment_runs(connection, task_id=task_id)
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
    results_root = (db.PROJECT_ROOT / "results").resolve()
    if results_root not in path.parents or not path.is_file():
        return "Recorded output file is unavailable or outside the permitted results directory."
    return path.read_text(encoding="utf-8", errors="replace")[:max_bytes]


def system_status(db_path: str | Path | None = None) -> dict[str, Any]:
    """Return safe diagnostics only; credentials and raw environment are excluded."""

    configuration = LLMConfiguration.from_environment()
    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        pending_count = len(db.list_pending_entries(connection, status="pending"))
        artifacts = db.list_code_artifacts(connection)
        database_name = connection.execute("SELECT current_database() AS name").fetchone()["name"]
    return {
        "llm_provider": configuration.provider,
        "llm_model": configuration.model,
        "remote_llm_available": configuration.remote_enabled,
        "database_engine": "postgresql",
        "database_name": database_name,
        "git_commit": get_current_commit_hash(),
        "pending_review_items": pending_count,
        "tested_code_artifacts": sum(item.status == "tested" for item in artifacts),
        "code_artifacts": len(artifacts),
    }


def _recent_records(connection) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for table in ("research_units", "papers", "theorems", "conjectures", "open_problems", "pending_entries", "experiment_runs"):
        for row in db.list_explorer_records(connection, table, limit=5):
            records.append({"table": table, "created_at": row.get("created_at"), "record": row})
    return sorted(records, key=lambda item: item["created_at"] or "", reverse=True)[:12]


def _pending_task_id(entry) -> int | None:
    value = entry.payload.get("task_id")
    return value if isinstance(value, int) else None


def _dump(items) -> list[dict[str, Any]]:
    return [item.model_dump() for item in items]

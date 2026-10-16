"""Research records over local MCP stdio.

The harness launches a task-scoped server for proof attempts. A trusted local
client may opt into all-task reads and narrowly scoped provisional writes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field, field_validator
from dotenv import load_dotenv

from . import db
from .schemas import Conjecture, ProofAttempt, ResearchEvent, ResearchUnit, ResearchUnitLink, StrictBase


RecordKind = Literal[
    "theorem", "conjecture", "derived_result", "open_problem",
    "proof_attempt", "reduction", "research_unit", "paper",
]
_KINDS = RecordKind.__args__
_MAX_RESULTS = 10
_MAX_TEXT = 4000
_READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False,
                             idempotentHint=True, openWorldHint=False)
_ADDITIVE_WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False,
                                  idempotentHint=False, openWorldHint=False)


class _WriteInput(StrictBase):
    task_id: int = Field(gt=0)
    parent_unit_id: int | None = Field(default=None, gt=0)
    uncertainty_note: str = Field(min_length=1, max_length=2000)

    @field_validator("uncertainty_note")
    @classmethod
    def nonblank_uncertainty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("uncertainty note must not be blank")
        return value


class _ConjectureWrite(_WriteInput):
    title: str = Field(min_length=1, max_length=300)
    statement: str = Field(min_length=1, max_length=12000)
    rationale: str = Field(min_length=1, max_length=4000)

    @field_validator("title", "statement", "rationale")
    @classmethod
    def nonblank_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("research text must not be blank")
        return value


class _ProofAttemptWrite(_WriteInput):
    target_type: Literal["theorem", "conjecture", "derived_result"]
    target_id: int = Field(gt=0)
    strategy: str = Field(min_length=1, max_length=12000)
    outcome: str = Field(min_length=1, max_length=4000)

    @field_validator("strategy", "outcome")
    @classmethod
    def nonblank_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("research text must not be blank")
        return value


def _records(connection: db.PostgresConnection, task_id: int | None, kind: RecordKind) -> list[Any]:
    readers = {
        "theorem": db.list_theorems,
        "conjecture": db.list_conjectures,
        "derived_result": db.list_derived_results,
        "open_problem": db.list_open_problems,
        "proof_attempt": db.list_proof_attempts,
        "reduction": db.list_reductions,
        "research_unit": db.list_research_units,
        "paper": db.list_papers,
    }
    return readers[kind](connection, task_id=task_id)


def _identifier(kind: RecordKind, record: Any) -> int:
    field = {
        "theorem": "theorem_id", "conjecture": "id", "derived_result": "id",
        "open_problem": "id", "proof_attempt": "id", "reduction": "id",
        "research_unit": "unit_id", "paper": "id",
    }[kind]
    return int(getattr(record, field))


def _bounded(value: Any, limit: int = _MAX_TEXT) -> Any:
    if isinstance(value, str):
        return value[:limit]
    if isinstance(value, list):
        return [_bounded(item, limit) for item in value[:20]]
    if isinstance(value, dict):
        return {key: _bounded(item, limit) for key, item in value.items()}
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return str(value)[:limit]


def _record_data(kind: RecordKind, record: Any, *, detail: bool) -> dict[str, Any]:
    data = record.model_dump(mode="json")
    if not detail:
        selected = ("title", "name", "statement", "strategy", "result_summary",
                    "purpose", "outcome_note", "status", "confidence", "target_type",
                    "target_id", "source_paper_id", "task_id")
        data = {key: data[key] for key in selected if key in data}
        data = _bounded(data, 500)
    else:
        data = _bounded(data)
        # Paper files and code are never provided through this research lookup.
        data.pop("pdf_path", None)
    if kind == "research_unit":
        data["unit_kind"] = record.kind
    return {**data, "kind": kind, "id": _identifier(kind, record)}


def search_records(
    task_id: int | None, query: str, kinds: list[RecordKind] | None = None,
    limit: int = 5, db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Search already stored records within one task; return short candidate summaries."""

    query = query.strip()
    if not 1 <= len(query) <= 300 or not 1 <= limit <= _MAX_RESULTS:
        raise ValueError("query or result limit is invalid")
    selected = kinds or list(_KINDS)
    if len(selected) > len(_KINDS) or any(kind not in _KINDS for kind in selected):
        raise ValueError("record kinds are invalid")
    terms = [term.casefold() for term in query.split() if term.strip()]
    matches: list[tuple[int, dict[str, Any]]] = []
    with db.get_connection(db_path) as connection:
        if task_id is not None and db.get_task(connection, task_id) is None:
            raise ValueError("research task does not exist")
        for kind in dict.fromkeys(selected):
            for record in _records(connection, task_id, kind):
                item = _record_data(kind, record, detail=False)
                haystack = " ".join(str(value) for value in item.values()).casefold()
                score = sum(term in haystack for term in terms)
                if score:
                    matches.append((score, item))
    matches.sort(key=lambda pair: (-pair[0], pair[1]["kind"], pair[1]["id"]))
    return {"records": [item for _, item in matches[:limit]]}


def get_record(
    task_id: int | None, kind: RecordKind, record_id: int,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Read one task-visible record and its directly linked evidence, if any."""

    if kind not in _KINDS or record_id < 1:
        raise ValueError("record kind or id is invalid")
    with db.get_connection(db_path) as connection:
        if task_id is not None and db.get_task(connection, task_id) is None:
            raise ValueError("research task does not exist")
        record = next((item for item in _records(connection, task_id, kind)
                       if _identifier(kind, item) == record_id), None)
        if record is None:
            raise ValueError("record is not available in this research task")
        result = _record_data(kind, record, detail=True)
        if kind in {"theorem", "conjecture", "open_problem", "derived_result", "reduction"}:
            evidence = db.list_evidence_spans(connection, kind, record_id)[:5]
            allowed_papers = {paper.id for paper in db.list_papers(connection, task_id=task_id)}
            result["evidence"] = [
                _bounded(item.model_dump(mode="json"), 1200)
                for item in evidence if item.paper_id in allowed_papers
            ]
    return result


def list_tasks(db_path: str | Path | None = None) -> dict[str, Any]:
    """List research objectives for a trusted all-task MCP client."""

    with db.get_connection(db_path) as connection:
        tasks = db.list_tasks(connection)
    return {"tasks": [
        {"task_id": task.task_id, "name": _bounded(task.name, 300),
         "description": _bounded(task.description, 1000)}
        for task in tasks[:100]
    ]}


def read_table(
    table_name: str, limit: int = 10, offset: int = 0,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Page through allow-listed research tables; never accept model-supplied SQL."""

    if table_name not in db.EXPLORABLE_TABLES:
        raise ValueError("table is not available for research browsing")
    if not 1 <= limit <= 20 or not 0 <= offset <= 1_000_000:
        raise ValueError("table page bounds are invalid")
    with db.get_connection(db_path) as connection:
        key = db._pk_column(connection, table_name)
        rows = connection.execute(
            f"SELECT * FROM {table_name} ORDER BY {key} LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    result: list[dict[str, Any]] = []
    size = 0
    for row in rows:
        item = _bounded(db._decode_explorer_row(row), 1000)
        encoded_size = len(json.dumps(item, ensure_ascii=False))
        if size + encoded_size > 40_000:
            break
        result.append(item)
        size += encoded_size
    return {"table": table_name, "rows": result, "offset": offset,
            "next_offset": offset + len(result), "truncated": len(result) < len(rows)}


def _check_write_scope(
    connection: db.PostgresConnection, task_id: int, parent_unit_id: int | None,
) -> None:
    if db.get_task(connection, task_id) is None:
        raise ValueError("research task does not exist")
    if parent_unit_id is not None:
        parent = db.get_research_unit(connection, parent_unit_id)
        if parent is None or parent.task_id != task_id:
            raise ValueError("parent research unit must belong to this task")


def _producing_unit(
    connection: db.PostgresConnection, task_id: int, kind: str, record_id: int,
) -> int | None:
    for link in db.list_research_units_for_object(connection, kind, record_id):
        if link.relation == "produces":
            unit = db.get_research_unit(connection, link.unit_id)
            if unit is not None and unit.task_id == task_id:
                return unit.unit_id
    return None


def add_provisional_conjecture(
    task_id: int, title: str, statement: str, rationale: str, uncertainty_note: str,
    parent_unit_id: int | None = None, db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Atomically store a provisional conjecture and its completed research unit."""

    item = _ConjectureWrite(task_id=task_id, title=title, statement=statement,
                            rationale=rationale, uncertainty_note=uncertainty_note,
                            parent_unit_id=parent_unit_id)
    with db.get_connection(db_path) as connection:
        # Serialize writes within a task so a retry cannot create a second copy.
        connection.execute("SELECT pg_advisory_xact_lock(18124, ?)", (item.task_id,))
        _check_write_scope(connection, item.task_id, item.parent_unit_id)
        for existing in db.list_conjectures(connection, item.task_id):
            if (existing.task_id == item.task_id
                    and existing.statement.strip().casefold() == item.statement.casefold()):
                unit_id = _producing_unit(connection, item.task_id, "conjecture", existing.id)
                if unit_id is not None:
                    return {"created": False, "task_id": item.task_id, "unit_id": unit_id,
                            "kind": "conjecture", "id": existing.id, "status": existing.status}
        conjecture_id = db.insert_conjecture(connection, Conjecture(
            title=item.title, statement=item.statement, motivation=item.rationale,
            task_id=item.task_id, confidence="needs_review", status="active",
            notes=f"Provisional MCP hypothesis. Uncertainty: {item.uncertainty_note}",
        ))
        unit_id = db.insert_research_unit(connection, ResearchUnit(
            task_id=item.task_id, parent_unit_id=item.parent_unit_id,
            kind="conjecture", title=item.title, purpose=item.rationale,
            status="finished", outcome_note=f"Provisional conjecture recorded. Uncertainty: {item.uncertainty_note}",
        ))
        db.link_research_unit(connection, ResearchUnitLink(
            unit_id=unit_id, relation="produces", object_type="conjecture",
            object_id=conjecture_id,
        ))
        db.insert_research_event(connection, ResearchEvent(
            task_id=item.task_id, event_type="mcp_research_write", object_type="research_unit",
            object_id=unit_id, summary=f"MCP added provisional conjecture: {item.title}",
            metadata={"source": "local_mcp", "record_type": "conjecture",
                      "record_id": conjecture_id},
        ))
    return {"created": True, "task_id": item.task_id, "unit_id": unit_id,
            "kind": "conjecture", "id": conjecture_id, "status": "active"}


def record_proof_attempt(
    task_id: int, target_type: Literal["theorem", "conjecture", "derived_result"],
    target_id: int, strategy: str, outcome: str, uncertainty_note: str,
    parent_unit_id: int | None = None, db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Atomically store an unverified proof attempt for a task-visible target."""

    item = _ProofAttemptWrite(task_id=task_id, target_type=target_type,
                              target_id=target_id, strategy=strategy, outcome=outcome,
                              uncertainty_note=uncertainty_note,
                              parent_unit_id=parent_unit_id)
    with db.get_connection(db_path) as connection:
        connection.execute("SELECT pg_advisory_xact_lock(18124, ?)", (item.task_id,))
        _check_write_scope(connection, item.task_id, item.parent_unit_id)
        if not any(_identifier(item.target_type, target) == item.target_id
                   for target in _records(connection, item.task_id, item.target_type)):
            raise ValueError("proof target is not available in this research task")
        notes = f"{item.outcome} Unverified: {item.uncertainty_note}"
        for existing in db.list_proof_attempts(connection, item.task_id):
            if (existing.task_id == item.task_id and existing.target_type == item.target_type
                    and existing.target_id == item.target_id
                    and existing.strategy == item.strategy and existing.notes == notes):
                unit_id = _producing_unit(connection, item.task_id, "proof_attempt", existing.id)
                if unit_id is not None:
                    return {"created": False, "task_id": item.task_id, "unit_id": unit_id,
                            "kind": "proof_attempt", "id": existing.id, "status": existing.status}
        attempt_id = db.insert_proof_attempt(connection, ProofAttempt(
            target_type=item.target_type, target_id=item.target_id,
            strategy=item.strategy, notes=notes, status="draft", task_id=item.task_id,
        ))
        title = f"Proof attempt for {item.target_type} {item.target_id}"
        unit_id = db.insert_research_unit(connection, ResearchUnit(
            task_id=item.task_id, parent_unit_id=item.parent_unit_id,
            kind="proof_attempt", title=title, purpose=item.strategy,
            status="finished", outcome_note=notes,
        ))
        db.link_research_unit(connection, ResearchUnitLink(
            unit_id=unit_id, relation="investigates", object_type=item.target_type,
            object_id=item.target_id,
        ))
        db.link_research_unit(connection, ResearchUnitLink(
            unit_id=unit_id, relation="produces", object_type="proof_attempt",
            object_id=attempt_id,
        ))
        db.insert_research_event(connection, ResearchEvent(
            task_id=item.task_id, event_type="mcp_research_write", object_type="research_unit",
            object_id=unit_id, summary=f"MCP recorded draft proof attempt for {item.target_type} {item.target_id}",
            metadata={"source": "local_mcp", "record_type": "proof_attempt",
                      "record_id": attempt_id},
        ))
    return {"created": True, "task_id": item.task_id, "unit_id": unit_id,
            "kind": "proof_attempt", "id": attempt_id, "status": "draft"}


def create_server(
    task_id: int | None, db_path: str | Path | None = None, *,
    enable_writes: bool = False, write_db_path: str | Path | None = None,
) -> FastMCP:
    """Create a local MCP server; writes require explicit all-task opt-in."""

    if task_id is not None and enable_writes:
        raise ValueError("task-scoped proof lookup cannot enable writes")

    server = FastMCP("research-harness")

    if task_id is None:
        @server.tool(name="list_research_tables", annotations=_READ_ONLY)
        def list_research_tables() -> dict[str, Any]:
            """List the harness's readable research database tables."""
            return {"tables": list(db.EXPLORABLE_TABLES)}

        @server.tool(name="read_research_table", annotations=_READ_ONLY)
        def read_research_table(table_name: str, limit: int = 10,
                                offset: int = 0) -> dict[str, Any]:
            """Read one bounded page from an allow-listed research table."""
            return read_table(table_name, limit, offset, db_path)

        @server.tool(name="list_research_tasks", annotations=_READ_ONLY)
        def list_research_tasks() -> dict[str, Any]:
            """List stored research tasks and their fixed objectives."""
            return list_tasks(db_path)

        @server.tool(name="search_research_records", annotations=_READ_ONLY)
        def search_research_records(query: str, kinds: list[RecordKind] | None = None,
                                    limit: int = 5, task_id: int | None = None) -> dict[str, Any]:
            """Search stored findings across tasks; optionally restrict to one task ID."""
            return search_records(task_id, query, kinds, limit, db_path)

        if enable_writes:
            write_path = write_db_path if write_db_path is not None else db_path

            @server.tool(name="add_provisional_conjecture", annotations=_ADDITIVE_WRITE)
            def add_conjecture_tool(
                task_id: int, title: str, statement: str, rationale: str,
                uncertainty_note: str, parent_unit_id: int | None = None,
            ) -> dict[str, Any]:
                """Create a provisional conjecture and linked research unit; never establish a theorem."""
                return add_provisional_conjecture(
                    task_id, title, statement, rationale, uncertainty_note,
                    parent_unit_id, write_path,
                )

            @server.tool(name="record_proof_attempt", annotations=_ADDITIVE_WRITE)
            def record_attempt_tool(
                task_id: int, target_type: Literal["theorem", "conjecture", "derived_result"],
                target_id: int, strategy: str, outcome: str, uncertainty_note: str,
                parent_unit_id: int | None = None,
            ) -> dict[str, Any]:
                """Create a draft proof attempt and linked research unit; no proof is certified."""
                return record_proof_attempt(
                    task_id, target_type, target_id, strategy, outcome,
                    uncertainty_note, parent_unit_id, write_path,
                )
    else:
        @server.tool(name="search_research_records", annotations=_READ_ONLY)
        def search_research_records(query: str, kinds: list[RecordKind] | None = None,
                                    limit: int = 5) -> dict[str, Any]:
            """Search stored research findings and units; returns IDs and short summaries."""
            return search_records(task_id, query, kinds, limit, db_path)

    @server.tool(name="get_research_record", annotations=_READ_ONLY)
    def get_research_record(kind: RecordKind, record_id: int) -> dict[str, Any]:
        """Get one record's statement, assumptions, status, provenance and evidence."""
        return get_record(task_id, kind, record_id, db_path)

    return server


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Local research MCP server")
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--task-id", type=int)
    scope.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--enable-writes", action="store_true",
                        help="allow two provisional write tools in all-task mode only")
    args = parser.parse_args()
    if args.task_id is not None and args.task_id < 1:
        parser.error("--task-id must be positive")
    if args.enable_writes and not args.all_tasks:
        parser.error("--enable-writes requires --all-tasks")
    db_path = os.environ.get("RESEARCH_HARNESS_MCP_DB_PATH") or None
    write_db_path = os.environ.get("RESEARCH_HARNESS_MCP_WRITE_DATABASE_URL") or None
    create_server(None if args.all_tasks else args.task_id, db_path,
                  enable_writes=args.enable_writes,
                  write_db_path=write_db_path).run(transport="stdio")


if __name__ == "__main__":
    main()

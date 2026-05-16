"""SQLite persistence helpers for the research harness."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from .schemas import (
    Conjecture,
    DerivedResult,
    Model,
    OpenProblem,
    Paper,
    PendingEntry,
    ProofAttempt,
    Reduction,
    Theorem,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "research.db"


def resolve_db_path(db_path: str | Path | None = None) -> Path:
    """Return a concrete database path and ensure its parent exists."""

    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open a SQLite connection with row dictionaries enabled."""

    connection = sqlite3.connect(resolve_db_path(db_path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(db_path: str | Path | None = None) -> None:
    """Create all harness tables if they do not already exist."""

    with get_connection(db_path) as connection:
        create_tables(connection)


def create_tables(connection: sqlite3.Connection) -> None:
    """Create the database schema."""

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            authors_json TEXT NOT NULL,
            year INTEGER NOT NULL,
            venue TEXT,
            pdf_path TEXT,
            notes TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            model_type TEXT NOT NULL,
            description TEXT,
            data_json TEXT NOT NULL,
            source_paper_id INTEGER,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(source_paper_id) REFERENCES papers(id)
        );

        CREATE TABLE IF NOT EXISTS theorems (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            statement TEXT NOT NULL,
            assumptions_json TEXT NOT NULL,
            conclusion TEXT,
            paper_id INTEGER,
            tags_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(paper_id) REFERENCES papers(id)
        );

        CREATE TABLE IF NOT EXISTS reductions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            source_problem TEXT NOT NULL,
            target_problem TEXT NOT NULL,
            statement TEXT NOT NULL,
            assumptions_json TEXT NOT NULL,
            paper_id INTEGER,
            tags_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(paper_id) REFERENCES papers(id)
        );

        CREATE TABLE IF NOT EXISTS open_problems (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            statement TEXT NOT NULL,
            context TEXT,
            status TEXT NOT NULL,
            paper_id INTEGER,
            tags_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(paper_id) REFERENCES papers(id)
        );

        CREATE TABLE IF NOT EXISTS pending_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_type TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            source_text TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            duplicate_of TEXT,
            warnings_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            reviewed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS derived_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            statement TEXT NOT NULL,
            dependencies_json TEXT NOT NULL,
            proof_sketch TEXT,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS conjectures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            statement TEXT NOT NULL,
            rationale TEXT,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS proof_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            target_type TEXT NOT NULL,
            target_id INTEGER NOT NULL,
            strategy TEXT NOT NULL,
            notes TEXT,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
    )


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _json_loads(value: str | None, default: Any) -> Any:
    if value is None:
        return default
    return json.loads(value)


def _last_insert_id(connection: sqlite3.Connection) -> int:
    row = connection.execute("SELECT last_insert_rowid() AS id").fetchone()
    return int(row["id"])


def insert_paper(connection: sqlite3.Connection, paper: Paper) -> int:
    """Insert a paper and return its id."""

    connection.execute(
        """
        INSERT INTO papers (title, authors_json, year, venue, pdf_path, notes)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (paper.title, _json_dumps(paper.authors), paper.year, paper.venue, paper.pdf_path, paper.notes),
    )
    return _last_insert_id(connection)


def list_papers(connection: sqlite3.Connection) -> list[Paper]:
    """List papers ordered by insertion id."""

    rows = connection.execute("SELECT * FROM papers ORDER BY id").fetchall()
    return [
        Paper(
            id=row["id"],
            title=row["title"],
            authors=_json_loads(row["authors_json"], []),
            year=row["year"],
            venue=row["venue"],
            pdf_path=row["pdf_path"],
            notes=row["notes"],
        )
        for row in rows
    ]


def insert_model(connection: sqlite3.Connection, model: Model) -> int:
    """Insert a model record and return its id."""

    connection.execute(
        """
        INSERT INTO models (name, model_type, description, data_json, source_paper_id)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            model.name,
            model.model_type,
            model.description,
            _json_dumps(model.data),
            model.source_paper_id,
        ),
    )
    return _last_insert_id(connection)


def list_models(connection: sqlite3.Connection) -> list[Model]:
    rows = connection.execute("SELECT * FROM models ORDER BY id").fetchall()
    return [
        Model(
            id=row["id"],
            name=row["name"],
            model_type=row["model_type"],
            description=row["description"],
            data=_json_loads(row["data_json"], {}),
            source_paper_id=row["source_paper_id"],
        )
        for row in rows
    ]


def insert_theorem(connection: sqlite3.Connection, theorem: Theorem) -> int:
    """Insert a theorem record and return its id."""

    connection.execute(
        """
        INSERT INTO theorems (title, statement, assumptions_json, conclusion, paper_id, tags_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            theorem.title,
            theorem.statement,
            _json_dumps(theorem.assumptions),
            theorem.conclusion,
            theorem.paper_id,
            _json_dumps(theorem.tags),
        ),
    )
    return _last_insert_id(connection)


def list_theorems(connection: sqlite3.Connection) -> list[Theorem]:
    rows = connection.execute("SELECT * FROM theorems ORDER BY id").fetchall()
    return [
        Theorem(
            id=row["id"],
            title=row["title"],
            statement=row["statement"],
            assumptions=_json_loads(row["assumptions_json"], []),
            conclusion=row["conclusion"],
            paper_id=row["paper_id"],
            tags=_json_loads(row["tags_json"], []),
        )
        for row in rows
    ]


def insert_reduction(connection: sqlite3.Connection, reduction: Reduction) -> int:
    """Insert a reduction record and return its id."""

    connection.execute(
        """
        INSERT INTO reductions
            (title, source_problem, target_problem, statement, assumptions_json, paper_id, tags_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            reduction.title,
            reduction.source_problem,
            reduction.target_problem,
            reduction.statement,
            _json_dumps(reduction.assumptions),
            reduction.paper_id,
            _json_dumps(reduction.tags),
        ),
    )
    return _last_insert_id(connection)


def list_reductions(connection: sqlite3.Connection) -> list[Reduction]:
    rows = connection.execute("SELECT * FROM reductions ORDER BY id").fetchall()
    return [
        Reduction(
            id=row["id"],
            title=row["title"],
            source_problem=row["source_problem"],
            target_problem=row["target_problem"],
            statement=row["statement"],
            assumptions=_json_loads(row["assumptions_json"], []),
            paper_id=row["paper_id"],
            tags=_json_loads(row["tags_json"], []),
        )
        for row in rows
    ]


def insert_open_problem(connection: sqlite3.Connection, problem: OpenProblem) -> int:
    """Insert an open problem record and return its id."""

    connection.execute(
        """
        INSERT INTO open_problems (title, statement, context, status, paper_id, tags_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            problem.title,
            problem.statement,
            problem.context,
            problem.status,
            problem.paper_id,
            _json_dumps(problem.tags),
        ),
    )
    return _last_insert_id(connection)


def list_open_problems(connection: sqlite3.Connection) -> list[OpenProblem]:
    rows = connection.execute("SELECT * FROM open_problems ORDER BY id").fetchall()
    return [
        OpenProblem(
            id=row["id"],
            title=row["title"],
            statement=row["statement"],
            context=row["context"],
            status=row["status"],
            paper_id=row["paper_id"],
            tags=_json_loads(row["tags_json"], []),
        )
        for row in rows
    ]


def insert_pending_entry(connection: sqlite3.Connection, entry: PendingEntry) -> int:
    """Insert a pending extracted entry and return its id."""

    connection.execute(
        """
        INSERT INTO pending_entries
            (entry_type, payload_json, source_text, status, duplicate_of, warnings_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            entry.entry_type,
            _json_dumps(entry.payload),
            entry.source_text,
            entry.status,
            entry.duplicate_of,
            _json_dumps(entry.warnings),
        ),
    )
    return _last_insert_id(connection)


def get_pending_entry(connection: sqlite3.Connection, entry_id: int) -> PendingEntry | None:
    """Fetch a pending entry by id."""

    row = connection.execute("SELECT * FROM pending_entries WHERE id = ?", (entry_id,)).fetchone()
    if row is None:
        return None
    return _row_to_pending_entry(row)


def list_pending_entries(
    connection: sqlite3.Connection,
    status: str | None = "pending",
) -> list[PendingEntry]:
    """List pending entries, optionally filtered by status."""

    if status is None:
        rows = connection.execute("SELECT * FROM pending_entries ORDER BY id").fetchall()
    else:
        rows = connection.execute(
            "SELECT * FROM pending_entries WHERE status = ? ORDER BY id",
            (status,),
        ).fetchall()
    return [_row_to_pending_entry(row) for row in rows]


def update_pending_status(
    connection: sqlite3.Connection,
    entry_id: int,
    status: str,
    duplicate_of: str | None = None,
    warnings: Iterable[str] | None = None,
) -> None:
    """Update curation status for a pending entry."""

    connection.execute(
        """
        UPDATE pending_entries
        SET status = ?,
            duplicate_of = COALESCE(?, duplicate_of),
            warnings_json = COALESCE(?, warnings_json),
            reviewed_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (
            status,
            duplicate_of,
            _json_dumps(list(warnings)) if warnings is not None else None,
            entry_id,
        ),
    )


def insert_derived_result(connection: sqlite3.Connection, result: DerivedResult) -> int:
    """Insert a derived result and return its id."""

    connection.execute(
        """
        INSERT INTO derived_results (title, statement, dependencies_json, proof_sketch, status)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            result.title,
            result.statement,
            _json_dumps(result.dependencies),
            result.proof_sketch,
            result.status,
        ),
    )
    return _last_insert_id(connection)


def list_derived_results(connection: sqlite3.Connection) -> list[DerivedResult]:
    """List derived results ordered by insertion id."""

    rows = connection.execute("SELECT * FROM derived_results ORDER BY id").fetchall()
    return [
        DerivedResult(
            id=row["id"],
            title=row["title"],
            statement=row["statement"],
            dependencies=_json_loads(row["dependencies_json"], []),
            proof_sketch=row["proof_sketch"],
            status=row["status"],
        )
        for row in rows
    ]


def update_derived_result_status(connection: sqlite3.Connection, result_id: int, status: str) -> None:
    """Update the review status for a derived result."""

    connection.execute("UPDATE derived_results SET status = ? WHERE id = ?", (status, result_id))


def insert_conjecture(connection: sqlite3.Connection, conjecture: Conjecture) -> int:
    """Insert a conjecture and return its id."""

    connection.execute(
        """
        INSERT INTO conjectures (title, statement, rationale, status)
        VALUES (?, ?, ?, ?)
        """,
        (conjecture.title, conjecture.statement, conjecture.rationale, conjecture.status),
    )
    return _last_insert_id(connection)


def list_conjectures(connection: sqlite3.Connection) -> list[Conjecture]:
    """List conjectures ordered by insertion id."""

    rows = connection.execute("SELECT * FROM conjectures ORDER BY id").fetchall()
    return [
        Conjecture(
            id=row["id"],
            title=row["title"],
            statement=row["statement"],
            rationale=row["rationale"],
            status=row["status"],
        )
        for row in rows
    ]


def update_conjecture_status(connection: sqlite3.Connection, conjecture_id: int, status: str) -> None:
    """Update the review status for a conjecture."""

    connection.execute("UPDATE conjectures SET status = ? WHERE id = ?", (status, conjecture_id))


def insert_proof_attempt(connection: sqlite3.Connection, attempt: ProofAttempt) -> int:
    """Insert a proof attempt and return its id."""

    connection.execute(
        """
        INSERT INTO proof_attempts (target_type, target_id, strategy, notes, status)
        VALUES (?, ?, ?, ?, ?)
        """,
        (attempt.target_type, attempt.target_id, attempt.strategy, attempt.notes, attempt.status),
    )
    return _last_insert_id(connection)


def list_proof_attempts(connection: sqlite3.Connection) -> list[ProofAttempt]:
    """List proof attempts ordered by insertion id."""

    rows = connection.execute("SELECT * FROM proof_attempts ORDER BY id").fetchall()
    return [
        ProofAttempt(
            id=row["id"],
            target_type=row["target_type"],
            target_id=row["target_id"],
            strategy=row["strategy"],
            notes=row["notes"],
            status=row["status"],
        )
        for row in rows
    ]


def update_proof_attempt_status(connection: sqlite3.Connection, attempt_id: int, status: str) -> None:
    """Update the review status for a proof attempt."""

    connection.execute("UPDATE proof_attempts SET status = ? WHERE id = ?", (status, attempt_id))


def _row_to_pending_entry(row: sqlite3.Row) -> PendingEntry:
    return PendingEntry(
        id=row["id"],
        entry_type=row["entry_type"],
        payload=_json_loads(row["payload_json"], {}),
        source_text=row["source_text"],
        status=row["status"],
        duplicate_of=row["duplicate_of"],
        warnings=_json_loads(row["warnings_json"], []),
    )

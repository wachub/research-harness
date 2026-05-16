"""Manual curation workflow for extracted pending entries."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from pydantic import ValidationError

from . import db
from .schemas import (
    Model,
    OpenProblem,
    PendingEntry,
    Reduction,
    SCHEMA_BY_ENTRY_TYPE,
    Theorem,
)


@dataclass(frozen=True)
class DuplicateCandidate:
    """A possible duplicate found during curation."""

    table: str
    entry_id: int
    score: float
    title: str


@dataclass(frozen=True)
class ApprovalResult:
    """Result of approving a pending entry."""

    pending_id: int
    inserted_table: str
    inserted_id: int
    warnings: list[str]
    duplicates: list[DuplicateCandidate]


def analyze_pending_entry(
    connection,
    entry: PendingEntry,
    duplicate_threshold: float = 0.82,
) -> tuple[list[str], list[DuplicateCandidate]]:
    """Return curation warnings and possible duplicates for a pending entry."""

    warnings = list(entry.warnings)
    warnings.extend(_missing_assumption_warnings(entry))
    duplicates = find_duplicates(connection, entry, duplicate_threshold)
    if duplicates:
        warnings.append("possible duplicate detected")
    return _dedupe_strings(warnings), duplicates


def find_duplicates(
    connection,
    entry: PendingEntry,
    duplicate_threshold: float = 0.82,
) -> list[DuplicateCandidate]:
    """Find records with similar titles or statements."""

    payload = entry.payload
    title = str(payload.get("title") or payload.get("name") or "")
    statement = str(payload.get("statement") or payload.get("description") or "")
    candidates: list[DuplicateCandidate] = []

    if entry.entry_type == "theorem":
        for theorem in db.list_theorems(connection):
            score = _record_similarity(title, statement, theorem.title, theorem.statement)
            if score >= duplicate_threshold:
                candidates.append(DuplicateCandidate("theorems", theorem.id or 0, score, theorem.title))
    elif entry.entry_type == "reduction":
        for reduction in db.list_reductions(connection):
            score = _record_similarity(title, statement, reduction.title, reduction.statement)
            if score >= duplicate_threshold:
                candidates.append(DuplicateCandidate("reductions", reduction.id or 0, score, reduction.title))
    elif entry.entry_type == "open_problem":
        for problem in db.list_open_problems(connection):
            score = _record_similarity(title, statement, problem.title, problem.statement)
            if score >= duplicate_threshold:
                candidates.append(DuplicateCandidate("open_problems", problem.id or 0, score, problem.title))
    elif entry.entry_type == "model":
        for model in db.list_models(connection):
            score = _record_similarity(title, statement, model.name, model.description or "")
            if score >= duplicate_threshold:
                candidates.append(DuplicateCandidate("models", model.id or 0, score, model.name))

    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)


def approve_pending(entry_id: int, db_path: str | Path | None = None) -> ApprovalResult:
    """Validate a pending entry, insert it into the main table, and mark approved."""

    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        entry = db.get_pending_entry(connection, entry_id)
        if entry is None:
            raise ValueError(f"pending entry {entry_id} does not exist")
        if entry.status != "pending":
            raise ValueError(f"pending entry {entry_id} is already {entry.status}")

        warnings, duplicates = analyze_pending_entry(connection, entry)
        inserted_table, inserted_id = _insert_validated_payload(connection, entry)
        duplicate_of = _format_duplicate(duplicates[0]) if duplicates else None
        db.update_pending_status(
            connection,
            entry_id,
            "approved",
            duplicate_of=duplicate_of,
            warnings=warnings,
        )
        return ApprovalResult(entry_id, inserted_table, inserted_id, warnings, duplicates)


def reject_pending(entry_id: int, reason: str | None = None, db_path: str | Path | None = None) -> None:
    """Reject a pending entry without inserting it into a main table."""

    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        entry = db.get_pending_entry(connection, entry_id)
        if entry is None:
            raise ValueError(f"pending entry {entry_id} does not exist")
        warnings = list(entry.warnings)
        if reason:
            warnings.append(f"rejected: {reason}")
        db.update_pending_status(connection, entry_id, "rejected", warnings=_dedupe_strings(warnings))


def _insert_validated_payload(connection, entry: PendingEntry) -> tuple[str, int]:
    schema = SCHEMA_BY_ENTRY_TYPE[entry.entry_type]
    try:
        validated = schema.model_validate(entry.payload)
    except ValidationError as exc:
        raise ValueError(f"pending entry {entry.id} failed validation: {exc}") from exc

    if isinstance(validated, Model):
        return "models", db.insert_model(connection, validated)
    if isinstance(validated, Theorem):
        return "theorems", db.insert_theorem(connection, validated)
    if isinstance(validated, Reduction):
        return "reductions", db.insert_reduction(connection, validated)
    if isinstance(validated, OpenProblem):
        return "open_problems", db.insert_open_problem(connection, validated)
    raise TypeError(f"unsupported pending entry type: {entry.entry_type}")


def _missing_assumption_warnings(entry: PendingEntry) -> list[str]:
    if entry.entry_type not in {"theorem", "reduction"}:
        return []
    assumptions = entry.payload.get("assumptions")
    if not assumptions:
        return ["missing assumptions"]
    return []


def _record_similarity(title_a: str, statement_a: str, title_b: str, statement_b: str) -> float:
    title_score = _similarity(title_a, title_b)
    statement_score = _similarity(statement_a, statement_b)
    return max(title_score, statement_score)


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, _normalize(left), _normalize(right)).ratio()


def _normalize(value: str) -> str:
    return " ".join(value.lower().split())


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _format_duplicate(candidate: DuplicateCandidate) -> str:
    return f"{candidate.table}:{candidate.entry_id}"

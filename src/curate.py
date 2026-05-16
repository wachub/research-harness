"""Manual curation workflow for extracted pending entries."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from pydantic import ValidationError

from . import db
from .schemas import (
    Concept,
    Conjecture,
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
class PendingAnalysis:
    """Curation analysis for one pending entry."""

    entry: PendingEntry
    warnings: list[str]
    duplicates: list[DuplicateCandidate]


@dataclass(frozen=True)
class ApprovalResult:
    """Result of approving a pending entry."""

    pending_id: int
    inserted_table: str
    inserted_id: int
    warnings: list[str]
    duplicates: list[DuplicateCandidate]


def curate_pending(db_path: str | Path | None = None, status: str | None = "pending") -> list[PendingAnalysis]:
    """Analyze pending entries without approving or rejecting them."""

    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        entries = db.list_pending_entries(connection, status=status)
        return [
            PendingAnalysis(entry=entry, warnings=warnings, duplicates=duplicates)
            for entry in entries
            for warnings, duplicates in [analyze_pending_entry(connection, entry)]
        ]


def analyze_pending_entry(
    connection,
    entry: PendingEntry,
    duplicate_threshold: float = 0.82,
) -> tuple[list[str], list[DuplicateCandidate]]:
    """Return curation warnings and possible duplicates for a pending entry."""

    warnings = list(entry.warnings)
    warnings.extend(_structural_warnings(entry))
    duplicates = find_duplicates(connection, entry, duplicate_threshold)
    if duplicates:
        warnings.append("possible duplicate detected")
    return _dedupe_strings(warnings), duplicates


def find_duplicates(
    connection,
    entry: PendingEntry,
    duplicate_threshold: float = 0.82,
) -> list[DuplicateCandidate]:
    """Find records with similar titles, statements, names, or aliases."""

    payload = entry.payload
    title = str(payload.get("title") or payload.get("name") or "")
    statement = str(payload.get("statement") or payload.get("description") or payload.get("summary") or "")
    candidates: list[DuplicateCandidate] = []

    if entry.entry_type == "theorem":
        for theorem in db.list_theorems(connection):
            score = _record_similarity(title, statement, theorem.title or "", theorem.statement)
            if score >= duplicate_threshold:
                candidates.append(DuplicateCandidate("theorems", theorem.id or 0, score, theorem.title or "theorem"))
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
    elif entry.entry_type == "concept":
        duplicate = db.find_concept_by_name_or_alias(connection, title)
        if duplicate:
            candidates.append(DuplicateCandidate("concepts", duplicate.concept_id or 0, 1.0, duplicate.name))
        for alias in payload.get("aliases", []):
            duplicate = db.find_concept_by_name_or_alias(connection, str(alias))
            if duplicate:
                candidates.append(DuplicateCandidate("concepts", duplicate.concept_id or 0, 1.0, duplicate.name))
    elif entry.entry_type == "conjecture_seed":
        for conjecture in db.list_conjectures(connection):
            score = _record_similarity(title, statement, conjecture.title or "", conjecture.statement)
            if score >= duplicate_threshold:
                candidates.append(DuplicateCandidate("conjectures", conjecture.id or 0, score, conjecture.title or "conjecture"))

    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)


def approve_pending(entry_id: int, db_path: str | Path | None = None) -> ApprovalResult:
    """Validate a pending entry, insert it into a main table, and mark approved."""

    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        entry = db.get_pending_entry(connection, entry_id)
        if entry is None:
            raise ValueError(f"pending entry {entry_id} does not exist")
        if entry.status not in {"pending", "flagged"}:
            raise ValueError(f"pending entry {entry_id} is already {entry.status}")
        if entry.entry_type == "paper_summary":
            raise ValueError("paper_summary entries are curation notes and are not directly approvable")

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


def flag_pending(entry_id: int, reason: str, db_path: str | Path | None = None) -> None:
    """Mark a pending entry as flagged for later review."""

    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        entry = db.get_pending_entry(connection, entry_id)
        if entry is None:
            raise ValueError(f"pending entry {entry_id} does not exist")
        warnings = _dedupe_strings([*entry.warnings, f"flagged: {reason}"])
        db.update_pending_status(connection, entry_id, "flagged", warnings=warnings)


def _insert_validated_payload(connection, entry: PendingEntry) -> tuple[str, int]:
    schema = SCHEMA_BY_ENTRY_TYPE.get(entry.entry_type)
    if schema is None:
        raise ValueError(f"pending entry type {entry.entry_type} is not directly approvable")
    try:
        validated = schema.model_validate(entry.payload)
    except ValidationError as exc:
        raise ValueError(f"pending entry {entry.id} failed validation: {exc}") from exc

    if isinstance(validated, Concept):
        return "concepts", db.insert_concept(connection, validated)
    if isinstance(validated, Model):
        return "models", db.insert_model(connection, validated)
    if isinstance(validated, Theorem):
        return "theorems", db.insert_theorem(connection, validated)
    if isinstance(validated, Reduction):
        return "reductions", db.insert_reduction(connection, validated)
    if isinstance(validated, OpenProblem):
        return "open_problems", db.insert_open_problem(connection, validated)
    if isinstance(validated, Conjecture):
        return "conjectures", db.insert_conjecture(connection, validated)
    raise TypeError(f"unsupported pending entry type: {entry.entry_type}")


def _structural_warnings(entry: PendingEntry) -> list[str]:
    warnings: list[str] = []
    if entry.entry_type in {"theorem", "reduction", "open_problem"} and not entry.payload.get("source_location"):
        warnings.append("missing source location")
    if entry.entry_type in {"theorem", "reduction"}:
        warnings.extend(_missing_assumption_warnings(entry))
    if entry.entry_type == "theorem":
        warnings.extend(_theorem_warnings(entry.payload))
    return warnings


def _missing_assumption_warnings(entry: PendingEntry) -> list[str]:
    assumption_fields = (
        "assumptions",
        "architecture_assumptions",
        "information_assumptions",
        "strategy_assumptions",
    )
    if not any(entry.payload.get(field) for field in assumption_fields):
        return ["missing assumptions"]
    return []


def _theorem_warnings(payload: dict) -> list[str]:
    warnings: list[str] = []
    statement = str(payload.get("statement", ""))
    lowered = statement.lower()
    has_complexity = any(payload.get(field) for field in ("complexity_upper", "complexity_lower"))
    if has_complexity and not payload.get("model_family"):
        warnings.append("theorem has complexity bound but no model family")
    if has_complexity and not payload.get("objective_family"):
        warnings.append("theorem has complexity bound but no objective family")
    if "decidable" in lowered or payload.get("theorem_type") in {"decidability", "undecidability"}:
        if not payload.get("architecture_assumptions"):
            warnings.append("decidability claim lacks architecture assumptions")
        if not payload.get("information_assumptions"):
            warnings.append("decidability claim lacks information assumptions")
    if any(term in lowered for term in ("efficient", "tractable", "hard")):
        if not (payload.get("complexity_upper") or payload.get("complexity_lower")):
            warnings.append("vague complexity term without complexity class")
    return warnings


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

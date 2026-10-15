"""Small transactional workflows for research activities and their outputs."""

from __future__ import annotations

from pathlib import Path

from . import db
from .schemas import DerivedResult, Paper, ResearchUnit, ResearchUnitLink


def record_partial_result(
    unit_id: int,
    title: str,
    statement: str,
    *,
    notes: str | None = None,
    followup_title: str | None = None,
    followup_purpose: str | None = None,
    followup_kind: str = "investigation",
    db_path: str | Path | None = None,
) -> tuple[int, int | None]:
    """Store a draft finding and optionally branch to a proposed follow-up unit."""

    if (followup_title is None) != (followup_purpose is None):
        raise ValueError("follow-up title and purpose must be supplied together")
    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        unit = db.get_research_unit(connection, unit_id)
        if unit is None:
            raise ValueError(f"research unit {unit_id} does not exist")
        result_id = db.insert_derived_result(
            connection,
            DerivedResult(
                title=title,
                statement=statement,
                status="draft",
                task_id=unit.task_id,
                notes=notes,
            ),
        )
        db.link_research_unit(
            connection,
            ResearchUnitLink(
                unit_id=unit_id,
                relation="produces",
                object_type="derived_result",
                object_id=result_id,
            ),
        )
        followup_id = None
        if followup_title is not None and followup_purpose is not None:
            followup_id = db.insert_research_unit(
                connection,
                ResearchUnit(
                    task_id=unit.task_id,
                    parent_unit_id=unit_id,
                    kind=followup_kind,
                    title=followup_title,
                    purpose=followup_purpose,
                ),
            )
            db.link_research_unit(
                connection,
                ResearchUnitLink(
                    unit_id=followup_id,
                    relation="uses",
                    object_type="derived_result",
                    object_id=result_id,
                ),
            )
        return result_id, followup_id


def add_paper_for_unit(
    unit_id: int,
    *,
    title: str,
    authors: list[str],
    year: int,
    venue: str | None = None,
    pdf_path: str | None = None,
    url: str | None = None,
    notes: str | None = None,
    db_path: str | Path | None = None,
) -> int:
    """Register a source found during a literature activity and link its provenance."""

    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        unit = db.get_research_unit(connection, unit_id)
        if unit is None:
            raise ValueError(f"research unit {unit_id} does not exist")
        paper_id = db.insert_paper(
            connection,
            Paper(
                title=title,
                authors=authors,
                year=year,
                venue=venue,
                pdf_path=pdf_path,
                url=url,
                notes=notes,
                task_id=unit.task_id,
            ),
        )
        db.link_research_unit(
            connection,
            ResearchUnitLink(
                unit_id=unit_id,
                relation="produces",
                object_type="paper",
                object_id=paper_id,
            ),
        )
        return paper_id

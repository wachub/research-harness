"""Manual ingestion helpers for literature metadata."""

from __future__ import annotations

from pathlib import Path

from . import db
from .schemas import Paper


def add_paper(
    title: str,
    authors: list[str],
    year: int,
    venue: str | None = None,
    pdf_path: str | None = None,
    notes: str | None = None,
    cluster_id: int | None = None,
    db_path: str | Path | None = None,
) -> int:
    """Validate and insert a paper record."""

    paper = Paper(
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        pdf_path=pdf_path,
        notes=notes,
        cluster_id=cluster_id,
    )
    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        return db.insert_paper(connection, paper)

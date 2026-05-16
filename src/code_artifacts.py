"""Helpers for registering reusable code artifacts stored in Git/filesystem."""

from __future__ import annotations

from pathlib import Path

from . import db
from .git_utils import get_current_commit_hash
from .schemas import CodeArtifact


def register_code_artifact(
    name: str,
    path: str,
    artifact_type: str,
    description: str | None = None,
    related_concepts: list[int | str] | None = None,
    related_conjectures: list[int] | None = None,
    tests_path: str | None = None,
    status: str = "draft",
    notes: str | None = None,
    db_path: str | Path | None = None,
) -> int:
    """Register a code artifact and record the current Git commit hash if available."""

    artifact = CodeArtifact(
        name=name,
        path=path,
        artifact_type=artifact_type,
        description=description,
        related_concepts=related_concepts or [],
        related_conjectures=related_conjectures or [],
        tests_path=tests_path,
        status=status,
        git_commit_hash=get_current_commit_hash(),
        notes=notes,
    )
    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        return db.insert_code_artifact(connection, artifact)


def list_code_artifacts(
    artifact_type: str | None = None,
    status: str | None = None,
    db_path: str | Path | None = None,
) -> list[CodeArtifact]:
    """List registered code artifacts."""

    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        return db.list_code_artifacts(connection, artifact_type=artifact_type, status=status)


def get_code_artifact(artifact_id: int, db_path: str | Path | None = None) -> CodeArtifact | None:
    """Return one code artifact by id."""

    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        return db.get_code_artifact(connection, artifact_id)


def update_code_artifact_status(
    artifact_id: int,
    status: str,
    db_path: str | Path | None = None,
) -> None:
    """Update an artifact status."""

    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        db.update_code_artifact_status(connection, artifact_id, status)


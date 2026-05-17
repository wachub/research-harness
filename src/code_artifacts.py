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
    entrypoint: str | None = None,
    language: str | None = None,
    description: str | None = None,
    cluster_id: int | None = None,
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
        entrypoint=entrypoint,
        language=language,
        description=description,
        cluster_id=cluster_id,
        related_concepts=related_concepts or [],
        related_conjectures=related_conjectures or [],
        tests_path=tests_path,
        status=status,
        git_commit_hash=get_current_commit_hash(),
        notes=notes,
    )
    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        for existing in db.list_code_artifacts(connection):
            if existing.name == artifact.name and existing.path == artifact.path:
                connection.execute(
                    """
                    UPDATE code_artifacts
                    SET artifact_type = ?,
                        entrypoint = COALESCE(?, entrypoint),
                        language = COALESCE(?, language),
                        description = COALESCE(?, description),
                        cluster_id = COALESCE(?, cluster_id),
                        related_concepts = ?,
                        related_conjectures = ?,
                        tests_path = COALESCE(?, tests_path),
                        status = ?,
                        git_commit_hash = COALESCE(?, git_commit_hash),
                        notes = COALESCE(?, notes)
                    WHERE artifact_id = ?
                    """,
                    (
                        artifact.artifact_type,
                        artifact.entrypoint,
                        artifact.language,
                        artifact.description,
                        artifact.cluster_id,
                        db._json_dumps(artifact.related_concepts),
                        db._json_dumps(artifact.related_conjectures),
                        artifact.tests_path,
                        artifact.status,
                        artifact.git_commit_hash,
                        artifact.notes,
                        existing.artifact_id,
                    ),
                )
                return existing.artifact_id or 0
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

"""Experiment execution and metadata recording."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import db
from .git_utils import get_current_commit_hash
from .schemas import ExperimentRun


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass(frozen=True)
class ExperimentExecution:
    """Return value for a completed experiment run."""

    run_id: int
    returncode: int
    result_file: Path
    result_summary: str


def run_experiment(
    artifact_id: int,
    command: str,
    input_path: str | None = None,
    output_path: str | None = None,
    cluster_id: int | None = None,
    conjecture_id: int | None = None,
    experiment_type: str | None = None,
    notes: str | None = None,
    db_path: str | Path | None = None,
    results_dir: str | Path | None = None,
) -> ExperimentExecution:
    """Execute a command, save stdout/stderr, and store experiment metadata."""

    git_commit_hash = get_current_commit_hash()
    result_path = _next_result_path(results_dir)
    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        if db.get_code_artifact(connection, artifact_id) is None:
            raise ValueError(f"code artifact {artifact_id} does not exist")

    completed = subprocess.run(
        command,
        shell=True,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    _write_result_file(result_path, command, completed)

    summary = "success" if completed.returncode == 0 else f"failure returncode={completed.returncode}"
    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        run_id = db.insert_experiment_run(
            connection,
            ExperimentRun(
                artifact_id=artifact_id,
                cluster_id=cluster_id,
                conjecture_id=conjecture_id,
                experiment_type=experiment_type or "manual",
                input_path=input_path,
                output_path=output_path or str(result_path),
                output_json={
                    "returncode": completed.returncode,
                    "stdout_path": str(result_path),
                },
                result_summary=summary,
                command_run=command,
                git_commit_hash=git_commit_hash,
                notes=notes,
            ),
        )
    return ExperimentExecution(run_id, completed.returncode, result_path, summary)


def _next_result_path(results_dir: str | Path | None = None) -> Path:
    directory = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return directory / f"experiment_{timestamp}.txt"


def _write_result_file(
    path: Path,
    command: str,
    completed: subprocess.CompletedProcess[str],
) -> None:
    content = (
        f"command: {command}\n"
        f"returncode: {completed.returncode}\n"
        "\n[stdout]\n"
        f"{completed.stdout}"
        "\n[stderr]\n"
        f"{completed.stderr}"
    )
    path.write_text(content, encoding="utf-8")

import sys
from pathlib import Path

from src import db
from src.code_artifacts import (
    get_code_artifact,
    list_code_artifacts,
    register_code_artifact,
    update_code_artifact_status,
)
from src.experiment_manager import run_experiment
from src.git_utils import get_current_commit_hash, get_working_tree_status, is_working_tree_clean


def test_registering_and_listing_code_artifacts(tmp_path):
    db_path = tmp_path / "research.db"
    db.initialize_database(db_path)

    artifact_id = register_code_artifact(
        name="Tiny ATS brute checker",
        path="src/experiments/ats_brute_solver.py",
        artifact_type="checker",
        description="Bounded brute-force checker for tiny safety games.",
        related_concepts=["ATS games", "safety objective"],
        related_conjectures=[],
        tests_path="tests/test_brute_solver.py",
        db_path=db_path,
    )

    artifacts = list_code_artifacts(db_path=db_path)
    artifact = get_code_artifact(artifact_id, db_path=db_path)

    assert len(artifacts) == 1
    assert artifact is not None
    assert artifact.name == "Tiny ATS brute checker"
    assert artifact.path == "src/experiments/ats_brute_solver.py"
    assert artifact.git_commit_hash is None or len(artifact.git_commit_hash) >= 7

    update_code_artifact_status(artifact_id, "tested", db_path=db_path)
    updated = get_code_artifact(artifact_id, db_path=db_path)

    assert updated is not None
    assert updated.status == "tested"


def test_git_utility_functions_do_not_crash():
    commit = get_current_commit_hash()
    status = get_working_tree_status()
    clean = is_working_tree_clean()

    assert commit is None or isinstance(commit, str)
    assert isinstance(status, str)
    assert isinstance(clean, bool)


def test_experiment_run_records_output_file(tmp_path):
    db_path = tmp_path / "research.db"
    results_dir = tmp_path / "results"
    db.initialize_database(db_path)
    artifact_id = register_code_artifact(
        name="Hello experiment script",
        path="experiments/hello.py",
        artifact_type="experiment_script",
        db_path=db_path,
    )

    command = f'"{sys.executable}" -c "print(\'hello experiment\')"'
    execution = run_experiment(
        artifact_id=artifact_id,
        command=command,
        experiment_type="smoke",
        db_path=db_path,
        results_dir=results_dir,
    )

    result_text = Path(execution.result_file).read_text(encoding="utf-8")
    with db.get_connection(db_path) as connection:
        runs = db.list_experiment_runs(connection)
        run = db.get_experiment_run(connection, execution.run_id)

    assert execution.returncode == 0
    assert "hello experiment" in result_text
    assert len(runs) == 1
    assert run is not None
    assert run.artifact_id == artifact_id
    assert run.result_summary == "success"
    assert run.command_run == command
    assert run.output_json["returncode"] == 0

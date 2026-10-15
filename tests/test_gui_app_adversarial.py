from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src import dashboard, db
from src.schemas import (CodeArtifact, Conjecture, EvidenceSpan, ExperimentRun,
                         OpenProblem, Paper, ResearchTask, ResearchUnit,
                         ResearchUnitLink, Theorem)


APP_PATH = Path(__file__).parents[1] / "src" / "gui" / "app.py"


def _task(db_path: Path, name: str = "GUI audit") -> int:
    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        return db.insert_task(connection, ResearchTask(name=name, description="<b>hostile</b> $(echo no)"))


def _app(monkeypatch, db_path: Path) -> AppTest:
    test_database_url = db.resolve_database_url()
    monkeypatch.setenv("TEST_DATABASE_URL", test_database_url)
    monkeypatch.setenv("DATABASE_URL", str(db_path))
    app = AppTest.from_file(str(APP_PATH))
    app.run(timeout=5)
    return app


def _switch(app: AppTest, page: str) -> AppTest:
    app.radio[0].set_value(page)
    app.run(timeout=5)
    return app


def _button(app: AppTest, key: str):
    return next(item for item in app.button if item.key == key)


def _text_input(app: AppTest, key: str):
    return next(item for item in app.text_input if item.key == key)


def test_streamlit_pages_and_project_tabs_render_against_populated_database(tmp_path, monkeypatch):
    db_path = tmp_path / "research.db"
    task_id = _task(db_path)
    hostile = "<script>alert(1)</script>\n$(rm -rf /)\n🔬" + "x" * 5_000
    with db.get_connection(db_path) as connection:
        paper_id = db.insert_paper(connection, Paper(title=hostile, authors=["A"], year=2026, task_id=task_id))
        theorem_id = db.insert_theorem(connection, Theorem(title="Known", statement=hostile, task_id=task_id))
        db.insert_conjecture(connection, Conjecture(statement=hostile, task_id=task_id))
        db.insert_open_problem(connection, OpenProblem(title="Open", statement=hostile, task_id=task_id))
        db.insert_evidence_span(connection, EvidenceSpan(paper_id=paper_id, entry_type="theorem", entry_id=theorem_id, quote_or_summary=hostile))
        db.insert_code_artifact(connection, CodeArtifact(name="artifact", path="../../evil.py", artifact_type="checker", task_id=task_id, status="tested"))
        db.insert_experiment_run(connection, ExperimentRun(task_id=task_id, experiment_type="manual", result_summary=hostile, output_json={"hostile": hostile}))
        connection.execute("INSERT INTO pending_entries (entry_type, payload_json, source_text, warnings_json) VALUES (?, ?, ?, ?)",
                           ("conjecture_seed", '{"statement":"Candidate"}', hostile, "[]"))

    monkeypatch.setenv("LLM_API_KEY", "sk-gui-audit-secret")
    app = _app(monkeypatch, db_path)
    assert not app.exception
    for page in ("Dashboard", "Research Tasks", "Database Explorer", "Experiments", "System Status"):
        _switch(app, page)
        assert not app.exception, page
    _switch(app, "Research Tasks")
    assert not app.tabs
    assert _button(app, "open_task_records")
    _switch(app, "System Status")
    assert all("sk-gui-audit-secret" not in str(item.value) for item in app.json)


def test_research_task_gui_shows_units_and_only_two_interventions(tmp_path, monkeypatch):
    db_path = tmp_path / "research.db"
    task_id = _task(db_path)
    with db.get_connection(db_path) as connection:
        unit_id = db.insert_research_unit(
            connection,
            ResearchUnit(
                task_id=task_id, kind="hypothesis", title="Check finite-memory case",
                purpose="Investigate a bounded case", status="active",
            ),
        )
    calls = []

    def run_steps(task_id, steps, focus, literature, path, **_kwargs):
        calls.append((task_id, steps, focus, literature, path))
        return {
            "status": "completed", "message": "One step completed.",
            "research_unit_ids": [unit_id + 1], "steps_completed": 1,
        }

    monkeypatch.setattr(dashboard, "run_research_steps", run_steps)
    app = _app(monkeypatch, db_path)
    assert app.radio[0].value == "Research Tasks"
    next(item for item in app.selectbox if item.key == "project_selector").set_value(task_id)
    app.run(timeout=5)
    assert not app.exception
    assert any("Check finite-memory case" in str(item.value) for item in app.dataframe)
    assert not any(item.key and item.key.startswith("unit_create") for item in app.button)

    _button(app, f"unit_focus_{task_id}_{unit_id}").click().run(timeout=5)
    assert next(item for item in app.selectbox if item.key == "unit_work_selector").value == unit_id
    assert calls == []
    _button(app, "continue_research").click()
    app.run(timeout=5)
    assert calls[-1] == (task_id, 1, unit_id, False, str(db_path))

    _button(app, "assess_literature").click()
    app.run(timeout=5)
    assert calls[-1] == (task_id, 1, unit_id, True, str(db_path))
    assert any("One step completed." in str(item.value) for item in app.markdown)
    with db.get_connection(db_path) as connection:
        assert db.get_task(connection, task_id).name == "GUI audit"
        assert db.get_research_unit(connection, unit_id).status == "active"


def test_literature_intervention_requires_selected_unit(tmp_path, monkeypatch):
    db_path = tmp_path / "research.db"
    task_id = _task(db_path)
    calls = []
    monkeypatch.setattr(dashboard, "run_research_steps", lambda *args: calls.append(args))
    app = _app(monkeypatch, db_path)
    next(item for item in app.selectbox if item.key == "project_selector").set_value(task_id)
    app.run(timeout=5)
    assert not app.exception
    assert _button(app, "assess_literature").disabled
    assert calls == []
    with db.get_connection(db_path) as connection:
        assert db.list_research_units(connection, task_id) == []


def test_review_controls_are_absent_even_with_legacy_pending_data(tmp_path, monkeypatch):
    db_path = tmp_path / "research.db"
    task_id = _task(db_path)
    with db.get_connection(db_path) as connection:
        entry_id = connection.execute(
            "INSERT INTO pending_entries (entry_type, payload_json, warnings_json) VALUES (?, ?, ?) RETURNING id",
            ("theorem", '{"statement":"Legacy candidate"}', "[]"),
        ).fetchone()["id"]
    app = _app(monkeypatch, db_path)
    assert "Review Queue" not in app.radio[0].options
    for page in app.radio[0].options:
        _switch(app, page)
        assert not any(item.key and item.key.startswith(("approve_", "reject_", "flag_")) for item in app.button)
    with db.get_connection(db_path) as connection:
        assert db.get_pending_entry(connection, entry_id).status == "pending"


def test_experiment_output_never_reads_project_secret_or_external_path(tmp_path):
    db_path = tmp_path / "research.db"
    task_id = _task(db_path)
    secret_path = db.PROJECT_ROOT / ".gui_audit_secret"
    secret_path.write_text("LLM_API_KEY=gui-audit-secret", encoding="utf-8")
    try:
        with db.get_connection(db_path) as connection:
            run_id = db.insert_experiment_run(connection, ExperimentRun(task_id=task_id, experiment_type="manual", output_path=str(secret_path)))
        displayed = dashboard.read_experiment_output(run_id, db_path)
    finally:
        secret_path.unlink(missing_ok=True)
    assert "gui-audit-secret" not in displayed
    assert "permitted results directory" in displayed


def test_experiments_page_renders_only_permitted_recorded_output(tmp_path, monkeypatch):
    db_path = tmp_path / "research.db"
    task_id = _task(db_path)
    output_path = db.PROJECT_ROOT / "results" / "gui_audit_output.txt"
    output_path.write_text("observational output only", encoding="utf-8")
    try:
        with db.get_connection(db_path) as connection:
            run_id = db.insert_experiment_run(
                connection,
                ExperimentRun(task_id=task_id, experiment_type="manual", output_path=str(output_path)),
            )
        app = _switch(_app(monkeypatch, db_path), "Experiments")
        checkbox = next(item for item in app.checkbox if item.key == f"show_output_{run_id}")
        checkbox.set_value(True)
        app.run(timeout=5)
        assert not app.exception
        assert any("observational output only" in str(item.value) for item in app.code)
    finally:
        output_path.unlink(missing_ok=True)


def test_streamlit_empty_research_state_navigates_all_pages(tmp_path, monkeypatch):
    db_path = tmp_path / "empty.db"
    db.initialize_database(db_path)
    app = _app(monkeypatch, db_path)
    for page in ("Dashboard", "Research Tasks", "Database Explorer", "Experiments", "System Status"):
        _switch(app, page)
        assert not app.exception, page


def test_database_explorer_is_read_only_under_filtering_and_selection(tmp_path, monkeypatch):
    db_path = tmp_path / "research.db"
    task_id = _task(db_path)
    with db.get_connection(db_path) as connection:
        connection.execute("INSERT INTO pending_entries (entry_type, payload_json, warnings_json) VALUES (?, ?, ?)",
                           ("conjecture_seed", '{"statement":"$(no write)"}', "[]"))
        before = {
            "pending": len(db.list_pending_entries(connection)),
            "conjectures": len(db.list_conjectures(connection)),
            "runs": len(db.list_experiment_runs(connection)),
        }
    app = _switch(_app(monkeypatch, db_path), "Database Explorer")
    app.segmented_control[0].set_value("Tables")
    app.run(timeout=5)
    next(item for item in app.selectbox if item.key == "explorer_table").set_value("pending_entries")
    app.run(timeout=5)
    next(item for item in app.text_input if item.key == "explorer_filter").set_value("$(no write)")
    app.run(timeout=5)
    assert not app.exception
    with db.get_connection(db_path) as connection:
        assert before == {
            "pending": len(db.list_pending_entries(connection)),
            "conjectures": len(db.list_conjectures(connection)),
            "runs": len(db.list_experiment_runs(connection)),
        }


def test_workspace_can_hide_and_reopen_without_losing_page(tmp_path, monkeypatch):
    db_path = tmp_path / "research.db"
    _task(db_path)
    app = _app(monkeypatch, db_path)
    _button(app, "hide_workspace").click().run(timeout=5)
    assert not app.exception
    assert not app.radio
    assert _button(app, "show_workspace")
    assert any(item.value == "Research tasks" for item in app.header)
    _button(app, "show_workspace").click().run(timeout=5)
    assert not app.exception
    assert app.radio[0].value == "Research Tasks"


def test_unit_and_record_navigation_exposes_linked_result_and_code(tmp_path, monkeypatch):
    db_path = tmp_path / "research.db"
    task_id = _task(db_path)
    with db.get_connection(db_path) as connection:
        unit_id = db.insert_research_unit(connection, ResearchUnit(
            task_id=task_id, kind="proof_attempt", title="Inspect reduction",
            purpose="Compare known bounds", status="finished",
        ))
        theorem_id = db.insert_theorem(connection, Theorem(
            title="Known bound", statement="Finite-state bound", task_id=task_id,
        ))
        artifact_id = db.insert_code_artifact(connection, CodeArtifact(
            name="bounded checker", path="src/brute_solver.py", artifact_type="checker", task_id=task_id,
        ))
        db.link_research_unit(connection, ResearchUnitLink(
            unit_id=unit_id, relation="uses", object_type="theorem", object_id=theorem_id,
        ))
        db.link_research_unit(connection, ResearchUnitLink(
            unit_id=unit_id, relation="produces", object_type="code_artifact", object_id=artifact_id,
        ))
    app = _app(monkeypatch, db_path)
    assert not app.exception
    assert any("Known bound" in str(item.value) and "bounded checker" in str(item.value)
               for item in app.dataframe)
    _switch(app, "Database Explorer")
    assert not app.exception
    assert any("Known bound" in str(item.value) for item in app.dataframe)
    app.segmented_control[0].set_value("Tables")
    app.run(timeout=5)
    next(item for item in app.selectbox if item.key == "explorer_table").set_value("theorems")
    app.run(timeout=5)
    assert not app.exception
    linked_unit = next(item for item in app.button
                       if item.key and item.key.startswith(f"explorer_object_units_theorems_{theorem_id}_"))
    linked_unit.click().run(timeout=5)
    assert not app.exception
    assert app.radio[0].value == "Research Tasks"
    assert next(item for item in app.selectbox if item.key == f"inspect_unit_{task_id}").value == unit_id


def test_task_record_shortcut_opens_first_result_and_raw_table(tmp_path, monkeypatch):
    db_path = tmp_path / "research.db"
    task_id = _task(db_path)
    with db.get_connection(db_path) as connection:
        db.insert_theorem(connection, Theorem(
            title="Inspectable result", statement="A recorded theorem", task_id=task_id,
        ))
    app = _app(monkeypatch, db_path)
    _button(app, "open_task_records").click().run(timeout=5)
    assert not app.exception
    assert app.radio[0].value == "Database Explorer"
    assert next(item for item in app.selectbox if item.key == "explorer_task_selector").value == task_id
    assert any("A recorded theorem" in item.value for item in app.markdown)
    _button(app, f"task_open_record_{task_id}_theorems").click().run(timeout=5)
    assert not app.exception
    assert next(item for item in app.selectbox if item.key == "explorer_table").value == "theorems"


@pytest.mark.parametrize("bad_url", ["mysql://localhost/not-supported", "postgresql://localhost:1/unreachable"])
def test_gui_shows_clear_error_for_invalid_database_urls(tmp_path, monkeypatch, bad_url):
    app = _app(monkeypatch, tmp_path / "healthy-test-schema")
    app.session_state["database_url"] = bad_url
    _switch(app, "Research Tasks")
    assert not app.exception
    assert app.error


def test_failed_research_run_shows_validation_details_and_log_id(tmp_path, monkeypatch):
    db_path = tmp_path / "research.db"
    _task(db_path)

    def failed_run(*_args, **_kwargs):
        return {
            "status": "stopped", "message": "LLM provider JSON failed validation: kind (missing)",
            "research_unit_ids": [], "steps_completed": 0,
            "error_type": "LLMError",
            "error_details": [{"field": "kind", "issue": "missing"}],
            "diagnostic_id": "abc123",
            "diagnostics": {"category": "quota_exhausted", "http_status": 429,
                            "hint": "Check API credits and project limits.",
                            "model": "test-model", "stage": "select_unit",
                            "literature_trace": [{"stage": "download_pdf", "status": "failed",
                                                  "host": "example.org", "http_status": 403,
                                                  "elapsed_seconds": 0.1}]},
        }

    monkeypatch.setattr(dashboard, "run_research_steps", failed_run)
    app = _app(monkeypatch, db_path)
    _button(app, "continue_research").click()
    app.run(timeout=5)

    assert not app.exception
    assert app.error
    assert any("abc123" in str(item.value) for item in app.caption)
    assert any("kind" in str(item.value) for item in app.json)
    assert any("quota_exhausted" in str(item.value) for item in app.json)
    assert any("Check API credits" in str(item.value) for item in app.info)
    assert any("Literature stages" in str(item.value) for item in app.caption)
    assert any("stage" in item.value.columns for item in app.dataframe)

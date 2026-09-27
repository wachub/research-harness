from __future__ import annotations

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src import dashboard, db
from src.llm import LLMClient, LLMResponse
from src.research_controller import ResearchController
from src.research_policy import ControllerMode
from src.schemas import CodeArtifact, Conjecture, EvidenceSpan, ExperimentRun, OpenProblem, Paper, PendingEntry, ResearchCluster, Theorem


APP_PATH = Path(__file__).parents[1] / "src" / "gui" / "app.py"


class SequenceProvider:
    provider_name = "gui-audit"
    model = "gui-audit-model"

    def __init__(self, responses):
        self.responses = list(responses)

    def complete(self, request):
        response = self.responses.pop(0)
        return LLMResponse(content=json.dumps(response), provider=self.provider_name, model=self.model)


def _cluster(db_path: Path, name: str = "GUI audit") -> int:
    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        return db.insert_cluster(connection, ResearchCluster(name=name, description="<b>hostile</b> $(echo no)", priority=4))


def _app(monkeypatch, db_path: Path) -> AppTest:
    monkeypatch.setattr(db, "DEFAULT_DATABASE_URL", str(db_path))
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


def _controller_action(action_type: str, parameters: dict) -> dict:
    return {
        "action": {
            "action_type": action_type,
            "reason": "Reviewable GUI audit action.",
            "expected_effect": "Produce only bounded or provisional output.",
            "references": [],
            "parameters": parameters,
        }
    }


def test_streamlit_pages_and_project_tabs_render_against_populated_database(tmp_path, monkeypatch):
    db_path = tmp_path / "research.db"
    cluster_id = _cluster(db_path)
    hostile = "<script>alert(1)</script>\n$(rm -rf /)\n🔬" + "x" * 5_000
    with db.get_connection(db_path) as connection:
        paper_id = db.insert_paper(connection, Paper(title=hostile, authors=["A"], year=2026, cluster_id=cluster_id))
        theorem_id = db.insert_theorem(connection, Theorem(title="Known", statement=hostile, cluster_id=cluster_id))
        db.insert_conjecture(connection, Conjecture(statement=hostile, cluster_id=cluster_id))
        db.insert_open_problem(connection, OpenProblem(title="Open", statement=hostile, cluster_id=cluster_id))
        db.insert_evidence_span(connection, EvidenceSpan(paper_id=paper_id, entry_type="theorem", entry_id=theorem_id, quote_or_summary=hostile))
        db.insert_code_artifact(connection, CodeArtifact(name="artifact", path="../../evil.py", artifact_type="checker", cluster_id=cluster_id, status="tested"))
        db.insert_experiment_run(connection, ExperimentRun(cluster_id=cluster_id, experiment_type="manual", result_summary=hostile, output_json={"hostile": hostile}))
        db.insert_pending_entry(connection, PendingEntry(entry_type="conjecture_seed", payload=Conjecture(statement="Candidate", cluster_id=cluster_id).model_dump(), source_text=hostile))

    monkeypatch.setenv("LLM_API_KEY", "sk-gui-audit-secret")
    app = _app(monkeypatch, db_path)
    assert not app.exception
    for page in ("Dashboard", "Projects", "Review Queue", "Database Explorer", "Experiments", "System Status"):
        _switch(app, page)
        assert not app.exception, page
    _switch(app, "Projects")
    assert {tab.label for tab in app.tabs} == {
        "Overview", "Known results", "Research frontier", "Literature/evidence", "Experiments", "Pending proposals", "Timeline"
    }
    _switch(app, "System Status")
    assert all("sk-gui-audit-secret" not in str(item.value) for item in app.json)


def test_review_queue_buttons_follow_existing_curation_and_are_not_repeated(tmp_path, monkeypatch):
    db_path = tmp_path / "research.db"
    cluster_id = _cluster(db_path)
    with db.get_connection(db_path) as connection:
        approve_id = db.insert_pending_entry(connection, PendingEntry(entry_type="open_problem", payload=OpenProblem(title="Approve", statement="Approve", cluster_id=cluster_id).model_dump()))
        reject_id = db.insert_pending_entry(connection, PendingEntry(entry_type="conjecture_seed", payload=Conjecture(statement="Reject", cluster_id=cluster_id).model_dump()))
        flag_id = db.insert_pending_entry(connection, PendingEntry(entry_type="conjecture_seed", payload=Conjecture(statement="Flag", cluster_id=cluster_id).model_dump()))

    app = _switch(_app(monkeypatch, db_path), "Review Queue")
    _button(app, f"approve_{approve_id}").click()
    app.run(timeout=5)
    with db.get_connection(db_path) as connection:
        assert db.get_pending_entry(connection, approve_id).status == "approved"
        assert len(db.list_open_problems(connection, cluster_id=cluster_id)) == 1
    assert not any(item.key == f"approve_{approve_id}" for item in app.button)

    _button(app, f"reject_{reject_id}").click()
    app.run(timeout=5)
    _button(app, f"flag_{flag_id}").click()
    app.run(timeout=5)
    assert app.warning
    with db.get_connection(db_path) as connection:
        assert db.get_pending_entry(connection, flag_id).status == "pending"
    _text_input(app, f"review_reason_{flag_id}").set_value("needs a source")
    _button(app, f"flag_{flag_id}").click()
    app.run(timeout=5)
    with db.get_connection(db_path) as connection:
        assert db.get_pending_entry(connection, reject_id).status == "rejected"
        assert db.get_pending_entry(connection, flag_id).status == "flagged"
        assert db.list_conjectures(connection, cluster_id=cluster_id) == []


def test_review_queue_rejects_malformed_payload_without_partial_durable_write(tmp_path, monkeypatch):
    db_path = tmp_path / "research.db"
    _cluster(db_path)
    with db.get_connection(db_path) as connection:
        entry_id = db.insert_pending_entry(connection, PendingEntry(entry_type="theorem", payload={}))
    app = _switch(_app(monkeypatch, db_path), "Review Queue")
    _button(app, f"approve_{entry_id}").click()
    app.run(timeout=5)
    assert not app.exception
    assert app.error
    with db.get_connection(db_path) as connection:
        assert db.get_pending_entry(connection, entry_id).status == "pending"
        assert db.list_theorems(connection) == []


def test_experiment_output_never_reads_project_secret_or_external_path(tmp_path):
    db_path = tmp_path / "research.db"
    cluster_id = _cluster(db_path)
    secret_path = db.PROJECT_ROOT / ".gui_audit_secret"
    secret_path.write_text("LLM_API_KEY=gui-audit-secret", encoding="utf-8")
    try:
        with db.get_connection(db_path) as connection:
            run_id = db.insert_experiment_run(connection, ExperimentRun(cluster_id=cluster_id, experiment_type="manual", output_path=str(secret_path)))
        displayed = dashboard.read_experiment_output(run_id, db_path)
    finally:
        secret_path.unlink(missing_ok=True)
    assert "gui-audit-secret" not in displayed
    assert "permitted results directory" in displayed


def test_experiments_page_renders_only_permitted_recorded_output(tmp_path, monkeypatch):
    db_path = tmp_path / "research.db"
    cluster_id = _cluster(db_path)
    output_path = db.PROJECT_ROOT / "results" / "gui_audit_output.txt"
    output_path.write_text("observational output only", encoding="utf-8")
    try:
        with db.get_connection(db_path) as connection:
            run_id = db.insert_experiment_run(
                connection,
                ExperimentRun(cluster_id=cluster_id, experiment_type="manual", output_path=str(output_path)),
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
    for page in ("Dashboard", "Projects", "Review Queue", "Database Explorer", "Experiments", "System Status"):
        _switch(app, page)
        assert not app.exception, page


def test_database_explorer_is_read_only_under_filtering_and_selection(tmp_path, monkeypatch):
    db_path = tmp_path / "research.db"
    cluster_id = _cluster(db_path)
    with db.get_connection(db_path) as connection:
        db.insert_pending_entry(connection, PendingEntry(entry_type="conjecture_seed", payload=Conjecture(statement="$(no write)", cluster_id=cluster_id).model_dump()))
        before = {
            "pending": len(db.list_pending_entries(connection)),
            "conjectures": len(db.list_conjectures(connection)),
            "runs": len(db.list_experiment_runs(connection)),
        }
    app = _switch(_app(monkeypatch, db_path), "Database Explorer")
    app.selectbox[0].set_value("pending_entries")
    next(item for item in app.text_input if item.label == "Filter rows").set_value("$(no write)")
    app.run(timeout=5)
    assert not app.exception
    with db.get_connection(db_path) as connection:
        assert before == {
            "pending": len(db.list_pending_entries(connection)),
            "conjectures": len(db.list_conjectures(connection)),
            "runs": len(db.list_experiment_runs(connection)),
        }


def test_controller_proposals_move_through_gui_review_without_direct_trust_write(tmp_path, monkeypatch):
    db_path = tmp_path / "research.db"
    cluster_id = _cluster(db_path)
    reject = _controller_action("create_pending_conjecture", {"statement": "Reject controller candidate", "rationale": "Reason", "assumptions": [], "uncertainty_note": "Unknown"})
    approve = _controller_action("create_pending_conjecture", {"statement": "Approve controller candidate", "rationale": "Reason", "assumptions": [], "uncertainty_note": "Unknown"})
    stop = _controller_action("stop", {})
    result = ResearchController(LLMClient(provider=SequenceProvider([reject, approve, stop])), db_path=db_path).run(
        "Audit GUI handoff", cluster_id, mode=ControllerMode.AUTONOMOUS
    )
    assert result.status == "stopped"
    with db.get_connection(db_path) as connection:
        entries = db.list_pending_entries(connection)
        assert len(entries) == 2
        assert db.list_conjectures(connection) == []
        entry_ids = {entry.payload["statement"]: entry.id for entry in entries}

    app = _switch(_app(monkeypatch, db_path), "Review Queue")
    _button(app, f"reject_{entry_ids['Reject controller candidate']}").click()
    app.run(timeout=5)
    _button(app, f"approve_{entry_ids['Approve controller candidate']}").click()
    app.run(timeout=5)
    with db.get_connection(db_path) as connection:
        assert db.get_pending_entry(connection, entry_ids["Reject controller candidate"]).status == "rejected"
        assert db.get_pending_entry(connection, entry_ids["Approve controller candidate"]).status == "approved"
        conjectures = db.list_conjectures(connection, cluster_id=cluster_id)
        assert [item.statement for item in conjectures] == ["Approve controller candidate"]


def test_controller_experiment_is_displayed_as_observation_without_claim_promotion(tmp_path, monkeypatch):
    db_path = tmp_path / "research.db"
    cluster_id = _cluster(db_path)
    with db.get_connection(db_path) as connection:
        artifact_id = db.insert_code_artifact(connection, CodeArtifact(name="fixed handler provenance", path="../../not-executed.py", artifact_type="checker", cluster_id=cluster_id, status="tested"))
    run = _controller_action("run_trusted_experiment", {"artifact_id": artifact_id, "kind": "ATS", "process_count": 2, "states_per_process": 2, "depth": 0, "seed": 0})
    stop = _controller_action("stop", {})
    result = ResearchController(LLMClient(provider=SequenceProvider([run, stop])), db_path=db_path).run(
        "Audit experiment handoff", cluster_id, mode=ControllerMode.AUTONOMOUS
    )
    assert result.status == "stopped"
    app = _switch(_app(monkeypatch, db_path), "Experiments")
    assert not app.exception
    assert any("observations, not proofs" in str(item.value) for item in app.caption)
    assert any("trusted_in_process:ats_bounded_safety" in str(item.value) for item in app.json)
    with db.get_connection(db_path) as connection:
        assert len(db.list_experiment_runs(connection, cluster_id=cluster_id)) == 1
        assert db.list_theorems(connection) == []
        assert db.list_conjectures(connection) == []


@pytest.mark.parametrize("bad_url", ["mysql://localhost/not-supported", "postgresql://localhost:1/unreachable"])
def test_gui_shows_clear_error_for_invalid_database_urls(tmp_path, monkeypatch, bad_url):
    app = _app(monkeypatch, tmp_path / "healthy-test-schema")
    next(item for item in app.text_input if item.label == "Database URL").set_value(bad_url)
    app.run(timeout=5)
    assert not app.exception
    assert app.error

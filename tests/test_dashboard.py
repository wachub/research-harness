from __future__ import annotations

from src import dashboard, db
from src.llm import LLMClient
from src.schemas import Conjecture, OpenProblem, Paper, ResearchTask, ResearchUnit
from tests.test_autonomous_research import SequenceProvider, _proposal


def _task(db_path, name="Dashboard task"):
    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        return db.insert_task(connection, ResearchTask(name=name))


def test_dashboard_summary_and_project_counts_are_scoped(tmp_path):
    db_path = tmp_path / "research.db"
    first = _task(db_path, "First")
    second = _task(db_path, "Second")
    with db.get_connection(db_path) as connection:
        db.insert_paper(connection, Paper(title="First paper", authors=["A"], year=2026, task_id=first))
        db.insert_conjecture(connection, Conjecture(statement="First candidate", task_id=first))
        db.insert_open_problem(connection, OpenProblem(title="Second problem", statement="Second", task_id=second))

    summary = dashboard.dashboard_summary(db_path)
    projects = {item["task_id"]: item for item in dashboard.project_summaries(db_path)}
    assert summary["counts"]["papers"] == 1
    assert projects[first]["papers"] == 1
    assert projects[first]["conjectures"] == 1
    assert projects[first]["open_problems"] == 0
    assert projects[second]["open_problems"] == 1


def test_event_history_includes_direct_research_writes(tmp_path):
    db_path = tmp_path / "research.db"
    task_id = _task(db_path)
    with db.get_connection(db_path) as connection:
        db.insert_open_problem(connection, OpenProblem(title="Question", statement="Question", task_id=task_id))
    timeline = dashboard.project_timeline(task_id, db_path)
    event_types = {item["event_type"] for item in timeline}
    assert "open_problem_created" in event_types
    assert timeline == sorted(timeline, key=lambda item: (item["timestamp"] or "", item["object_type"], item["object_id"] or 0))


def test_explorer_tolerates_malformed_optional_json_and_rejects_unknown_tables(tmp_path):
    db_path = tmp_path / "research.db"
    _task(db_path)
    with db.get_connection(db_path) as connection:
        connection.execute("INSERT INTO pending_entries (entry_type, payload_json, warnings_json) VALUES (?, ?, ?)", ("paper_summary", "{}", "not-json"))
    rows = dashboard.explorer_records("pending_entries", db_path)
    assert rows[0]["warnings_json"] == {"_malformed_json": "not-json"}
    try:
        dashboard.explorer_records("sqlite_master", db_path)
    except ValueError as exc:
        assert "unsupported table" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("unknown table was accepted")


def test_system_status_never_surfaces_llm_api_key(tmp_path, monkeypatch):
    db_path = tmp_path / "research.db"
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "safe-model")
    monkeypatch.setenv("LLM_API_KEY", "super-secret-value")
    status = dashboard.system_status(db_path)
    assert status["llm_provider"] == "openai"
    assert status["remote_llm_available"] is True
    assert "super-secret-value" not in str(status)
    assert "api_key" not in status


def test_dashboard_run_can_continue_a_selected_unit_without_approval(tmp_path):
    db_path = tmp_path / "research.db"
    task_id = _task(db_path)
    with db.get_connection(db_path) as connection:
        first_id = db.insert_research_unit(
            connection, ResearchUnit(task_id=task_id, kind="question", title="First", purpose="Explore first", status="active"),
        )
        second_id = db.insert_research_unit(
            connection, ResearchUnit(task_id=task_id, kind="question", title="Second", purpose="Explore second", status="active"),
        )

    output = dashboard.run_research_steps(
        task_id, unit_id=second_id, db_path=db_path,
        client=LLMClient(provider=SequenceProvider([_proposal("analysis", "Follow second")])),
    )
    assert output["status"] == "completed"
    with db.get_connection(db_path) as connection:
        created = db.get_research_unit(connection, output["research_unit_ids"][0])
    assert created.parent_unit_id == second_id != first_id
    assert "pending" not in output

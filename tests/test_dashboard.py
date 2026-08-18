from __future__ import annotations

from src import dashboard, db
from src.schemas import Conjecture, OpenProblem, Paper, PendingEntry, ResearchCluster


def _cluster(db_path, name="Dashboard cluster"):
    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        return db.insert_cluster(connection, ResearchCluster(name=name, status="active"))


def test_dashboard_summary_and_project_counts_are_scoped(tmp_path):
    db_path = tmp_path / "research.db"
    first = _cluster(db_path, "First")
    second = _cluster(db_path, "Second")
    with db.get_connection(db_path) as connection:
        db.insert_paper(connection, Paper(title="First paper", authors=["A"], year=2026, cluster_id=first))
        db.insert_conjecture(connection, Conjecture(statement="First candidate", cluster_id=first))
        db.insert_open_problem(connection, OpenProblem(title="Second problem", statement="Second", cluster_id=second))

    summary = dashboard.dashboard_summary(db_path)
    projects = {item["cluster_id"]: item for item in dashboard.project_summaries(db_path)}
    assert summary["counts"]["papers"] == 1
    assert projects[first]["papers"] == 1
    assert projects[first]["conjectures"] == 1
    assert projects[first]["open_problems"] == 0
    assert projects[second]["open_problems"] == 1


def test_event_history_merges_core_writes_and_pending_review(tmp_path):
    db_path = tmp_path / "research.db"
    cluster_id = _cluster(db_path)
    with db.get_connection(db_path) as connection:
        pending_id = db.insert_pending_entry(
            connection,
            PendingEntry(
                entry_type="open_problem",
                payload=OpenProblem(title="Question", statement="Question", cluster_id=cluster_id).model_dump(),
            ),
        )
    dashboard.approve_review_item(pending_id, db_path)
    timeline = dashboard.project_timeline(cluster_id, db_path)
    event_types = {item["event_type"] for item in timeline}
    assert {"pending_created", "pending_approved", "open_problem_created"} <= event_types
    assert timeline == sorted(timeline, key=lambda item: (item["timestamp"] or "", item["object_type"], item["object_id"] or 0))


def test_review_actions_delegate_to_existing_curation_paths(tmp_path):
    db_path = tmp_path / "research.db"
    cluster_id = _cluster(db_path)
    with db.get_connection(db_path) as connection:
        reject_id = db.insert_pending_entry(
            connection,
            PendingEntry(entry_type="conjecture_seed", payload=Conjecture(statement="Reject me", cluster_id=cluster_id).model_dump()),
        )
        flag_id = db.insert_pending_entry(
            connection,
            PendingEntry(entry_type="conjecture_seed", payload=Conjecture(statement="Flag me", cluster_id=cluster_id).model_dump()),
        )
    dashboard.reject_review_item(reject_id, "duplicate", db_path)
    dashboard.flag_review_item(flag_id, "needs source", db_path)
    with db.get_connection(db_path) as connection:
        assert db.get_pending_entry(connection, reject_id).status == "rejected"
        flagged = db.get_pending_entry(connection, flag_id)
        assert flagged.status == "flagged"
        assert any("needs source" in warning for warning in flagged.warnings)


def test_explorer_tolerates_malformed_optional_json_and_rejects_unknown_tables(tmp_path):
    db_path = tmp_path / "research.db"
    _cluster(db_path)
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

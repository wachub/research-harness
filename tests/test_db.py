from src import db
from src.schemas import Paper


def test_database_inserts_paper(tmp_path):
    db_path = tmp_path / "research.db"
    db.initialize_database(db_path)

    with db.get_connection(db_path) as connection:
        paper_id = db.insert_paper(
            connection,
            Paper(
                title="A Tiny Theory Paper",
                authors=["A. Author"],
                year=2026,
                venue="Workshop",
            ),
        )
        papers = db.list_papers(connection)

    assert paper_id == 1
    assert len(papers) == 1
    assert papers[0].title == "A Tiny Theory Paper"
    assert papers[0].authors == ["A. Author"]


def test_historical_pending_rows_remain_readable_without_approval_api(tmp_path):
    db_path = tmp_path / "research.db"
    db.initialize_database(db_path)

    with db.get_connection(db_path) as connection:
        connection.execute(
            "INSERT INTO pending_entries (entry_type, payload_json, warnings_json) VALUES (?, ?, ?)",
            ("theorem", '{"statement":"Historical candidate"}', "[]"),
        )
        pending = db.list_pending_entries(connection)
        theorems = db.list_theorems(connection)
    assert len(pending) == 1 and pending[0].payload["statement"] == "Historical candidate"
    assert theorems == []
    assert not hasattr(db, "update_pending_status")


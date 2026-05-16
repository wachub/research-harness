from src import db
from src.curate import approve_pending
from src.schemas import Paper, PendingEntry


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


def test_pending_approval_writes_to_theorems(tmp_path):
    db_path = tmp_path / "research.db"
    db.initialize_database(db_path)

    with db.get_connection(db_path) as connection:
        pending_id = db.insert_pending_entry(
            connection,
            PendingEntry(
                entry_type="theorem",
                payload={
                    "title": "Safety theorem",
                    "statement": "All reachable states are safe.",
                    "assumptions": ["finite game"],
                    "conclusion": "safety holds",
                    "paper_id": None,
                    "tags": ["safety"],
                },
            ),
        )

    result = approve_pending(pending_id, db_path=db_path)

    with db.get_connection(db_path) as connection:
        theorems = db.list_theorems(connection)
        pending = db.get_pending_entry(connection, pending_id)

    assert result.inserted_table == "theorems"
    assert len(theorems) == 1
    assert theorems[0].title == "Safety theorem"
    assert pending is not None
    assert pending.status == "approved"


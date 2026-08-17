from pathlib import Path

from src import db
from src.cli import main
from src.literature import (
    generate_verification_tasks,
    quality_check_literature,
    query_literature,
    run_research_demo,
    write_research_memo,
)


def test_literature_tables_are_created(tmp_path):
    db_path = tmp_path / "research.db"
    db.initialize_database(db_path)

    with db.get_connection(db_path) as connection:
        names = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }

    assert "research_topics" in names
    assert "literature_notes" in names
    assert "literature_summaries" in names


def test_research_demo_creates_report_and_database_rows(tmp_path):
    db_path = tmp_path / "research.db"
    report_path = tmp_path / "demo_literature_map.md"

    result = run_research_demo(dry_run=True, db_path=db_path, output_path=report_path)

    with db.get_connection(db_path) as connection:
        topics = db.list_research_topics(connection)
        notes = db.list_literature_notes(connection, topic_id=result.topic_id)
        summaries = db.list_literature_summaries(connection, topic_id=result.topic_id)
        evidence = db.list_evidence_spans(connection, entry_type="literature_note")
        theorems = db.list_theorems(connection)

    assert result.topics_created == 1
    assert result.notes_loaded >= 5
    assert result.summaries_created >= 5
    assert result.evidence_spans_linked >= 5
    assert report_path.exists()
    assert "# Literature Map" in report_path.read_text(encoding="utf-8")
    assert len(topics) >= 1
    assert len(notes) == result.notes_loaded
    assert len(summaries) == result.summaries_created
    assert len(evidence) == result.evidence_spans_linked
    assert theorems == []


def test_query_literature_returns_stored_evidence_answer(tmp_path):
    db_path = tmp_path / "research.db"
    result = run_research_demo(dry_run=True, db_path=db_path, output_path=tmp_path / "map.md")

    answer = query_literature(
        result.topic_id,
        "What is known about global safety in CDM or ATS games?",
        db_path=db_path,
    )

    assert not answer.insufficient
    assert "Global safety CDM games are EXPTIME-complete" in answer.answer
    assert "evidence_id=" in answer.answer
    assert "results/approved/" in answer.answer


def test_query_literature_reports_insufficient_evidence(tmp_path):
    db_path = tmp_path / "research.db"
    result = run_research_demo(dry_run=True, db_path=db_path, output_path=tmp_path / "map.md")

    answer = query_literature(
        result.topic_id,
        "What does this say about quantum cryptography and lattice signatures?",
        db_path=db_path,
    )

    assert answer.insufficient
    assert "insufficient stored evidence" in answer.answer

    protein_answer = query_literature(
        result.topic_id,
        "What does this say about protein folding?",
        db_path=db_path,
    )

    assert protein_answer.insufficient
    assert "insufficient stored evidence" in protein_answer.answer


def test_cli_research_demo_smoke(tmp_path, capsys):
    db_path = tmp_path / "research.db"
    report_path = tmp_path / "demo.md"

    exit_code = main(
        [
            "--db",
            str(db_path),
            "research-demo",
            "--dry-run",
            "--output",
            str(report_path),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Research demo complete" in output
    assert report_path.exists()


def test_research_memo_llm_option_falls_back_without_provider(tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    db_path = tmp_path / "research.db"
    result = run_research_demo(dry_run=True, db_path=db_path, output_path=tmp_path / "map.md")
    memo_path = tmp_path / "memo.md"

    memo = write_research_memo(
        result.topic_id,
        "What is known?",
        db_path=db_path,
        output_path=memo_path,
        use_llm=True,
    )

    assert not memo.used_llm
    assert "# Research Memo" in memo_path.read_text(encoding="utf-8")


def test_research_memo_creates_markdown_with_required_sections(tmp_path):
    db_path = tmp_path / "research.db"
    result = run_research_demo(dry_run=True, db_path=db_path, output_path=tmp_path / "map.md")
    memo_path = tmp_path / "memo.md"

    memo = write_research_memo(
        result.topic_id,
        "Does causal ordering in two-decision-maker ATS/CDM games plausibly recover decidability for distributed safety synthesis?",
        db_path=db_path,
        output_path=memo_path,
    )
    text = memo_path.read_text(encoding="utf-8")

    assert memo.memo_path == memo_path
    assert memo.evidence_count >= 5
    assert "# Research Memo" in text
    assert "## Known Results" in text
    assert "## Conjecture" in text
    assert "## Reasons for Caution" in text
    assert "## Next Verification Step" in text
    assert "conjecture:" in text
    assert "needs verification" in text
    assert "evidence " in text
    assert "results/approved/" in text


def test_quality_check_literature_creates_report(tmp_path):
    db_path = tmp_path / "research.db"
    result = run_research_demo(dry_run=True, db_path=db_path, output_path=tmp_path / "map.md")
    write_research_memo(
        result.topic_id,
        "Does causal ordering in two-decision-maker ATS/CDM games plausibly recover decidability for distributed safety synthesis?",
        db_path=db_path,
        output_path=tmp_path / "memo.md",
    )
    report_path = tmp_path / "quality.md"

    quality = quality_check_literature(result.topic_id, db_path=db_path, output_path=report_path)
    text = report_path.read_text(encoding="utf-8")

    assert quality.report_path == report_path
    assert "# Quality Report" in text
    assert "## Coverage" in text
    assert "## Citation / Evidence Quality" in text
    assert "## Recommended Fixes" in text
    assert "insufficient stored evidence" in text


def test_generate_verification_tasks_creates_at_least_three_tasks(tmp_path):
    db_path = tmp_path / "research.db"
    result = run_research_demo(dry_run=True, db_path=db_path, output_path=tmp_path / "map.md")
    write_research_memo(
        result.topic_id,
        "Does causal ordering in two-decision-maker ATS/CDM games plausibly recover decidability for distributed safety synthesis?",
        db_path=db_path,
        output_path=tmp_path / "memo.md",
    )
    tasks_path = tmp_path / "tasks.md"

    tasks = generate_verification_tasks(result.topic_id, db_path=db_path, output_path=tasks_path)
    text = tasks_path.read_text(encoding="utf-8")

    assert tasks.tasks_path == tasks_path
    assert "# Verification Tasks" in text
    assert tasks.task_count >= 3
    assert text.count("## Task ") >= 3
    assert "Evidence currently available:" in text
    assert "needs verification" in text

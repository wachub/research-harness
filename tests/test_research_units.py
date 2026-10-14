from __future__ import annotations

from pathlib import Path

import pytest

from src import db
from src.cli import main
from src.research_units import add_paper_for_unit, record_partial_result
from src.schemas import ResearchTask, ResearchUnit, ResearchUnitLink


def _task(path: Path, name: str = "Research task") -> int:
    with db.get_connection(path) as connection:
        db.create_tables(connection)
        return db.insert_task(connection, ResearchTask(name=name))


def test_partial_result_branches_without_establishing_a_claim(tmp_path):
    path = tmp_path / "units.db"
    task_id = _task(path)
    with db.get_connection(path) as connection:
        parent_id = db.insert_research_unit(
            connection,
            ResearchUnit(task_id=task_id, kind="hypothesis", title="Test a fragment", purpose="Look for a bound", status="active"),
        )
    result_id, child_id = record_partial_result(
        parent_id, "Small case", "The checked small case has no counterexample.",
        followup_title="Check larger cases", followup_purpose="Test whether the observation persists",
        db_path=path,
    )
    with db.get_connection(path) as connection:
        result = next(item for item in db.list_derived_results(connection, task_id) if item.id == result_id)
        child = db.get_research_unit(connection, child_id)
        assert result.status == "draft"
        assert child.parent_unit_id == parent_id
        assert db.next_research_unit(connection, task_id).unit_id == parent_id
        assert ResearchUnitLink(unit_id=parent_id, relation="produces", object_type="derived_result", object_id=result_id) in db.list_research_unit_links(connection, parent_id)
        assert ResearchUnitLink(unit_id=child_id, relation="uses", object_type="derived_result", object_id=result_id) in db.list_research_unit_links(connection, child_id)
        db.update_research_unit(connection, parent_id, "finished", "Partial result; larger cases remain.")
        assert db.next_research_unit(connection, task_id).unit_id == child_id
        assert any(event.object_type == "research_unit" for event in db.list_research_events(connection, task_id))


def test_unit_links_validate_targets_and_literature_intake(tmp_path):
    path = tmp_path / "links.db"
    task_id = _task(path)
    other_id = _task(path, "Other task")
    with db.get_connection(path) as connection:
        unit_id = db.insert_research_unit(
            connection, ResearchUnit(task_id=task_id, kind="literature_review", title="Survey", purpose="Find relevant sources"),
        )
        with pytest.raises(ValueError, match="same task"):
            db.insert_research_unit(
                connection, ResearchUnit(task_id=other_id, kind="question", title="Wrong parent", purpose="Invalid", parent_unit_id=unit_id),
            )
        with pytest.raises(ValueError, match="does not exist"):
            db.link_research_unit(
                connection, ResearchUnitLink(unit_id=unit_id, relation="uses", object_type="paper", object_id=999999),
            )
    paper_id = add_paper_for_unit(unit_id, title="Relevant paper", authors=["A. Researcher"], year=2025, db_path=path)
    with db.get_connection(path) as connection:
        paper = db.get_paper(connection, paper_id)
        assert paper.task_id == task_id
        assert ResearchUnitLink(unit_id=unit_id, relation="produces", object_type="paper", object_id=paper_id) in db.list_research_unit_links(connection, unit_id)
        other_paper = db.insert_paper(
            connection, paper.model_copy(update={"id": None, "title": "Other source", "task_id": other_id}),
        )
        with pytest.raises(ValueError, match="another task"):
            db.link_research_unit(
                connection, ResearchUnitLink(unit_id=unit_id, relation="produces", object_type="paper", object_id=other_paper),
            )
        db.link_research_unit(
            connection, ResearchUnitLink(unit_id=unit_id, relation="uses", object_type="paper", object_id=other_paper),
        )
        assert {item.id for item in db.list_papers(connection, task_id)} == {paper_id, other_paper}
        second_unit = db.insert_research_unit(
            connection, ResearchUnit(task_id=other_id, kind="literature_review", title="Reuse source", purpose="Compare tasks"),
        )
        db.link_research_unit(
            connection, ResearchUnitLink(unit_id=second_unit, relation="uses", object_type="paper", object_id=paper_id),
        )
        assert {item.id for item in db.list_papers(connection, other_id)} == {paper_id, other_paper}


def test_cli_creates_and_selects_a_research_unit(tmp_path, capsys):
    path = tmp_path / "cli.db"
    task_id = _task(path)
    assert main(["--db", str(path), "add-research-unit", "--task-id", str(task_id),
                 "--kind", "literature_review", "--title", "Survey", "--purpose", "Gather sources"]) == 0
    assert main(["--db", str(path), "next-research-unit", "--task-id", str(task_id)]) == 0
    assert '"title": "Survey"' in capsys.readouterr().out

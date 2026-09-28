from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import db
from src.cli import main
from src.curate import approve_pending
from src.llm import LLMClient, LLMResponse
from src.research_controller import ResearchController
from src.research_policy import ControllerMode
from src.research_units import add_paper_for_unit, record_partial_result
from src.schemas import OpenProblem, PendingEntry, ResearchTask, ResearchUnit, ResearchUnitLink


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


def test_approved_candidate_keeps_research_unit_provenance(tmp_path):
    path = tmp_path / "review.db"
    task_id = _task(path)
    with db.get_connection(path) as connection:
        unit_id = db.insert_research_unit(
            connection, ResearchUnit(task_id=task_id, kind="investigation", title="Identify gap", purpose="Compare assumptions"),
        )
        pending_id = db.insert_pending_entry(
            connection,
            PendingEntry(entry_type="open_problem", payload=OpenProblem(title="Gap", statement="Does the bound extend?", task_id=task_id).model_dump()),
        )
        db.link_research_unit(
            connection, ResearchUnitLink(unit_id=unit_id, relation="produces", object_type="pending_entry", object_id=pending_id),
        )
    approval = approve_pending(pending_id, db_path=path)
    with db.get_connection(path) as connection:
        assert ResearchUnitLink(unit_id=unit_id, relation="produces", object_type="open_problem", object_id=approval.inserted_id) in db.list_research_unit_links(connection, unit_id)


class SequenceProvider:
    provider_name = "fake"
    model = "fake-unit-controller"

    def __init__(self, responses: list[dict]):
        self.responses = list(responses)

    def complete(self, request):
        return LLMResponse(content=json.dumps(self.responses.pop(0)), provider=self.provider_name, model=self.model)


def _action(action_type: str, parameters: dict | None = None, references: list[dict] | None = None) -> dict:
    return {"action": {
        "action_type": action_type, "reason": "A bounded next step", "expected_effect": "A reviewable activity",
        "references": references or [], "parameters": parameters or {},
    }}


def test_controller_spawns_literature_branch_with_validated_problem_link(tmp_path):
    path = tmp_path / "controller.db"
    task_id = _task(path)
    with db.get_connection(path) as connection:
        unit_id = db.insert_research_unit(
            connection, ResearchUnit(task_id=task_id, kind="hypothesis", title="Examine bound", purpose="Check assumptions", status="ready"),
        )
        problem_id = db.insert_open_problem(
            connection, OpenProblem(title="Missing case", statement="What happens in three processes?", task_id=task_id),
        )
    provider = SequenceProvider([
        _action(
            "request_literature_review",
            {"question": "Find relevant three-process results", "rationale": "Assumptions are unclear", "uncertainty_note": "Sources may be missing"},
            [{"kind": "open_problem", "object_id": problem_id}],
        ),
        _action("stop"),
    ])
    run = ResearchController(LLMClient(provider=provider), db_path=path).run(
        "Investigate the gap", task_id, unit_id=unit_id, mode=ControllerMode.AUTONOMOUS,
    )
    assert json.loads(run.log_path.read_text(encoding="utf-8").splitlines()[0])["research_unit_id"] == unit_id
    assert run.status == "stopped"
    with db.get_connection(path) as connection:
        children = [unit for unit in db.list_research_units(connection, task_id) if unit.parent_unit_id == unit_id]
        assert len(children) == 1
        assert children[0].kind == "literature_review"
        assert children[0].status == "proposed"
        assert db.get_research_unit(connection, unit_id).status == "blocked"
        assert db.next_research_unit(connection, task_id).unit_id == children[0].unit_id
        assert ResearchUnitLink(
            unit_id=children[0].unit_id, relation="investigates", object_type="open_problem", object_id=problem_id,
        ) in db.list_research_unit_links(connection, children[0].unit_id)
        assert db.list_papers(connection, task_id) == []


def test_controller_selects_promising_open_unit_and_validates_choice(tmp_path):
    path = tmp_path / "choice.db"
    task_id = _task(path)
    with db.get_connection(path) as connection:
        first_id = db.insert_research_unit(
            connection, ResearchUnit(task_id=task_id, kind="hypothesis", title="First", purpose="Check a known case", status="active"),
        )
        second_id = db.insert_research_unit(
            connection, ResearchUnit(task_id=task_id, kind="hypothesis", title="Second", purpose="Resolve the key gap", status="active"),
        )
    action = _action(
        "create_pending_open_problem",
        {"question": "Does the key gap extend?", "rationale": "Unclear boundary", "assumptions": [], "uncertainty_note": "Unverified"},
    )
    controller = ResearchController(
        LLMClient(provider=SequenceProvider([
            {"unit_id": second_id, "rationale": "The second unit is more relevant to the goal."},
            action,
        ])),
        db_path=path,
    )
    run = controller.run("Resolve the key gap", task_id, mode=ControllerMode.AUTONOMOUS, max_steps=1)
    assert run.status == "max_steps"
    assert controller.selected_unit_id == second_id
    with db.get_connection(path) as connection:
        pending = db.list_pending_entries(connection)
        assert len(pending) == 1
        assert ResearchUnitLink(
            unit_id=second_id, relation="produces", object_type="pending_entry", object_id=pending[0].id,
        ) in db.list_research_unit_links(connection, second_id)
        assert db.list_research_unit_links(connection, first_id) == []

    invalid = ResearchController(
        LLMClient(provider=SequenceProvider([
            {"unit_id": 999999, "rationale": "Invented candidate."},
        ])),
        db_path=path,
    )
    with pytest.raises(ValueError, match="unknown unit"):
        invalid.run("Resolve the key gap", task_id, mode=ControllerMode.AUTONOMOUS, max_steps=1)
    with db.get_connection(path) as connection:
        assert len(db.list_pending_entries(connection)) == 1

    forced = ResearchController(
        LLMClient(provider=SequenceProvider([_action("stop")])),
        db_path=path,
    )
    forced.run("Resolve the key gap", task_id, unit_id=first_id, mode=ControllerMode.AUTONOMOUS, max_steps=1)
    assert forced.selected_unit_id == first_id


def test_cli_creates_and_selects_a_research_unit(tmp_path, capsys):
    path = tmp_path / "cli.db"
    task_id = _task(path)
    assert main(["--db", str(path), "add-research-unit", "--task-id", str(task_id),
                 "--kind", "literature_review", "--title", "Survey", "--purpose", "Gather sources"]) == 0
    assert main(["--db", str(path), "next-research-unit", "--task-id", str(task_id)]) == 0
    assert '"title": "Survey"' in capsys.readouterr().out


def test_extraction_under_unit_scopes_approved_problem_to_task(tmp_path):
    path = tmp_path / "extract.db"
    task_id = _task(path)
    with db.get_connection(path) as connection:
        unit_id = db.insert_research_unit(
            connection, ResearchUnit(task_id=task_id, kind="literature_review", title="Read source", purpose="Extract open questions"),
        )
    assert main(["--db", str(path), "extract-from-text", "--text",
                 "Open problem: Does finite memory suffice for global safety?",
                 "--unit-id", str(unit_id)]) == 0
    with db.get_connection(path) as connection:
        entries = db.list_pending_entries(connection)
        assert len(entries) == 1
        assert entries[0].entry_type == "open_problem"
        assert entries[0].payload["task_id"] == task_id
        pending_id = entries[0].id
    approval = approve_pending(pending_id, db_path=path)
    with db.get_connection(path) as connection:
        assert len(db.list_open_problems(connection, task_id=task_id)) == 1
        assert ResearchUnitLink(
            unit_id=unit_id, relation="produces", object_type="open_problem", object_id=approval.inserted_id,
        ) in db.list_research_unit_links(connection, unit_id)

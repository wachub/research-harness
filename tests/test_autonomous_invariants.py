"""Adversarial invariants for the bounded, autonomous research workflow."""

from __future__ import annotations

import json

import pytest

from src import autonomous_research, db
from src.autonomous_research import AutonomousResearch
from src.literature.discovery import LiteratureError
from src.llm import LLMClient, LLMResponse
from src.schemas import CodeArtifact, ResearchTask, ResearchUnit, Theorem


class RecordingProvider:
    provider_name = "test"
    model = "test-model"

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return LLMResponse(
            content=json.dumps(self.responses.pop(0)),
            provider=self.provider_name,
            model=self.model,
        )


def proposal(kind="analysis", **changes):
    value = {
        "kind": kind,
        "title": "Investigate a bounded question",
        "purpose": "Examine the current task",
        "outcome": "Tentative observation",
        "rationale": "Relevant to the objective",
        "uncertainty_note": "Not verified",
        "references": [],
    }
    value.update(changes)
    return value


@pytest.fixture
def research_db(tmp_path, monkeypatch):
    monkeypatch.setattr(autonomous_research, "DIAGNOSTIC_LOG", tmp_path / "errors.jsonl")
    path = tmp_path / "research.db"
    with db.get_connection(path) as connection:
        db.create_tables(connection)
        task_id = db.insert_task(connection, ResearchTask(name="Primary goal"))
    return path, task_id


def add_unit(path, task_id, title="Existing unit"):
    with db.get_connection(path) as connection:
        return db.insert_research_unit(connection, ResearchUnit(
            task_id=task_id, kind="analysis", title=title, purpose="Prior work",
            status="finished",
        ))


def completed_events(connection, task_id):
    return [event for event in db.list_research_events(connection, task_id)
            if event.event_type == "research_step_completed"]


@pytest.mark.parametrize("steps", [0, -1, 101])
def test_invalid_step_count_rejected_before_model_or_database_write(research_db, steps):
    path, task_id = research_db
    provider = RecordingProvider([])
    with pytest.raises(ValueError, match="steps must be between"):
        AutonomousResearch(LLMClient(provider=provider), path).run(task_id, steps=steps)
    assert provider.requests == []
    with db.get_connection(path) as connection:
        assert db.list_research_units(connection, task_id) == []


def test_selected_unit_must_belong_to_task_before_model_call(research_db):
    path, task_id = research_db
    with db.get_connection(path) as connection:
        other_task = db.insert_task(connection, ResearchTask(name="Other goal"))
    other_unit = add_unit(path, other_task)
    provider = RecordingProvider([])
    with pytest.raises(ValueError, match="does not belong"):
        AutonomousResearch(LLMClient(provider=provider), path).run(task_id, unit_id=other_unit)
    assert provider.requests == []
    with db.get_connection(path) as connection:
        assert db.list_research_units(connection, task_id) == []


def test_manual_focus_applies_only_to_first_step(research_db):
    path, task_id = research_db
    first = add_unit(path, task_id, "First branch")
    selected = add_unit(path, task_id, "Selected branch")
    provider = RecordingProvider([
        proposal(title="Work on selected branch"),
        {"unit_id": first, "rationale": "Return to first branch"},
        proposal(title="Work on first branch"),
    ])
    result = AutonomousResearch(LLMClient(provider=provider), path).run(
        task_id, steps=2, unit_id=selected,
    )
    assert result.status == "completed"
    assert len(result.unit_ids) == 2
    assert [request.model_role for request in provider.requests] == [
        "research_step", "unit_selection", "research_step",
    ]
    with db.get_connection(path) as connection:
        new_units = [db.get_research_unit(connection, unit_id) for unit_id in result.unit_ids]
        assert new_units[0].parent_unit_id == selected
        assert new_units[1].parent_unit_id == first
        assert len(completed_events(connection, task_id)) == 2


@pytest.mark.parametrize("selection", [999999, "other_task"])
def test_model_cannot_select_unit_outside_candidate_set(research_db, selection):
    path, task_id = research_db
    add_unit(path, task_id, "One")
    add_unit(path, task_id, "Two")
    with db.get_connection(path) as connection:
        other_task = db.insert_task(connection, ResearchTask(name="Other goal"))
    other_unit = add_unit(path, other_task)
    chosen = other_unit if selection == "other_task" else selection
    provider = RecordingProvider([{"unit_id": chosen, "rationale": "Choose this"}])
    result = AutonomousResearch(LLMClient(provider=provider), path).run(task_id)
    assert result.status == "stopped"
    assert result.unit_ids == ()
    assert len(provider.requests) == 1
    with db.get_connection(path) as connection:
        assert len(db.list_research_units(connection, task_id)) == 2
        assert completed_events(connection, task_id) == []


@pytest.mark.parametrize("reference_field", ["references", "target"])
def test_model_cannot_use_out_of_task_theorem(research_db, reference_field):
    path, task_id = research_db
    with db.get_connection(path) as connection:
        other_task = db.insert_task(connection, ResearchTask(name="Other goal"))
        foreign_theorem = db.insert_theorem(connection, Theorem(
            statement="A result belonging only to another task", task_id=other_task,
        ))
    foreign_reference = {"kind": "theorem", "object_id": foreign_theorem}
    if reference_field == "references":
        response = proposal(references=[foreign_reference])
    else:
        response = proposal("proof_attempt", target=foreign_reference)
    provider = RecordingProvider([response])
    result = AutonomousResearch(LLMClient(provider=provider), path).run(task_id)
    assert result.status == "stopped"
    with db.get_connection(path) as connection:
        assert db.list_research_units(connection, task_id) == []
        assert db.list_proof_attempts(connection, task_id) == []
        assert completed_events(connection, task_id) == []


@pytest.mark.parametrize("bad_proposal", [
    proposal("partial_result", statement="A claimed theorem", status="proved"),
    proposal("conjecture", statement="A claimed theorem", confidence="verified"),
    proposal("approve_pending"),
    proposal("partial_result"),
])
def test_invalid_or_status_promoting_model_output_never_persists(research_db, bad_proposal):
    path, task_id = research_db
    provider = RecordingProvider([bad_proposal])
    result = AutonomousResearch(LLMClient(provider=provider), path).run(task_id)
    assert result.status == "stopped"
    assert result.unit_ids == ()
    with db.get_connection(path) as connection:
        assert db.list_research_units(connection, task_id) == []
        assert db.list_derived_results(connection, task_id) == []
        assert db.list_conjectures(connection, task_id) == []
        assert completed_events(connection, task_id) == []


def test_second_step_failure_keeps_only_first_committed_step(research_db):
    path, task_id = research_db
    provider = RecordingProvider([
        proposal("partial_result", statement="Provisional first finding"),
        proposal("partial_result"),
    ])
    result = AutonomousResearch(LLMClient(provider=provider), path).run(task_id, steps=2)
    assert result.status == "stopped"
    assert len(result.unit_ids) == 1
    with db.get_connection(path) as connection:
        assert len(db.list_research_units(connection, task_id)) == 1
        results = db.list_derived_results(connection, task_id)
        assert len(results) == 1 and results[0].status == "draft"
        assert len(completed_events(connection, task_id)) == 1


def test_failed_literature_second_step_does_not_create_blocked_unit(research_db, monkeypatch):
    path, task_id = research_db

    def unavailable(*_args, **_kwargs):
        raise LiteratureError("No accessible full text")

    monkeypatch.setattr(autonomous_research, "discover_full_text", unavailable)
    provider = RecordingProvider([proposal("analysis"), proposal("literature_review")])
    result = AutonomousResearch(LLMClient(provider=provider), path).run(task_id, steps=2)
    assert result.status == "stopped"
    assert len(result.unit_ids) == 1
    assert result.blocked_unit_ids == ()
    assert result.diagnostic_id
    with db.get_connection(path) as connection:
        units = db.list_research_units(connection, task_id)
        assert [item.unit_id for item in units] == list(result.unit_ids)
        assert len(completed_events(connection, task_id)) == 1
        assert db.list_papers(connection, task_id) == []


def test_failed_unit_insert_rolls_back_provisional_result(research_db, monkeypatch):
    path, task_id = research_db

    def fail_unit_insert(*_args, **_kwargs):
        raise ValueError("injected unit failure")

    monkeypatch.setattr(db, "insert_research_unit", fail_unit_insert)
    provider = RecordingProvider([proposal("partial_result", statement="Provisional finding")])
    result = AutonomousResearch(LLMClient(provider=provider), path).run(task_id)
    assert result.status == "stopped"
    with db.get_connection(path) as connection:
        assert db.list_derived_results(connection, task_id) == []
        assert db.list_research_units(connection, task_id) == []
        assert completed_events(connection, task_id) == []


def test_untrusted_artifact_cannot_be_executed_or_recorded(research_db):
    path, task_id = research_db
    with db.get_connection(path) as connection:
        artifact_id = db.insert_code_artifact(connection, CodeArtifact(
            name="Untrusted checker", path="src/experiments/ats_brute_solver.py",
            artifact_type="checker", task_id=task_id, status="draft",
        ))
    provider = RecordingProvider([proposal("bounded_experiment", experiment={
        "artifact_id": artifact_id, "kind": "ATS", "process_count": 2,
        "states_per_process": 2, "depth": 0,
    })])
    result = AutonomousResearch(LLMClient(provider=provider), path).run(task_id)
    assert result.status == "stopped"
    with db.get_connection(path) as connection:
        assert db.list_experiment_runs(connection, task_id=task_id) == []
        assert db.list_research_units(connection, task_id) == []

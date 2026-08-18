from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import db
from src.llm import LLMClient, LLMError, LLMResponse
from src.research_controller import ResearchController
from src.research_policy import ControllerMode
from src.schemas import CodeArtifact, PendingEntry, ResearchCluster, Theorem


class SequenceProvider:
    provider_name = "fake"
    model = "fake-controller"

    def __init__(self, responses=None, error=None):
        self.responses = list(responses or [])
        self.error = error

    def complete(self, request):
        if self.error:
            raise self.error
        payload = self.responses.pop(0) if self.responses else {"action": {"action_type": "stop", "reason": "done", "expected_effect": "end", "references": [], "parameters": {}}}
        return LLMResponse(content=json.dumps(payload), provider=self.provider_name, model=self.model)


def _action(kind, parameters=None, references=None):
    return {"action": {"action_type": kind, "reason": "safe reason", "expected_effect": "safe effect", "references": references or [], "parameters": parameters or {}}}


def _cluster(db_path: Path, name="Cluster"):
    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        return db.insert_cluster(connection, ResearchCluster(name=name))


def _run(db_path, cluster_id, responses=None, error=None, **kwargs):
    controller = ResearchController(LLMClient(provider=SequenceProvider(responses, error)), db_path=db_path, **kwargs)
    return controller.run("Investigate safely", cluster_id, mode=ControllerMode.AUTONOMOUS)


def _pending_count(db_path):
    with db.get_connection(db_path) as connection:
        return len(db.list_pending_entries(connection))


def _tested_artifact(db_path, cluster_id):
    with db.get_connection(db_path) as connection:
        return db.insert_code_artifact(
            connection,
            CodeArtifact(
                name="Tiny checker",
                path="src/experiments/ats_brute_solver.py",
                artifact_type="checker",
                cluster_id=cluster_id,
                status="tested",
            ),
        )


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        _action("arbitrary_shell_command", {"command": "touch should_not_exist"}),
        _action("run_trusted_experiment", {"artifact_id": 999, "depth": 999}),
    ],
)
def test_malformed_unknown_and_excessive_actions_never_write_or_execute(tmp_path, payload):
    db_path = tmp_path / "research.db"
    cluster_id = _cluster(db_path)
    result = _run(db_path, cluster_id, [payload, payload])
    assert result.status == "invalid_output"
    assert _pending_count(db_path) == 0
    assert not (tmp_path / "should_not_exist").exists()


def test_cross_cluster_and_unknown_ids_are_rejected(tmp_path):
    db_path = tmp_path / "research.db"
    selected = _cluster(db_path, "Selected")
    other = _cluster(db_path, "Other")
    with db.get_connection(db_path) as connection:
        theorem_id = db.insert_theorem(connection, Theorem(title="Other theorem", statement="Other", cluster_id=other))
    invalid = _action(
        "create_pending_conjecture",
        {"statement": "Candidate", "rationale": "Reason", "assumptions": [], "uncertainty_note": "Unknown"},
        [{"kind": "theorem", "object_id": theorem_id}],
    )
    result = _run(db_path, selected, [invalid, invalid])
    assert result.status == "invalid_output"
    assert _pending_count(db_path) == 0


def test_unregistered_artifact_and_conjecture_promotion_are_rejected(tmp_path):
    db_path = tmp_path / "research.db"
    cluster_id = _cluster(db_path)
    unregistered = _action(
        "run_trusted_experiment",
        {"artifact_id": 999, "kind": "ATS", "process_count": 2, "states_per_process": 2, "depth": 2, "seed": 0},
    )
    result = _run(db_path, cluster_id, [unregistered, _action("stop")])
    assert result.status == "stopped"
    assert result.steps[0].result["status"] == "invalid_action"
    promoted = _action("mark_conjecture_proved", {"conjecture_id": 1})
    result = _run(db_path, cluster_id, [promoted, promoted])
    assert result.status == "invalid_output"


def test_forbidden_approval_pauses_autonomous_mode_without_approval(tmp_path):
    db_path = tmp_path / "research.db"
    cluster_id = _cluster(db_path)
    with db.get_connection(db_path) as connection:
        pending_id = db.insert_pending_entry(
            connection,
            PendingEntry(entry_type="conjecture_seed", payload={"statement": "Candidate"}),
        )
    result = _run(db_path, cluster_id, [_action("approve_pending", {"pending_id": pending_id})])
    assert result.status == "paused_for_review"
    with db.get_connection(db_path) as connection:
        assert db.get_pending_entry(connection, pending_id).status == "pending"
        assert db.list_conjectures(connection) == []


def test_repeated_action_and_duplicate_pending_proposal_are_safe(tmp_path):
    db_path = tmp_path / "research.db"
    cluster_id = _cluster(db_path)
    create = _action("create_pending_conjecture", {"statement": "Candidate", "rationale": "Reason", "assumptions": [], "uncertainty_note": "Unknown"})
    repeat = _run(db_path, cluster_id, [create, create])
    assert repeat.status == "repeated_action"
    assert _pending_count(db_path) == 1
    duplicate = _run(db_path, cluster_id, [create, _action("stop")])
    assert duplicate.steps[0].success is False
    assert _pending_count(db_path) == 1


def test_provider_failures_and_failed_handler_stop_without_partial_write(tmp_path, monkeypatch):
    db_path = tmp_path / "research.db"
    cluster_id = _cluster(db_path)
    failure = _run(db_path, cluster_id, error=LLMError("network unavailable"))
    assert failure.status == "provider_failure"
    create = _action("create_pending_conjecture", {"statement": "Candidate", "rationale": "Reason", "assumptions": [], "uncertainty_note": "Unknown"})
    monkeypatch.setattr("src.research_controller.db.insert_pending_entry", lambda *args: (_ for _ in ()).throw(RuntimeError("database failed")))
    failed = _run(db_path, cluster_id, [create, create])
    assert failed.status in {"repeated_action", "action_failure"}
    assert _pending_count(db_path) == 0


def test_failed_trusted_experiment_never_creates_a_run(tmp_path, monkeypatch):
    db_path = tmp_path / "research.db"
    cluster_id = _cluster(db_path)
    artifact_id = _tested_artifact(db_path, cluster_id)
    action = _action(
        "run_trusted_experiment",
        {"artifact_id": artifact_id, "kind": "ATS", "process_count": 2, "states_per_process": 2, "depth": 2, "seed": 0},
    )
    monkeypatch.setattr(
        "src.research_controller.run_trusted_bounded_experiment",
        lambda *args: (_ for _ in ()).throw(RuntimeError("checker failed")),
    )
    result = _run(db_path, cluster_id, [action, _action("stop")])
    assert result.steps[0].result["status"] == "action_failure"
    with db.get_connection(db_path) as connection:
        assert db.list_experiment_runs(connection, cluster_id=cluster_id) == []


def test_provider_metadata_logs_are_redacted_and_interrupt_is_clean(tmp_path, monkeypatch):
    db_path = tmp_path / "research.db"
    cluster_id = _cluster(db_path)
    log_path = tmp_path / "controller.jsonl"
    controller = ResearchController(LLMClient(provider=SequenceProvider([_action("inspect_state")])), db_path=db_path)
    result = controller.run("Inspect safely", cluster_id, mode=ControllerMode.AUTONOMOUS, max_steps=1, log_path=log_path)
    assert result.status == "max_steps"
    record = json.loads(log_path.read_text().splitlines()[0])
    assert "api_key" not in json.dumps(record["provider"]).lower()
    monkeypatch.setattr(controller, "_next_action", lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
    interrupted = controller.run("Interrupt", cluster_id, mode=ControllerMode.AUTONOMOUS, log_path=tmp_path / "interrupt.jsonl")
    assert interrupted.status == "interrupted"

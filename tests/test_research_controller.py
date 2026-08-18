from __future__ import annotations

import json
from pathlib import Path

from src import db
from src.llm import LLMClient, LLMError, LLMResponse
from src.research_controller import ResearchController
from src.research_policy import AuthorityDecision, ControllerMode, authority_for
from src.schemas import CodeArtifact, PendingEntry, ResearchCluster


class SequenceProvider:
    provider_name = "fake"
    model = "fake-controller"

    def __init__(self, responses: list[dict] | None = None, error: Exception | None = None):
        self.responses = list(responses or [])
        self.error = error

    def complete(self, request):
        if self.error:
            raise self.error
        payload = self.responses.pop(0) if self.responses else _action("stop")
        return LLMResponse(content=json.dumps(payload), provider=self.provider_name, model=self.model)


def _action(action_type: str, parameters: dict | None = None, references: list[dict] | None = None) -> dict:
    return {
        "action": {
            "action_type": action_type,
            "reason": "This is a bounded next step.",
            "expected_effect": "Produce only a reviewable or bounded result.",
            "references": references or [],
            "parameters": parameters or {},
        }
    }


def _cluster(db_path: Path, name: str = "Controller cluster") -> int:
    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        return db.insert_cluster(connection, ResearchCluster(name=name))


def _controller(db_path: Path, responses: list[dict], approval=None) -> ResearchController:
    return ResearchController(LLMClient(provider=SequenceProvider(responses)), db_path=db_path, approval_callback=approval)


def _pending_count(db_path: Path) -> int:
    with db.get_connection(db_path) as connection:
        return len(db.list_pending_entries(connection))


def _tested_artifact(db_path: Path, cluster_id: int) -> int:
    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
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


def test_authority_policy_is_centralized_for_both_modes():
    assert authority_for("inspect_state", ControllerMode.INTERACTIVE) is AuthorityDecision.AUTO
    assert authority_for("create_pending_conjecture", ControllerMode.INTERACTIVE) is AuthorityDecision.ASK
    assert authority_for("create_pending_conjecture", ControllerMode.AUTONOMOUS) is AuthorityDecision.AUTO
    assert authority_for("approve_pending", ControllerMode.AUTONOMOUS) is AuthorityDecision.BLOCK
    assert authority_for("arbitrary_shell_command", ControllerMode.INTERACTIVE) is AuthorityDecision.BLOCK


def test_interactive_mode_asks_and_rejection_creates_no_pending_state(tmp_path):
    db_path = tmp_path / "research.db"
    cluster_id = _cluster(db_path)
    action = _action(
        "create_pending_conjecture",
        {"statement": "A cautious candidate.", "rationale": "Worth review.", "assumptions": [], "uncertainty_note": "Uncertain."},
    )
    result = _controller(db_path, [action], approval=lambda _: "reject").run(
        "Investigate a question", cluster_id, mode=ControllerMode.INTERACTIVE
    )
    assert result.status == "rejected_by_user"
    assert result.steps[0].policy_decision == "ASK"
    assert _pending_count(db_path) == 0


def test_interactive_approval_executes_a_provisional_action_only(tmp_path):
    db_path = tmp_path / "research.db"
    cluster_id = _cluster(db_path)
    action = _action(
        "create_pending_open_problem",
        {"question": "Does a bounded fragment suffice?", "rationale": "The state is incomplete.", "assumptions": [], "uncertainty_note": "Needs review."},
    )
    result = _controller(db_path, [action, _action("stop")], approval=lambda _: "approve").run(
        "Investigate a question", cluster_id, mode=ControllerMode.INTERACTIVE
    )
    assert result.status == "stopped"
    assert result.steps[0].success is True
    with db.get_connection(db_path) as connection:
        assert db.list_open_problems(connection, cluster_id=cluster_id) == []
        assert db.list_pending_entries(connection)[0].entry_type == "open_problem"


def test_autonomous_mode_creates_only_pending_conjecture_then_stops(tmp_path):
    db_path = tmp_path / "research.db"
    cluster_id = _cluster(db_path)
    create = _action(
        "create_pending_conjecture",
        {"statement": "A cautious candidate.", "rationale": "Worth review.", "assumptions": ["finite arena"], "uncertainty_note": "Uncertain."},
    )
    result = _controller(db_path, [create, _action("stop")]).run(
        "Investigate a question", cluster_id, mode=ControllerMode.AUTONOMOUS
    )
    assert result.status == "stopped"
    assert result.steps[0].policy_decision == "AUTO"
    assert _pending_count(db_path) == 1
    with db.get_connection(db_path) as connection:
        assert db.list_conjectures(connection, cluster_id=cluster_id) == []


def test_read_only_action_runs_in_both_modes(tmp_path):
    for mode in (ControllerMode.INTERACTIVE, ControllerMode.AUTONOMOUS):
        db_path = tmp_path / f"{mode.value}.db"
        cluster_id = _cluster(db_path)
        result = _controller(db_path, [_action("inspect_state"), _action("stop")]).run(
            "Inspect state", cluster_id, mode=mode
        )
        assert result.steps[0].success is True
        assert result.steps[0].policy_decision == "AUTO"


def test_trusted_experiment_runs_in_process_with_bounded_parameters(tmp_path, monkeypatch):
    db_path = tmp_path / "research.db"
    cluster_id = _cluster(db_path)
    artifact_id = _tested_artifact(db_path, cluster_id)
    monkeypatch.setattr("src.experiment_manager.run_experiment", lambda **_: (_ for _ in ()).throw(AssertionError("shell runner used")))
    run = _action(
        "run_trusted_experiment",
        {"artifact_id": artifact_id, "kind": "ATS", "process_count": 2, "states_per_process": 2, "depth": 2, "seed": 7},
    )
    result = _controller(db_path, [run, _action("stop")]).run(
        "Test tiny games", cluster_id, mode=ControllerMode.AUTONOMOUS
    )
    assert result.status == "stopped"
    with db.get_connection(db_path) as connection:
        runs = db.list_experiment_runs(connection, cluster_id=cluster_id)
    assert len(runs) == 1
    assert runs[0].command_run == "trusted_in_process:ats_bounded_safety"
    assert runs[0].output_json["depth"] == 2


def test_explicit_stop_and_max_steps_are_bounded(tmp_path):
    db_path = tmp_path / "research.db"
    cluster_id = _cluster(db_path)
    stopped = _controller(db_path, [_action("stop")]).run("Stop", cluster_id, mode=ControllerMode.AUTONOMOUS)
    assert stopped.status == "stopped"
    limited = _controller(db_path, [_action("inspect_state"), _action("stop")]).run(
        "Bound", cluster_id, mode=ControllerMode.AUTONOMOUS, max_steps=1
    )
    assert limited.status == "max_steps"


def test_pause_every_and_unavailable_provider_do_not_fake_actions(tmp_path):
    db_path = tmp_path / "research.db"
    cluster_id = _cluster(db_path)
    result = _controller(db_path, [_action("inspect_state")]).run(
        "Pause", cluster_id, mode=ControllerMode.AUTONOMOUS, pause_every=1
    )
    assert result.status == "paused_for_review"
    unavailable = ResearchController(LLMClient(), db_path=db_path).run(
        "Offline", cluster_id, mode=ControllerMode.AUTONOMOUS
    )
    assert unavailable.status == "unavailable"
    assert unavailable.log_path is None

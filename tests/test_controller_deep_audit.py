from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import db
from src.cli import main as cli_main
from src.llm import LLMClient, LLMConfiguration, LLMError, LLMResponse
from src.research_controller import ResearchController
from src.research_policy import AuthorityDecision, ControllerMode, FORBIDDEN_ACTIONS, PROVISIONAL_ACTIONS, READ_ONLY_ACTIONS, authority_for
from src.research_actions import BoundedExperimentParameters
from src.schemas import CodeArtifact, ResearchCluster


class RawProvider:
    provider_name = "audit"
    model = "audit-model"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        response = self.responses.pop(0) if self.responses else "{}"
        if isinstance(response, Exception):
            raise response
        return LLMResponse(content=response, provider=self.provider_name, model=self.model)


class BrokenProvider:
    provider_name = "broken"
    model = "broken-model"

    def __init__(self, response):
        self.response = response
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _cluster(db_path: Path, name: str = "Controller audit") -> int:
    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        return db.insert_cluster(connection, ResearchCluster(name=name))


def _action(action_type: str = "inspect_state", parameters: dict | None = None, references: list[dict] | None = None, **extra) -> str:
    action = {
        "action_type": action_type,
        "reason": "audit reason",
        "expected_effect": "audit effect",
        "references": references or [],
        "parameters": parameters or {},
        **extra,
    }
    return json.dumps({"action": action})


def _controller(db_path: Path, responses, approval=None) -> ResearchController:
    return ResearchController(LLMClient(provider=RawProvider(responses)), db_path=db_path, approval_callback=approval)


def _trusted_artifact(db_path: Path, cluster_id: int, **overrides) -> int:
    fields = {
        "name": "trusted",
        "path": "/bin/sh; rm -rf /; ../../evil.py",
        "artifact_type": "checker",
        "cluster_id": cluster_id,
        "status": "tested",
        **overrides,
    }
    with db.get_connection(db_path) as connection:
        return db.insert_code_artifact(connection, CodeArtifact(**fields))


def test_authority_matrix_is_exhaustive_and_unknown_is_never_permissive():
    for action in READ_ONLY_ACTIONS:
        assert authority_for(action, ControllerMode.INTERACTIVE) is AuthorityDecision.AUTO
        assert authority_for(action, ControllerMode.AUTONOMOUS) is AuthorityDecision.AUTO
    for action in PROVISIONAL_ACTIONS:
        assert authority_for(action, ControllerMode.INTERACTIVE) is AuthorityDecision.ASK
        assert authority_for(action, ControllerMode.AUTONOMOUS) is AuthorityDecision.AUTO
    for action in FORBIDDEN_ACTIONS | {"unknown_action"}:
        assert authority_for(action, ControllerMode.INTERACTIVE) is AuthorityDecision.BLOCK
        assert authority_for(action, ControllerMode.AUTONOMOUS) is AuthorityDecision.BLOCK


@pytest.mark.parametrize(
    "response",
    ["not json", "", "[]", "{}", json.dumps({"wrong": {}}), _action(parameters={}, unexpected="forbidden"), _action("unknown_action"), _action("inspect_state", {"unexpected": 1}), _action("inspect_state", {}, [{"kind": "research_cluster", "object_id": -1}])],
)
def test_invalid_structured_output_never_executes_or_writes_and_hits_cutoff(tmp_path, response):
    db_path = tmp_path / "research.db"
    cluster_id = _cluster(db_path)
    log_path = tmp_path / "audit.jsonl"
    result = _controller(db_path, [response, response]).run("Audit", cluster_id, mode=ControllerMode.AUTONOMOUS, log_path=log_path)
    assert result.status == "invalid_output"
    assert len(result.steps) == 2
    with db.get_connection(db_path) as connection:
        assert db.list_pending_entries(connection) == []
        assert db.list_conjectures(connection) == []
        assert db.list_experiment_runs(connection) == []


def test_duplicate_references_are_rejected_before_action_execution(tmp_path):
    db_path = tmp_path / "research.db"
    cluster_id = _cluster(db_path)
    duplicate = [{"kind": "research_cluster", "object_id": cluster_id}] * 2
    result = _controller(db_path, [_action(references=duplicate), _action(references=duplicate)]).run(
        "Audit", cluster_id, mode=ControllerMode.AUTONOMOUS
    )
    assert result.status == "invalid_output"


def test_invalid_llm_response_and_controller_log_never_leak_configured_secret(tmp_path, monkeypatch):
    db_path = tmp_path / "research.db"
    cluster_id = _cluster(db_path)
    monkeypatch.setenv("LLM_API_KEY", "sk-controller-audit-secret")
    leaked = json.dumps({"action": {"action_type": "inspect_state", "reason": "sk-controller-audit-secret", "expected_effect": "x", "references": [], "parameters": {"bad": "sk-controller-audit-secret"}}})
    log_path = tmp_path / "secret.jsonl"
    result = _controller(db_path, [leaked, leaked]).run("Audit", cluster_id, mode=ControllerMode.AUTONOMOUS, log_path=log_path)
    assert result.status == "invalid_output"
    assert "sk-controller-audit-secret" not in log_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("response", [LLMError("provider failed"), LLMError("provider failed")])
def test_provider_failures_stop_without_stale_actions(tmp_path, response):
    db_path = tmp_path / "research.db"
    cluster_id = _cluster(db_path)
    result = _controller(db_path, [response, response]).run("Audit", cluster_id, mode=ControllerMode.AUTONOMOUS)
    assert result.status == "provider_failure"
    assert all(step.action is None for step in result.steps)


@pytest.mark.parametrize("response", [{"choices": []}, None, TimeoutError("timed out")])
def test_malformed_provider_envelopes_and_timeouts_are_safe_provider_failures(tmp_path, response):
    db_path = tmp_path / "research.db"
    cluster_id = _cluster(db_path)
    provider = BrokenProvider(response)
    controller = ResearchController(LLMClient(provider=provider), db_path=db_path)
    result = controller.run("Audit", cluster_id, mode=ControllerMode.AUTONOMOUS, max_steps=2)
    assert result.status == "provider_failure"
    assert provider.calls == 2
    assert all(step.action is None for step in result.steps)
    assert all(step.result["status"] == "provider_failure" for step in result.steps)


def test_provider_failure_between_valid_responses_never_reuses_the_prior_action(tmp_path):
    db_path = tmp_path / "research.db"
    cluster_id = _cluster(db_path)
    provider = RawProvider([_action("inspect_state"), LLMError("temporary failure"), _action("stop")])
    result = ResearchController(LLMClient(provider=provider), db_path=db_path).run(
        "Audit", cluster_id, mode=ControllerMode.AUTONOMOUS, max_steps=3
    )
    assert result.status == "stopped"
    assert [step.action["action_type"] if step.action else None for step in result.steps] == ["inspect_state", None, "stop"]
    assert result.steps[1].result["status"] == "provider_failure"
    assert provider.calls == 3


@pytest.mark.parametrize("response, expected", [("approve", "stopped"), ("reject", "rejected_by_user"), ("stop", "stopped_by_user"), ("pause", "paused_for_review"), ("", "rejected_by_user"), ("unexpected", "rejected_by_user")])
def test_interactive_callback_variants_never_bypass_pending_boundary(tmp_path, response, expected):
    db_path = tmp_path / "research.db"
    cluster_id = _cluster(db_path)
    create = _action("create_pending_conjecture", {"statement": "Candidate", "rationale": "Reason", "assumptions": [], "uncertainty_note": "Unknown"})
    result = _controller(db_path, [create, _action("stop")], approval=lambda _: response).run(
        "Audit", cluster_id, mode=ControllerMode.INTERACTIVE
    )
    assert result.status == expected
    with db.get_connection(db_path) as connection:
        assert len(db.list_pending_entries(connection)) == (1 if response == "approve" else 0)
        assert db.list_conjectures(connection) == []


def test_interactive_missing_callback_pauses_without_write(tmp_path):
    db_path = tmp_path / "research.db"
    cluster_id = _cluster(db_path)
    create = _action("create_pending_open_problem", {"question": "Question", "rationale": "Reason", "assumptions": [], "uncertainty_note": "Unknown"})
    result = _controller(db_path, [create]).run("Audit", cluster_id, mode=ControllerMode.INTERACTIVE)
    assert result.status == "paused_for_review"
    with db.get_connection(db_path) as connection:
        assert db.list_pending_entries(connection) == []


@pytest.mark.parametrize(
    "action_type, parameters, entry_type",
    [
        ("create_pending_conjecture", {"statement": "Candidate", "rationale": "Reason", "assumptions": ["finite arena"], "uncertainty_note": "Unknown"}, "conjecture_seed"),
        ("create_pending_open_problem", {"question": "Question", "rationale": "Reason", "assumptions": ["finite arena"], "uncertainty_note": "Unknown"}, "open_problem"),
    ],
)
def test_controller_proposals_preserve_review_metadata_and_never_become_durable(tmp_path, action_type, parameters, entry_type):
    db_path = tmp_path / "research.db"
    cluster_id = _cluster(db_path)
    result = _controller(db_path, [_action(action_type, parameters), _action("stop")]).run(
        "Audit", cluster_id, mode=ControllerMode.AUTONOMOUS
    )
    assert result.status == "stopped"
    with db.get_connection(db_path) as connection:
        entries = db.list_pending_entries(connection)
        assert len(entries) == 1
        entry = entries[0]
        assert entry.entry_type == entry_type
        assert entry.payload["cluster_id"] == cluster_id
        assert entry.status == "pending"
        assert "Controller-proposed; requires human review" in entry.warnings
        assert any("uncertainty: Unknown" in warning for warning in entry.warnings)
        assert db.list_conjectures(connection) == []
        assert db.list_open_problems(connection) == []


@pytest.mark.parametrize("process_count, depth", [(2, 0), (3, 5)])
def test_trusted_experiment_boundary_values_are_in_process_and_observational(tmp_path, monkeypatch, process_count, depth):
    db_path = tmp_path / "research.db"
    cluster_id = _cluster(db_path)
    artifact_id = _trusted_artifact(db_path, cluster_id)
    monkeypatch.setattr("src.research_actions.get_current_commit_hash", lambda: "audit")
    parameters = {"artifact_id": artifact_id, "kind": "ATS", "process_count": process_count, "states_per_process": 2, "depth": depth, "seed": 1}
    result = _controller(db_path, [_action("run_trusted_experiment", parameters), _action("stop")]).run("Audit", cluster_id, mode=ControllerMode.AUTONOMOUS)
    assert result.status == "stopped"
    with db.get_connection(db_path) as connection:
        runs = db.list_experiment_runs(connection, cluster_id=cluster_id)
        assert len(runs) == 1
        assert runs[0].command_run == "trusted_in_process:ats_bounded_safety"
        assert db.list_theorems(connection) == []
        assert db.list_conjectures(connection) == []


@pytest.mark.parametrize("field, value", [("process_count", 1), ("process_count", 4), ("states_per_process", 1), ("states_per_process", 3), ("depth", -1), ("depth", 6)])
def test_trusted_experiment_parameter_limits_reject_just_outside_bounds(field, value):
    payload = {"artifact_id": 1, "kind": "ATS", "process_count": 2, "states_per_process": 2, "depth": 5, "seed": 0, field: value}
    with pytest.raises(Exception):
        BoundedExperimentParameters.model_validate(payload)


def test_untested_or_cross_cluster_artifacts_are_rejected_before_runs(tmp_path):
    db_path = tmp_path / "research.db"
    selected = _cluster(db_path, "Selected controller audit")
    other = _cluster(db_path, "Other controller audit")
    untested = _trusted_artifact(db_path, selected, status="draft")
    foreign = _trusted_artifact(db_path, other)
    for artifact_id in (untested, foreign):
        params = {"artifact_id": artifact_id, "kind": "ATS", "process_count": 2, "states_per_process": 2, "depth": 1, "seed": 0}
        result = _controller(db_path, [_action("run_trusted_experiment", params), _action("stop")]).run("Audit", selected, mode=ControllerMode.AUTONOMOUS)
        assert result.steps[0].result["status"] == "invalid_action"
    with db.get_connection(db_path) as connection:
        assert db.list_experiment_runs(connection, cluster_id=selected) == []


def test_log_appends_redacts_secret_and_default_log_can_be_redirected(tmp_path, monkeypatch):
    db_path = tmp_path / "research.db"
    cluster_id = _cluster(db_path)
    monkeypatch.setenv("LLM_API_KEY", "sk-log-audit")
    log_path = tmp_path / "nested" / "controller.jsonl"
    for goal in ("first sk-log-audit", "second sk-log-audit"):
        result = _controller(db_path, [_action("stop")]).run(goal, cluster_id, mode=ControllerMode.AUTONOMOUS, log_path=log_path)
        assert result.status == "stopped"
    lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert all(line["step"]["policy_decision"] == "AUTO" for line in lines)
    assert "sk-log-audit" not in log_path.read_text(encoding="utf-8")
    monkeypatch.setattr("src.research_controller.DEFAULT_LOG_DIRECTORY", tmp_path / "default-logs")
    result = _controller(db_path, [_action("stop")]).run("default", cluster_id, mode=ControllerMode.AUTONOMOUS)
    assert result.log_path and result.log_path.parent == tmp_path / "default-logs"


@pytest.mark.parametrize("goal, max_steps, pause_every", [("", 1, None), (" ", 1, None), ("Audit", 0, None), ("Audit", -1, None), ("Audit", 101, None), ("Audit", 1, 0), ("Audit", 1, -1)])
def test_invalid_controller_bounds_fail_before_provider_or_log_write(tmp_path, goal, max_steps, pause_every):
    db_path = tmp_path / "research.db"
    cluster_id = _cluster(db_path)
    provider = RawProvider([_action("stop")])
    log_path = tmp_path / "should-not-exist.jsonl"
    with pytest.raises(ValueError):
        ResearchController(LLMClient(provider=provider), db_path=db_path).run(
            goal, cluster_id, mode=ControllerMode.AUTONOMOUS, max_steps=max_steps, pause_every=pause_every, log_path=log_path
        )
    assert provider.calls == 0
    assert not log_path.exists()


def test_invalid_cluster_and_unavailable_client_cause_no_provider_or_log_write(tmp_path):
    db_path = tmp_path / "research.db"
    cluster_id = _cluster(db_path)
    provider = RawProvider([_action("stop")])
    missing_log = tmp_path / "missing.jsonl"
    with pytest.raises(ValueError):
        ResearchController(LLMClient(provider=provider), db_path=db_path).run(
            "Audit", cluster_id + 999_999, mode=ControllerMode.AUTONOMOUS, log_path=missing_log
        )
    assert provider.calls == 0
    assert not missing_log.exists()
    unavailable = ResearchController(LLMClient(configuration=LLMConfiguration()), db_path=db_path).run(
        "Audit", cluster_id, mode=ControllerMode.AUTONOMOUS, log_path=tmp_path / "offline.jsonl"
    )
    assert unavailable.status == "unavailable"
    assert unavailable.log_path is None
    assert not (tmp_path / "offline.jsonl").exists()


def test_two_distinct_handler_failures_hit_cutoff_without_writes(tmp_path, monkeypatch):
    db_path = tmp_path / "research.db"
    cluster_id = _cluster(db_path)
    first = _action("propose_subquestions", {"subquestions": [{"question": "first", "rationale": "why", "uncertainty_note": "unknown"}]})
    second = _action("propose_subquestions", {"subquestions": [{"question": "second", "rationale": "why", "uncertainty_note": "unknown"}]})
    controller = _controller(db_path, [first, second])
    monkeypatch.setattr(controller, "_execute", lambda *_: (_ for _ in ()).throw(RuntimeError("handler failure")))
    result = controller.run("Audit", cluster_id, mode=ControllerMode.AUTONOMOUS)
    assert result.status == "action_failure"
    assert len(result.steps) == 2
    assert all(step.result["status"] == "action_failure" for step in result.steps)
    with db.get_connection(db_path) as connection:
        assert db.list_pending_entries(connection) == []
        assert db.list_experiment_runs(connection) == []


def test_cli_research_loop_handles_offline_invalid_cluster_and_invalid_output(tmp_path, monkeypatch, capsys):
    db_path = tmp_path / "research.db"
    cluster_id = _cluster(db_path)
    missing_llm = cli_main(["--db", str(db_path), "research-loop", "--goal", "Audit", "--cluster-id", str(cluster_id), "--mode", "autonomous"])
    assert missing_llm == 2
    assert "requires --llm" in capsys.readouterr().out

    monkeypatch.setattr("src.cli.ProviderLLMClient", lambda: LLMClient(configuration=LLMConfiguration()))
    unavailable = cli_main(["--db", str(db_path), "research-loop", "--goal", "Audit", "--cluster-id", str(cluster_id), "--llm", "--mode", "autonomous"])
    assert unavailable == 2
    assert '"status": "unavailable"' in capsys.readouterr().out

    provider = RawProvider(["{}", "{}"])
    monkeypatch.setattr("src.cli.ProviderLLMClient", lambda: LLMClient(provider=provider))
    log_path = tmp_path / "cli.jsonl"
    invalid = cli_main(["--db", str(db_path), "research-loop", "--goal", "Audit", "--cluster-id", str(cluster_id), "--llm", "--mode", "autonomous", "--log", str(log_path)])
    assert invalid == 2
    assert '"status": "invalid_output"' in capsys.readouterr().out
    assert len(log_path.read_text(encoding="utf-8").splitlines()) == 2

    before_calls = provider.calls
    bad_cluster = cli_main(["--db", str(db_path), "research-loop", "--goal", "Audit", "--cluster-id", str(cluster_id + 99), "--llm", "--mode", "autonomous", "--log", str(tmp_path / "missing.jsonl")])
    assert bad_cluster == 2
    assert "does not exist" in capsys.readouterr().out
    assert provider.calls == before_calls
    assert not (tmp_path / "missing.jsonl").exists()


def test_cli_research_loop_invalid_mode_is_a_clean_parser_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["--db", str(tmp_path / "research.db"), "research-loop", "--goal", "Audit", "--cluster-id", "1", "--llm", "--mode", "unsafe"])
    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "invalid choice" in stderr
    assert "Traceback" not in stderr


def test_cli_research_loop_runs_interactive_and_successful_bounded_paths(tmp_path, monkeypatch, capsys):
    db_path = tmp_path / "research.db"
    cluster_id = _cluster(db_path)
    create = _action("create_pending_conjecture", {"statement": "CLI candidate", "rationale": "Reason", "assumptions": [], "uncertainty_note": "Unknown"})
    monkeypatch.setattr("src.cli.ProviderLLMClient", lambda: LLMClient(provider=RawProvider([create, _action("stop")])))
    monkeypatch.setattr("src.cli._controller_approval_prompt", lambda _: "approve")
    exit_code = cli_main(["--db", str(db_path), "research-loop", "--goal", "Audit", "--cluster-id", str(cluster_id), "--llm", "--mode", "interactive", "--max-steps", "2", "--pause-every", "1", "--log", str(tmp_path / "interactive.jsonl")])
    assert exit_code == 0
    assert '"status": "paused_for_review"' in capsys.readouterr().out
    with db.get_connection(db_path) as connection:
        assert len(db.list_pending_entries(connection)) == 1
        assert db.list_conjectures(connection) == []

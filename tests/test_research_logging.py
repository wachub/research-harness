"""Regression coverage for private, bounded research diagnostics."""

import json
from concurrent.futures import ThreadPoolExecutor

from src import autonomous_research, db
from src.autonomous_research import AutonomousResearch
from src.llm import LLMClient, LLMConfiguration, LLMError, LLMMessage, LLMRequest, LLMResponse
from src.schemas import ResearchTask
from tests.test_autonomous_research import SequenceProvider, _proposal


def _task(path):
    with db.get_connection(path) as connection:
        db.create_tables(connection)
        return db.insert_task(connection, ResearchTask(name="Logging test"))


def test_failed_step_records_stage_timing_and_completed_units(tmp_path, monkeypatch):
    log = tmp_path / "research_errors.jsonl"
    monkeypatch.setattr(autonomous_research, "DIAGNOSTIC_LOG", log)
    path = tmp_path / "research.db"
    task_id = _task(path)
    provider = SequenceProvider([_proposal("analysis", "Initial analysis"), {}])
    result = AutonomousResearch(LLMClient(provider=provider), path).run(task_id, steps=2)
    assert result.status == "stopped" and len(result.unit_ids) == 1
    assert result.diagnostics["completed_unit_ids"] == list(result.unit_ids)
    trace = result.diagnostics["step_trace"]
    assert [item["stage"] for item in trace] == ["select_unit", "load_context", "propose_step"]
    assert [item["status"] for item in trace] == ["completed", "completed", "failed"]
    assert all(item["elapsed_seconds"] >= 0 for item in trace)
    saved = json.loads(log.read_text())
    assert saved["diagnostics"] == result.diagnostics
    assert saved["incident_id"] == result.diagnostic_id


def test_failure_after_model_call_retains_safe_model_timing(tmp_path, monkeypatch):
    log = tmp_path / "research_errors.jsonl"
    monkeypatch.setattr(autonomous_research, "DIAGNOSTIC_LOG", log)
    path = tmp_path / "research.db"
    task_id = _task(path)
    runner = AutonomousResearch(LLMClient(provider=SequenceProvider([_proposal("analysis", "Investigate")])), path)
    monkeypatch.setattr(runner, "_execute", lambda *_: (_ for _ in ()).throw(ValueError("private raw output")))
    result = runner.run(task_id)
    assert result.status == "stopped" and result.diagnostics["stage"] == "execute_step"
    assert result.diagnostics["step_trace"][-1]["status"] == "failed"
    call, = result.diagnostics["model_calls"]
    assert call["role"] == "research_step" and call["elapsed_seconds"] >= 0
    assert "private raw output" not in log.read_text()


def test_malformed_diagnostics_cannot_break_logging_or_expose_raw_values(tmp_path, monkeypatch):
    log = tmp_path / "research_errors.jsonl"
    monkeypatch.setattr(autonomous_research, "DIAGNOSTIC_LOG", log)
    monkeypatch.setenv("OPENALEX_API_KEY", "openalex-secret")
    trace = [{"stage": "download_pdf", "status": "failed", "host": "example.test",
              "raw_response": "private source text", "http_status": 403} for _ in range(220)]
    exc = LLMError("Provider failed at https://example.test/private?key=openalex-secret", diagnostics={
        "category": "invalid_response", "literature_trace": trace,
        "raw_response": "private source text", "hint": "private source text",
        "unserializable": object(),
    })
    incident = autonomous_research._record_research_failure(exc, 1, 2)
    assert incident
    record = json.loads(log.read_text())
    assert record["diagnostics"]["category"] == "invalid_response"
    assert len(record["diagnostics"]["literature_trace"]) == 200
    assert record["diagnostics"]["literature_trace"][-1]["http_status"] == 403
    assert "private source text" not in log.read_text()
    assert "openalex-secret" not in log.read_text()
    assert "https://example.test" not in log.read_text()


def test_failure_log_is_private_even_when_existing_file_is_permissive(tmp_path, monkeypatch):
    log = tmp_path / "research_errors.jsonl"
    log.write_text("", encoding="utf-8")
    log.chmod(0o666)
    monkeypatch.setattr(autonomous_research, "DIAGNOSTIC_LOG", log)
    assert autonomous_research._record_research_failure(LLMError("Invalid response"), 1, 1)
    assert log.stat().st_mode & 0o777 == 0o600


def test_concurrent_failures_append_complete_json_lines(tmp_path, monkeypatch):
    log = tmp_path / "research_errors.jsonl"
    monkeypatch.setattr(autonomous_research, "DIAGNOSTIC_LOG", log)
    with ThreadPoolExecutor(max_workers=8) as pool:
        incidents = list(pool.map(lambda number: autonomous_research._record_research_failure(
            LLMError("Invalid response"), number, 1), range(20)))
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(records) == len(set(incidents)) == 20
    assert {record["incident_id"] for record in records} == set(incidents)


def test_llm_call_metadata_has_timing_and_no_untrusted_usage_or_secret():
    class Provider:
        provider_name = "fake"
        model = "fake"

        def complete(self, request):
            return LLMResponse("{}", "session-secret", "model/session-secret",
                               {"total_tokens": 12, "raw_response": "private", "prompt_tokens": -3})

    client = LLMClient(provider=Provider(), configuration=LLMConfiguration(api_key="session-secret"))
    client.complete(LLMRequest(messages=(LLMMessage(role="user", content="private prompt"),),
                               model_role="research_step"))
    metadata, = client.call_metadata
    assert metadata["role"] == "research_step" and metadata["elapsed_seconds"] >= 0
    assert metadata["usage"] == {"total_tokens": 12}
    assert "session-secret" not in json.dumps(metadata)
    assert "private" not in json.dumps(metadata)
    assert "session-secret" not in json.dumps(client.metadata())

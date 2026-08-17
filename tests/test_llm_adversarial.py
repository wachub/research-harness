import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from pydantic import BaseModel

from src import db
from src.cli import main
from src.llm import LLMClient, LLMError, LLMMessage, LLMRequest, OpenAICompatibleProvider


class Answer(BaseModel):
    answer: str


def _chat_response(content, *, model="mock-model", usage=True):
    payload = {
        "model": model,
        "choices": [{"message": {"content": content}}],
    }
    if usage:
        payload["usage"] = {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
    return json.dumps(payload).encode("utf-8")


@contextmanager
def _mock_chat_server(responses):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - required HTTP handler name.
            size = int(self.headers.get("Content-Length", "0"))
            self.server.requests.append(
                {
                    "path": self.path,
                    "authorization": self.headers.get("Authorization"),
                    "payload": json.loads(self.rfile.read(size).decode("utf-8")),
                }
            )
            response = self.server.responses.pop(0)
            if response.get("delay"):
                time.sleep(response["delay"])
            self.send_response(response.get("status", 200))
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(response.get("body", b""))

        def log_message(self, format, *args):  # noqa: A003 - suppress test-server logging.
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.responses = list(responses)
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", server
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def _provider(base_url, *, max_attempts=3):
    return OpenAICompatibleProvider(
        provider_name="openai-compatible",
        model="test-model",
        api_key="test-secret-key",
        base_url=base_url,
        max_attempts=max_attempts,
        retry_delay_seconds=0,
    )


def _request():
    return LLMRequest(messages=(LLMMessage(role="user", content="tiny prompt"),), json_mode=True)


def test_openai_compatible_http_path_handles_fenced_json_delays_and_missing_usage():
    with _mock_chat_server(
        [{"delay": 0.01, "body": _chat_response("```json\n{\"answer\": \"ok\"}\n```", usage=False)}]
    ) as (base_url, server):
        client = LLMClient(provider=_provider(base_url))
        answer = client.complete_json(_request(), Answer)

    assert answer.answer == "ok"
    assert server.requests[0]["path"] == "/v1/chat/completions"
    assert server.requests[0]["authorization"] == "Bearer test-secret-key"
    assert server.requests[0]["payload"]["response_format"] == {"type": "json_object"}
    assert client.metadata()["usage"] == {}
    assert client.metadata()["model"] == "mock-model"


@pytest.mark.parametrize(
    ("status", "expected_attempts"),
    [(400, 1), (401, 1), (429, 3), (500, 3)],
)
def test_http_errors_are_safe_and_retry_only_retryable_statuses(status, expected_attempts):
    with _mock_chat_server(
        [{"status": status, "body": b'{"error":"test-secret-key"}'}] * expected_attempts
    ) as (base_url, server):
        with pytest.raises(LLMError) as exc_info:
            _provider(base_url).complete(_request())

    assert str(status) in str(exc_info.value)
    assert "test-secret-key" not in str(exc_info.value)
    assert len(server.requests) == expected_attempts


def test_retryable_failure_can_recover_on_a_later_http_attempt():
    with _mock_chat_server(
        [
            {"status": 429, "body": b'{"error":"rate limited"}'},
            {"body": _chat_response('{"answer":"recovered"}')},
        ]
    ) as (base_url, server):
        answer = LLMClient(provider=_provider(base_url)).complete_json(_request(), Answer)

    assert answer.answer == "recovered"
    assert len(server.requests) == 2


def test_network_timeout_and_connection_failure_exhaust_retries_safely(monkeypatch):
    def timeout(*args, **kwargs):
        raise TimeoutError("test-secret-key")

    monkeypatch.setattr("src.llm.urllib.request.urlopen", timeout)
    with pytest.raises(LLMError) as exc_info:
        _provider("http://127.0.0.1:1/v1").complete(_request())

    assert "TimeoutError" in str(exc_info.value)
    assert "test-secret-key" not in str(exc_info.value)


@pytest.mark.parametrize(
    "body",
    [
        b"not-json",
        b"[]",
        b'{"choices": []}',
        _chat_response(""),
    ],
)
def test_malformed_provider_responses_raise_safe_llm_errors(body):
    with _mock_chat_server([{"body": body}] * 3) as (base_url, _server):
        with pytest.raises(LLMError):
            _provider(base_url).complete(_request())


def test_invalid_base_url_is_normalized_to_a_safe_llm_error():
    with pytest.raises(LLMError):
        _provider("not a URL", max_attempts=1).complete(_request())


def _plan_response():
    return {
        "interpreted_goal": "Investigate the supplied goal.",
        "relevant_existing_state": [
            {"kind": "research_cluster", "object_id": 1, "relevance": "Selected cluster."}
        ],
        "recommended_cluster_id": 1,
        "cluster_rationale": "Selected cluster is relevant.",
        "proposed_subquestions": [
            {
                "question": "Which assumptions matter?",
                "rationale": "The goal needs a precise information model.",
                "dependencies_or_assumptions": ["local observation"],
                "uncertainty_note": "Question only.",
            }
        ],
        "proposed_conjectures": [
            {
                "statement": "A restricted fragment may have finite-memory strategies.",
                "rationale": "Candidate only.",
                "dependencies_or_assumptions": ["causal order"],
                "uncertainty_note": "Unverified.",
            }
        ],
        "proposed_literature_tasks": [],
        "proposed_experiments": [],
        "uncertainty_note": "The plan establishes no result.",
    }


def _configure_loopback_provider(monkeypatch, base_url):
    monkeypatch.setenv("LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("LLM_MODEL", "loopback-model")
    monkeypatch.setenv("LLM_API_KEY", "test-secret-key")
    monkeypatch.setenv("LLM_BASE_URL", base_url)


def test_cli_remote_extraction_and_planning_keep_all_proposals_pending(monkeypatch, tmp_path, capsys):
    extraction = {
        "candidates": [
            {"entry_type": "theorem", "payload": {"statement": "Remote candidate theorem."}}
        ]
    }
    with _mock_chat_server(
        [
            {"body": _chat_response(json.dumps(extraction))},
            {"body": _chat_response(json.dumps(_plan_response()))},
        ]
    ) as (base_url, server):
        _configure_loopback_provider(monkeypatch, base_url)
        db_path = tmp_path / "research.db"
        extract_exit = main(
            ["--db", str(db_path), "extract-from-text", "--text", "source", "--llm"]
        )
        extract_output = capsys.readouterr().out
        plan_exit = main(
            [
                "--db",
                str(db_path),
                "plan-research",
                "--goal",
                "Investigate ATS safety.",
                "--cluster-id",
                "1",
                "--llm",
                "--save-pending",
            ]
        )
        plan_output = capsys.readouterr().out

    with db.get_connection(db_path) as connection:
        pending = db.list_pending_entries(connection)
        theorems = db.list_theorems(connection)
        conjectures = db.list_conjectures(connection)
        problems = db.list_open_problems(connection)
    assert extract_exit == plan_exit == 0
    assert "Inserted pending entries (provider)" in extract_output
    assert "Saved pending proposals" in plan_output
    assert "test-secret-key" not in extract_output + plan_output
    assert len(server.requests) == 2
    assert [entry.entry_type for entry in pending] == ["theorem", "conjecture_seed", "open_problem"]
    assert all(entry.status == "pending" for entry in pending)
    assert theorems == conjectures == problems == []


def test_cli_malformed_remote_plan_and_invalid_cluster_write_nothing(monkeypatch, tmp_path, capsys):
    with _mock_chat_server([{"body": b"not-json"}] * 3) as (base_url, server):
        _configure_loopback_provider(monkeypatch, base_url)
        db_path = tmp_path / "research.db"
        db.initialize_database(db_path)
        exit_code = main(
            ["--db", str(db_path), "plan-research", "--goal", "goal", "--llm", "--save-pending"]
        )
        output = capsys.readouterr().out
        with pytest.raises(SystemExit, match="cluster 999 does not exist"):
            main(
                ["--db", str(db_path), "plan-research", "--goal", "goal", "--cluster-id", "999", "--llm"]
            )

    with db.get_connection(db_path) as connection:
        pending = db.list_pending_entries(connection)
    assert exit_code == 2
    assert "did not produce a valid plan" in output
    assert "test-secret-key" not in output
    assert len(server.requests) >= 1
    assert pending == []


def test_cli_invalid_typed_extraction_candidate_writes_nothing(monkeypatch, tmp_path, capsys):
    invalid_candidate = {
        "candidates": [
            {
                "entry_type": "concept",
                "payload": {"name": "Candidate", "concept_type": "model", "unexpected": "field"},
            }
        ]
    }
    with _mock_chat_server([{"body": _chat_response(json.dumps(invalid_candidate))}]) as (base_url, _server):
        _configure_loopback_provider(monkeypatch, base_url)
        db_path = tmp_path / "research.db"
        db.initialize_database(db_path)
        exit_code = main(
            ["--db", str(db_path), "extract-from-text", "--text", "source", "--llm"]
        )
        output = capsys.readouterr().out

    with db.get_connection(db_path) as connection:
        pending = db.list_pending_entries(connection)
    assert exit_code == 2
    assert "LLM extraction failed validation" in output
    assert "test-secret-key" not in output
    assert pending == []

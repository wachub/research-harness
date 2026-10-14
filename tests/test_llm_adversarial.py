import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from pydantic import BaseModel

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


def test_optional_json_schema_is_sent_through_shared_provider():
    schema = Answer.model_json_schema()
    with _mock_chat_server([{"body": _chat_response('{"answer":"ok"}')}]) as (base_url, server):
        client = LLMClient(provider=_provider(base_url))
        answer = client.complete_json(
            LLMRequest(messages=(LLMMessage(role="user", content="tiny prompt"),), response_schema=schema),
            Answer,
        )

    assert answer.answer == "ok"
    assert server.requests[0]["payload"]["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "structured_response", "schema": schema},
    }

import json
import io
import urllib.error
from email.message import Message

import pytest
from pydantic import BaseModel

from src.extract import LLMClient as ExtractionLLMClient
from src.llm import LLMClient, LLMConfiguration, LLMError, LLMMessage, LLMRequest, LLMResponse, OpenAICompatibleProvider
from src.literature import demo


class Answer(BaseModel):
    answer: str


class FakeProvider:
    provider_name = "fake"
    model = "fake-model"

    def complete(self, request: LLMRequest) -> LLMResponse:
        assert request.json_mode
        return LLMResponse(
            content='{"answer": "stored evidence only"}',
            provider=self.provider_name,
            model=self.model,
            usage={"total_tokens": 12},
        )


def test_configuration_defaults_to_offline_placeholder(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)

    configuration = LLMConfiguration.from_environment()

    assert configuration.provider == "placeholder"
    assert configuration.model == "placeholder-model"
    assert not configuration.remote_enabled


def test_openai_compatible_configuration_uses_environment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("LLM_MODEL", "deepseek-chat")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com/v1/")

    configuration = LLMConfiguration.from_environment()

    assert configuration.remote_enabled
    assert configuration.model == "deepseek-chat"
    assert configuration.base_url == "https://api.deepseek.com/v1"


def test_client_validates_structured_response_and_exposes_safe_metadata():
    client = LLMClient(provider=FakeProvider())

    answer = client.complete_json(
        LLMRequest(messages=(LLMMessage(role="user", content="respond"),), json_mode=True),
        Answer,
    )

    assert answer.answer == "stored evidence only"
    assert client.metadata() == {
        "provider": "fake",
        "model": "fake-model",
        "usage": {"total_tokens": 12},
        "remote_available": True,
    }


class FakeExtractionProvider:
    provider_name = "fake"
    model = "fake-model"

    def __init__(self, candidates: list[dict]):
        self.candidates = candidates

    def complete(self, request: LLMRequest) -> LLMResponse:
        assert request.json_mode
        return LLMResponse(
            content=json.dumps({"candidates": self.candidates}),
            provider=self.provider_name,
            model=self.model,
        )


def test_remote_extraction_validates_payload_before_pending_entry_creation():
    client = ExtractionLLMClient(
        provider=FakeExtractionProvider(
            [
                {
                    "entry_type": "theorem",
                    "payload": {
                        "statement": "Two-process ATS safety is decidable under the stated assumptions.",
                        "theorem_type": "decidability",
                        "model_family": "ATS games",
                        "objective_family": "safety",
                    },
                }
            ]
        ),
        use_configured_provider=True,
    )

    entries = client.extract_pending_entries("Theorem source text")

    assert client.used_remote
    assert not client.dry_run
    assert entries[0].entry_type == "theorem"
    assert entries[0].payload["statement"].startswith("Two-process ATS")
    assert "dry-run extraction" not in entries[0].warnings


def test_invalid_remote_extraction_payload_is_rejected_before_queueing():
    client = ExtractionLLMClient(
        provider=FakeExtractionProvider(
            [{"entry_type": "theorem", "payload": {"statement": ""}}]
        ),
        use_configured_provider=True,
    )

    with pytest.raises(LLMError, match="failed validation"):
        client.extract_pending_entries("Theorem source text")


def test_memo_organizer_uses_shared_client_and_rejects_unknown_citations(monkeypatch):
    class FakeMemoClient:
        available = True

        def complete_json(self, request: LLMRequest, response_model):
            assert request.json_mode
            return response_model.model_validate(
                {
                    "memo": (
                        "# Research Memo\n\n## Known Results\n"
                        "- Stored result (evidence 7, source=results/approved/known.json)\n\n"
                        "## Conjecture\n- conjecture: needs verification\n"
                    )
                }
            )

    monkeypatch.setattr(demo, "LLMClient", FakeMemoClient)
    records = [
        {
            "evidence": [
                {"evidence_id": 7, "source_path": "results/approved/known.json"}
            ]
        }
    ]

    memo = demo._organize_memo_with_llm("question", "deterministic memo", records)

    assert memo is not None
    assert "evidence 7" in memo
    assert not demo._uses_only_stored_citations(
        "## Known Results\n- bad (evidence 8)\n## Conjecture",
        records,
    )


def test_validation_error_exposes_fields_without_model_values():
    from src.schemas import StrictBase

    class RequiredProposal(StrictBase):
        kind: str
        title: str

    class BadProvider:
        provider_name = "fake"
        model = "fake"

        def complete(self, request):
            return LLMResponse(
                content=json.dumps({"research_activities": [{"secret": "test-secret-key"}]}),
                provider="fake", model="fake",
            )

    with pytest.raises(LLMError) as caught:
        LLMClient(provider=BadProvider()).complete_json(
            LLMRequest(messages=(LLMMessage(role="user", content="test"),)),
            RequiredProposal,
        )

    assert {tuple(item.values()) for item in caught.value.details} == {
        ("kind", "missing"),
        ("title", "missing"),
        ("research_activities", "extra_forbidden"),
    }
    assert "test-secret-key" not in str(caught.value)


@pytest.mark.parametrize("status,code,error_type,category", [
    (429, "insufficient_quota", None, "quota_exhausted"),
    (429, "credit_balance_exhausted", None, "quota_exhausted"),
    (429, "project_spend_limit_exceeded", None, "quota_exhausted"),
    (429, None, "insufficient_quota", "quota_exhausted"),
    (429, "rate_limit_exceeded", None, "rate_limited"),
    (429, "slow_down", "rate_limit_error", "rate_limited"),
    (429, None, None, "rate_limit_or_quota"),
    (401, "invalid_api_key", None, "authentication"),
    (403, None, None, "permission_denied"),
    (404, "model_not_found", None, "not_found"),
    (400, "unsupported_parameter", "invalid_request_error", "invalid_request"),
    (503, "server_is_overloaded", None, "provider_unavailable"),
])
def test_http_diagnostics_are_actionable_and_do_not_echo_content(monkeypatch, status, code, error_type, category):
    headers = Message()
    headers["x-request-id"] = "req-test-123"
    headers["Retry-After"] = "30"
    secret = "session-only-secret"
    body = json.dumps({"error": {"code": code, "type": error_type,
        "message": f"Bearer {secret}; private research text", "param": "temperature"}}).encode()

    def fail(*args, **kwargs):
        raise urllib.error.HTTPError("https://example.test/v1", status, "ignored", headers, io.BytesIO(body))

    monkeypatch.setattr("src.llm.urllib.request.urlopen", fail)
    config = LLMConfiguration("openai-compatible", "default-model", secret,
                              "https://user:password@example.test/v1?secret=hidden")
    client = LLMClient(configuration=config, model_overrides={"unit_selection": "routed-model"})
    client.provider._retry_delay_seconds = 0
    with pytest.raises(LLMError) as caught:
        client.complete(LLMRequest(messages=(LLMMessage("user", "private research text"),), model_role="unit_selection"))
    details = caught.value.diagnostics
    assert details["category"] == category
    assert details["http_status"] == status
    assert details["model"] == "routed-model"
    assert details["model_role"] == "unit_selection"
    assert details["endpoint_host"] == "example.test"
    assert details["request_id"] == "req-test-123"
    assert details["retry_after_seconds"] == 30
    assert details["attempts"] == (3 if status == 429 or status >= 500 else 1)
    assert len(details["attempt_history"]) == details["attempts"]
    assert details["elapsed_seconds"] >= 0
    serialized = json.dumps(details) + str(caught.value)
    for private in (secret, "private research text", "password", "hidden", "Bearer"):
        assert private not in serialized


@pytest.mark.parametrize("body", [b"not JSON", b"[]", b'{"error": []}', b'{"error":{"code":{},"type":[]}}',
                                  b'{"error":{"code":"private_text","type":"private_text","message":"private_text"}}'])
def test_unknown_http_body_is_not_logged(monkeypatch, body):
    def fail(*args, **kwargs):
        raise urllib.error.HTTPError("https://example.test", 429, "private_text", {}, io.BytesIO(body))
    monkeypatch.setattr("src.llm.urllib.request.urlopen", fail)
    provider = OpenAICompatibleProvider(provider_name="test", model="test", api_key="secret",
                                        base_url="https://example.test", max_attempts=1)
    with pytest.raises(LLMError) as caught:
        provider.complete(LLMRequest(messages=()))
    assert caught.value.diagnostics["category"] == "rate_limit_or_quota"
    assert "private_text" not in json.dumps(caught.value.diagnostics)


def test_http_metadata_redacts_session_key_and_bounds_body_read(monkeypatch):
    secret = "session-secret"
    class BoundedBody(io.BytesIO):
        def read(self, size=-1):
            assert size == 16_384
            return super().read(size)
    def fail(*args, **kwargs):
        raise urllib.error.HTTPError("https://example.test", 401, "ignored",
                                     {"x-request-id": f"req-{secret}", "Retry-After": "nan"},
                                     BoundedBody(b"x" * 20_000))
    monkeypatch.setattr("src.llm.urllib.request.urlopen", fail)
    client = LLMClient(configuration=LLMConfiguration("openai", secret, secret, "https://example.test"))
    with pytest.raises(LLMError) as caught:
        client.complete(LLMRequest(messages=()))
    assert secret not in json.dumps(caught.value.diagnostics)
    assert caught.value.diagnostics["request_id"] == "[redacted]"
    assert "retry_after_seconds" not in caught.value.diagnostics


def test_validation_diagnostic_does_not_expose_session_key_as_field():
    from src.schemas import StrictBase
    class Expected(StrictBase):
        answer: str
    class Provider:
        provider_name = "fake"
        model = "fake"
        def complete(self, request):
            return LLMResponse('{"session_secret":"private result"}', "fake", "fake")
    client = LLMClient(provider=Provider(), configuration=LLMConfiguration(api_key="session_secret"))
    with pytest.raises(LLMError) as caught:
        client.complete_json(LLMRequest(messages=(), model_role="proof_attempt"), Expected)
    assert "session_secret" not in str(caught.value) + json.dumps(caught.value.details)
    assert caught.value.diagnostics["category"] == "schema_validation"
    assert caught.value.diagnostics["model_role"] == "proof_attempt"


@pytest.mark.parametrize("retry_after,expected", [("not-a-date", False), ("-1", False),
                                                  ("Wed, 01 Jan 2098 00:00:00 GMT", True)])
def test_retry_after_date_or_malformed_header(monkeypatch, retry_after, expected):
    def fail(*args, **kwargs):
        raise urllib.error.HTTPError("https://example.test", 429, "ignored",
                                     {"Retry-After": retry_after}, io.BytesIO(b"{}"))
    monkeypatch.setattr("src.llm.urllib.request.urlopen", fail)
    provider = OpenAICompatibleProvider(provider_name="test", model="test", api_key="secret",
                                        base_url="https://example.test", max_attempts=1)
    with pytest.raises(LLMError) as caught:
        provider.complete(LLMRequest(messages=()))
    assert ("retry_after_seconds" in caught.value.diagnostics) is expected

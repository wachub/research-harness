import json

import pytest
from pydantic import BaseModel

from src.extract import LLMClient as ExtractionLLMClient
from src.llm import LLMClient, LLMConfiguration, LLMError, LLMMessage, LLMRequest, LLMResponse
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

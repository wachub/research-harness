"""Legacy seed memo safety checks unrelated to the retired review queue."""

import json

from src.llm import LLMClient, LLMError, LLMRequest, LLMResponse
from src.literature import demo


class StaticProvider:
    provider_name = "test-provider"
    model = "test-model"

    def __init__(self, content: str | None = None, error: Exception | None = None):
        self.content = content
        self.error = error

    def complete(self, request: LLMRequest) -> LLMResponse:
        if self.error is not None:
            raise self.error
        return LLMResponse(content=self.content or "", provider=self.provider_name, model=self.model)


def test_memo_falls_back_for_unknown_or_missing_evidence_references(monkeypatch):
    records = [{"evidence": [{"evidence_id": 7, "source_path": "results/approved/known.json"}]}]
    for memo in (
        "# Research Memo\n## Known Results\n- Claim (evidence 8)\n## Conjecture",
        "# Research Memo\n## Known Results\n- Claim (results/approved/invented.json)\n## Conjecture",
        "# Research Memo\n## Known Results\n- Uncited claim\n## Conjecture",
    ):
        client = LLMClient(provider=StaticProvider(json.dumps({"memo": memo})))
        monkeypatch.setattr(demo, "LLMClient", lambda: client)
        assert demo._organize_memo_with_llm("question", "deterministic", records) is None


def test_memo_malformed_or_failed_remote_response_falls_back_deterministically(monkeypatch, tmp_path):
    db_path = tmp_path / "research.db"
    seed = demo.run_research_demo(dry_run=True, db_path=db_path, output_path=tmp_path / "map.md")
    for index, provider in enumerate((StaticProvider("not json"), StaticProvider(error=LLMError("network failure")))):
        monkeypatch.setattr(demo, "LLMClient", lambda: LLMClient(provider=provider))
        memo_path = tmp_path / f"memo_{index}.md"
        result = demo.write_research_memo(
            seed.topic_id, "What is known?", db_path=db_path, output_path=memo_path, use_llm=True,
        )
        assert not result.used_llm
        assert "results/approved/" in memo_path.read_text(encoding="utf-8")

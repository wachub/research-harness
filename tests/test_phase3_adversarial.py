import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic import BaseModel, ConfigDict, Field

from src import db
from src.extract import LLMClient as ExtractionLLMClient
from src.extract import extract_from_text
from src.llm import LLMClient, LLMConfiguration, LLMError, LLMRequest, LLMResponse
from src.literature import demo
from src.research_planning import plan_research, save_plan_as_pending
from src.schemas import Conjecture, EvidenceSpan, ExperimentRun, OpenProblem, Paper, Theorem


class StaticProvider:
    provider_name = "test-provider"
    model = "test-model"

    def __init__(self, content: str | None = None, error: Exception | None = None):
        self.content = content
        self.error = error
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return LLMResponse(content=self.content or "", provider=self.provider_name, model=self.model)


class StrictAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)


def _plan_payload(*, reference_kind="research_cluster", reference_id=1, subquestions=None, conjectures=None):
    return {
        "interpreted_goal": "Investigate the supplied ATS safety objective.",
        "relevant_existing_state": [
            {"kind": reference_kind, "object_id": reference_id, "relevance": "Stored context."}
        ],
        "recommended_cluster_id": 1,
        "cluster_rationale": "The requested cluster is relevant.",
        "proposed_subquestions": subquestions
        if subquestions is not None
        else [
            {
                "question": "Which observation assumptions are required?",
                "rationale": "The objective does not fix an information model.",
                "dependencies_or_assumptions": ["local observations"],
                "uncertainty_note": "Question only; not an established open problem.",
            }
        ],
        "proposed_conjectures": conjectures
        if conjectures is not None
        else [
            {
                "statement": "A bounded causal-order fragment may admit finite-memory strategies.",
                "rationale": "Candidate for human review only.",
                "dependencies_or_assumptions": ["bounded causal order"],
                "uncertainty_note": "Unverified conjecture.",
            }
        ],
        "proposed_literature_tasks": [],
        "proposed_experiments": [],
        "uncertainty_note": "Nothing in this plan is established or executed.",
    }


def _remote_extraction_client(content: str) -> ExtractionLLMClient:
    return ExtractionLLMClient(provider=StaticProvider(content), use_configured_provider=True)


def _pending_count(db_path: Path) -> int:
    with db.get_connection(db_path) as connection:
        return len(db.list_pending_entries(connection, status=None))


def test_no_remote_or_unsupported_provider_cannot_plan_or_write(monkeypatch, tmp_path):
    db_path = tmp_path / "research.db"
    db.initialize_database(db_path)
    monkeypatch.setenv("LLM_PROVIDER", "unsupported")
    monkeypatch.setenv("LLM_API_KEY", "not-used")

    result = plan_research("Investigate ATS safety.", use_llm=True, db_path=db_path)

    assert not result.available
    assert "No remote LLM provider" in result.message
    assert _pending_count(db_path) == 0
    assert not LLMClient(configuration=LLMConfiguration("unsupported", "m", "key", "http://x")).available


def test_remote_extraction_multiple_valid_candidates_remain_pending_only(tmp_path):
    db_path = tmp_path / "research.db"
    db.initialize_database(db_path)
    client = _remote_extraction_client(
        json.dumps(
            {
                "candidates": [
                    {"entry_type": "theorem", "payload": {"statement": "A candidate theorem."}},
                    {"entry_type": "concept", "payload": {"name": "Candidate concept", "concept_type": "model"}},
                ]
            }
        )
    )

    entry_ids = extract_from_text("source text", db_path=db_path, client=client)

    with db.get_connection(db_path) as connection:
        pending = db.list_pending_entries(connection)
        assert db.list_theorems(connection) == []
    assert len(entry_ids) == 2
    assert {entry.entry_type for entry in pending} == {"theorem", "concept"}
    assert all(entry.status == "pending" for entry in pending)


@pytest.mark.parametrize(
    "content",
    [
        "This is prose, not JSON.",
        json.dumps({"candidates": [{"entry_type": "unsupported", "payload": {}}]}),
    ],
)
def test_invalid_remote_extraction_envelopes_fall_back_without_durable_writes(content, tmp_path):
    db_path = tmp_path / "research.db"
    db.initialize_database(db_path)

    entry_ids = extract_from_text("plain source text", db_path=db_path, client=_remote_extraction_client(content))

    with db.get_connection(db_path) as connection:
        pending = db.list_pending_entries(connection)
        assert db.list_theorems(connection) == []
        assert db.list_conjectures(connection) == []
    assert len(entry_ids) == 1
    assert pending[0].entry_type == "paper_summary"
    assert "LLM provider fallback; deterministic extraction used" in pending[0].warnings


def test_invalid_remote_extraction_payload_is_rejected_without_queueing(tmp_path):
    db_path = tmp_path / "research.db"
    db.initialize_database(db_path)
    client = _remote_extraction_client(
        json.dumps(
            {"candidates": [{"entry_type": "concept", "payload": {"name": "x", "concept_type": "model", "extra": "no"}}]}
        )
    )

    with pytest.raises(LLMError, match="failed validation"):
        extract_from_text("plain source text", db_path=db_path, client=client)

    assert _pending_count(db_path) == 0


def test_partial_invalid_remote_extraction_writes_nothing(tmp_path):
    db_path = tmp_path / "research.db"
    db.initialize_database(db_path)
    client = _remote_extraction_client(
        json.dumps(
            {
                "candidates": [
                    {"entry_type": "theorem", "payload": {"statement": "Valid first candidate."}},
                    {"entry_type": "theorem", "payload": {"statement": ""}},
                ]
            }
        )
    )

    with pytest.raises(LLMError, match="failed validation"):
        extract_from_text("source", db_path=db_path, client=client)

    assert _pending_count(db_path) == 0


def test_remote_extraction_does_not_reuse_stale_remote_state_after_a_failure(tmp_path):
    class SequenceProvider(StaticProvider):
        def __init__(self):
            super().__init__()
            self.responses = [
                json.dumps({"candidates": [{"entry_type": "theorem", "payload": {"statement": "Remote candidate."}}]}),
                "not json",
            ]

        def complete(self, request):
            self.requests.append(request)
            return LLMResponse(content=self.responses.pop(0), provider=self.provider_name, model=self.model)

    db_path = tmp_path / "research.db"
    db.initialize_database(db_path)
    client = ExtractionLLMClient(provider=SequenceProvider(), use_configured_provider=True)

    first_ids = extract_from_text("first source", db_path=db_path, client=client)
    second_ids = extract_from_text("second source", db_path=db_path, client=client)

    with db.get_connection(db_path) as connection:
        entries = db.list_pending_entries(connection)
    assert len(first_ids) == len(second_ids) == 1
    assert entries[0].entry_type == "theorem"
    assert entries[1].entry_type == "paper_summary"
    assert "LLM provider fallback; deterministic extraction used" in entries[1].warnings


@pytest.mark.parametrize(
    "content",
    [
        "{}",
        '{"answer": ""}',
        '{"answer": "ok", "extra": "not allowed"}',
    ],
)
def test_structured_output_missing_invalid_or_extra_fields_are_safe(content):
    with pytest.raises(LLMError, match="failed validation"):
        LLMClient(provider=StaticProvider(content)).complete_json(
            LLMRequest(messages=(), json_mode=True),
            StrictAnswer,
        )


def test_empty_and_long_offline_extraction_remain_safe_and_pending(tmp_path):
    db_path = tmp_path / "research.db"
    db.initialize_database(db_path)

    empty_ids = extract_from_text("", db_path=db_path)
    long_ids = extract_from_text("x" * 200_000, db_path=db_path)

    with db.get_connection(db_path) as connection:
        pending = db.list_pending_entries(connection)
    assert len(empty_ids) == len(long_ids) == 1
    assert pending[0].payload["summary"] == "Empty extraction input"
    assert len(pending[1].payload["summary"]) == 1000
    assert all(entry.status == "pending" for entry in pending)


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
    for provider in (StaticProvider("not json"), StaticProvider(error=LLMError("network failure"))):
        monkeypatch.setattr(demo, "LLMClient", lambda: LLMClient(provider=provider))
        memo_path = tmp_path / f"memo_{len(provider.requests)}.md"
        result = demo.write_research_memo(
            seed.topic_id,
            "What is known?",
            db_path=db_path,
            output_path=memo_path,
            use_llm=True,
        )
        assert not result.used_llm
        assert "results/approved/" in memo_path.read_text(encoding="utf-8")


def test_planner_rejects_unknown_and_out_of_scope_references_without_writes(tmp_path):
    db_path = tmp_path / "research.db"
    db.initialize_database(db_path)
    for reference_id in (999, 2):
        result = plan_research(
            "Investigate ATS safety.",
            cluster_id=1,
            use_llm=True,
            db_path=db_path,
            client=LLMClient(provider=StaticProvider(json.dumps(_plan_payload(reference_id=reference_id)))),
        )
        assert not result.available
        assert _pending_count(db_path) == 0


def test_planner_handles_zero_and_oversized_or_unsupported_proposals_without_writes(tmp_path):
    db_path = tmp_path / "research.db"
    db.initialize_database(db_path)
    zero_plan = _plan_payload(subquestions=[], conjectures=[])
    zero_result = plan_research(
        "Investigate ATS safety.",
        cluster_id=1,
        use_llm=True,
        db_path=db_path,
        client=LLMClient(provider=StaticProvider(json.dumps(zero_plan))),
    )
    assert zero_result.available
    assert save_plan_as_pending(zero_result, "goal", db_path) == []

    oversized = _plan_payload(subquestions=[_plan_payload()["proposed_subquestions"][0]] * 6)
    unsupported = _plan_payload()
    unsupported["unexpected_proposal_type"] = [{"task": "do something"}]
    for payload in (oversized, unsupported):
        result = plan_research(
            "Investigate ATS safety.",
            cluster_id=1,
            use_llm=True,
            db_path=db_path,
            client=LLMClient(provider=StaticProvider(json.dumps(payload))),
        )
        assert not result.available
    assert _pending_count(db_path) == 0


def test_planner_rejects_theorem_ids_from_another_cluster(tmp_path):
    db_path = tmp_path / "research.db"
    db.initialize_database(db_path)
    with db.get_connection(db_path) as connection:
        cluster_two = db.insert_cluster(connection, db.ResearchCluster(name="Separate cluster"))
        theorem_id = db.insert_theorem(connection, Theorem(statement="Other cluster theorem.", cluster_id=cluster_two))
    payload = _plan_payload(reference_kind="theorem", reference_id=theorem_id)

    result = plan_research(
        "Investigate ATS safety.",
        cluster_id=1,
        use_llm=True,
        db_path=db_path,
        client=LLMClient(provider=StaticProvider(json.dumps(payload))),
    )

    assert not result.available
    assert _pending_count(db_path) == 0


def test_invalid_goal_and_cluster_cannot_reach_provider_or_write(tmp_path):
    db_path = tmp_path / "research.db"
    db.initialize_database(db_path)
    provider = StaticProvider(json.dumps(_plan_payload()))
    for goal in ("", "   "):
        with pytest.raises(ValidationError):
            plan_research(goal, use_llm=True, db_path=db_path, client=LLMClient(provider=provider))
    with pytest.raises(ValueError, match="cluster 999"):
        plan_research("goal", cluster_id=999, use_llm=True, db_path=db_path, client=LLMClient(provider=provider))
    assert provider.requests == []
    assert _pending_count(db_path) == 0


def test_failed_planner_request_and_repeated_pending_saves_never_create_durable_state(tmp_path):
    db_path = tmp_path / "research.db"
    db.initialize_database(db_path)
    failed = plan_research(
        "goal",
        use_llm=True,
        db_path=db_path,
        client=LLMClient(provider=StaticProvider(error=LLMError("network failure"))),
    )
    assert not failed.available
    assert _pending_count(db_path) == 0

    result = plan_research(
        "goal",
        cluster_id=1,
        use_llm=True,
        db_path=db_path,
        client=LLMClient(provider=StaticProvider(json.dumps(_plan_payload()))),
    )
    first = save_plan_as_pending(result, "goal", db_path)
    second = save_plan_as_pending(result, "goal", db_path)
    with db.get_connection(db_path) as connection:
        assert db.list_conjectures(connection) == []
        assert db.list_open_problems(connection) == []
    assert len(first) == len(second) == 2
    assert _pending_count(db_path) == 4


def test_planner_context_includes_populated_cluster_state_without_executing_commands(tmp_path):
    db_path = tmp_path / "research.db"
    db.initialize_database(db_path)
    with db.get_connection(db_path) as connection:
        paper_id = db.insert_paper(connection, Paper(title="Paper", authors=["A"], year=2026, cluster_id=1))
        theorem_id = db.insert_theorem(connection, Theorem(statement="Stored theorem.", cluster_id=1))
        db.insert_conjecture(connection, Conjecture(statement="Stored conjecture.", cluster_id=1))
        db.insert_open_problem(connection, OpenProblem(title="Stored gap", statement="Stored gap.", cluster_id=1))
        db.insert_evidence_span(connection, EvidenceSpan(paper_id=paper_id, entry_type="theorem", entry_id=theorem_id, quote_or_summary="Stored evidence."))
        db.insert_experiment_run(connection, ExperimentRun(cluster_id=1, experiment_type="smoke", result_summary="not executed by planner"))
    dependency = "$(touch should-not-run); SELECT * FROM theorems"
    payload = _plan_payload(
        conjectures=[
            {
                "statement": "Candidate only.",
                "rationale": "Text only.",
                "dependencies_or_assumptions": [dependency],
                "uncertainty_note": "Unverified.",
            }
        ]
    )
    provider = StaticProvider(json.dumps(payload))

    result = plan_research(
        "x" * 100_000,
        cluster_id=1,
        use_llm=True,
        db_path=db_path,
        client=LLMClient(provider=provider),
    )
    pending_ids = save_plan_as_pending(result, "goal", db_path)

    prompt = json.loads(provider.requests[0].messages[-1].content)
    assert prompt["research_state"]["theorems"][0]["id"] == theorem_id
    assert prompt["research_state"]["evidence"][0]["paper_id"] == paper_id
    assert prompt["research_state"]["experiment_runs"][0]["summary"] == "not executed by planner"
    assert len(pending_ids) == 2
    assert not (tmp_path / "should-not-run").exists()


def test_pending_plan_writes_are_atomic_when_an_insert_fails(monkeypatch, tmp_path):
    db_path = tmp_path / "research.db"
    db.initialize_database(db_path)
    result = plan_research(
        "Investigate ATS safety.",
        cluster_id=1,
        use_llm=True,
        db_path=db_path,
        client=LLMClient(provider=StaticProvider(json.dumps(_plan_payload()))),
    )
    original_insert = db.insert_pending_entry
    calls = 0

    def fail_second_insert(connection, entry):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("forced insert failure")
        return original_insert(connection, entry)

    monkeypatch.setattr(db, "insert_pending_entry", fail_second_insert)
    with pytest.raises(RuntimeError, match="forced insert failure"):
        save_plan_as_pending(result, "goal", db_path)

    assert _pending_count(db_path) == 0

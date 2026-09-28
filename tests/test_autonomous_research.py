from __future__ import annotations

import json
import urllib.error

import pytest

from src import autonomous_research, db
from src.autonomous_research import AutonomousResearch
from src.literature.discovery import FullTextPaper, LiteratureError, SourceClaim, _openalex_works, discover_full_text, extract_source_claims
from src.llm import LLMClient, LLMResponse
from src.schemas import CodeArtifact, ResearchTask, ResearchUnit


@pytest.fixture(autouse=True)
def _isolate_diagnostic_log(tmp_path, monkeypatch):
    monkeypatch.setattr(autonomous_research, "DIAGNOSTIC_LOG", tmp_path / "research_errors.jsonl")


class SequenceProvider:
    provider_name = "fake"
    model = "fake"

    def __init__(self, responses):
        self.responses = list(responses)

    def complete(self, request):
        return LLMResponse(content=json.dumps(self.responses.pop(0)), provider="fake", model="fake")


def _proposal(kind, title, statement=None, references=None):
    return {
        "kind": kind, "title": title, "purpose": "Investigate finite memory",
        "outcome": "A tentative observation", "rationale": "Relevant to task",
        "uncertainty_note": "Not verified", "references": references or [],
        "statement": statement,
    }


def test_two_steps_create_linear_finished_units_and_draft_result(tmp_path):
    path = tmp_path / "research.db"
    with db.get_connection(path) as connection:
        db.create_tables(connection)
        task_id = db.insert_task(connection, ResearchTask(name="Goal", description="Finite-memory safety"))
    provider = SequenceProvider([
        _proposal("analysis", "Inspect state"),
        _proposal("partial_result", "Partial observation", "A finite-memory candidate exists"),
    ])
    result = AutonomousResearch(LLMClient(provider=provider), path).run(task_id, steps=2)
    assert result.status == "completed"
    assert len(result.unit_ids) == 2
    with db.get_connection(path) as connection:
        units = db.list_research_units(connection, task_id)
        results = db.list_derived_results(connection, task_id)
    assert [unit.status for unit in units] == ["finished", "finished"]
    assert units[1].parent_unit_id == units[0].unit_id
    assert len(results) == 1 and results[0].status == "draft"


def test_forced_literature_rejects_non_literature_plan_without_writes(tmp_path):
    path = tmp_path / "research.db"
    with db.get_connection(path) as connection:
        db.create_tables(connection)
        task_id = db.insert_task(connection, ResearchTask(name="Goal"))
        unit_id = db.insert_research_unit(connection, ResearchUnit(
            task_id=task_id, kind="analysis", title="Start", purpose="Start",
            status="finished",
        ))
    provider = SequenceProvider([_proposal("analysis", "Ignore instruction")])
    result = AutonomousResearch(LLMClient(provider=provider), path).run(
        task_id, unit_id=unit_id, literature_from_unit=True,
    )
    assert result.status == "stopped"
    with db.get_connection(path) as connection:
        assert len(db.list_research_units(connection, task_id)) == 1




def test_literature_step_stores_page_linked_unverified_claim_without_pdf_blob(tmp_path, monkeypatch):
    path = tmp_path / "research.db"
    with db.get_connection(path) as connection:
        db.create_tables(connection)
        task_id = db.insert_task(connection, ResearchTask(name="Goal", description="Survey safety"))
    paper = FullTextPaper(
        "Open paper", ("Ada",), 2025, "https://example.org/paper",
        "https://example.org/paper.pdf", "Journal",
        ("Theorem 1. Safety is decidable.",),
    )
    claim = SourceClaim(
        kind="theorem", title="Safety", statement="Safety is decidable",
        page=1, quote="Theorem 1. Safety is decidable.",
    )
    monkeypatch.setattr("src.autonomous_research.discover_full_text", lambda *_: paper)
    monkeypatch.setattr("src.autonomous_research.extract_source_claims", lambda *_: [claim])
    provider = SequenceProvider([_proposal("literature_review", "Survey source")])
    result = AutonomousResearch(LLMClient(provider=provider), path).run(task_id)
    assert result.status == "completed"
    with db.get_connection(path) as connection:
        papers = db.list_papers(connection, task_id)
        theorems = db.list_theorems(connection, task_id)
        evidence = db.list_evidence_spans(connection)
        links = db.list_research_unit_links(connection, result.unit_ids[0])
    assert papers[0].pdf_path is None
    assert theorems[0].confidence == "pending"
    assert evidence[0].page_start == 1 and evidence[0].confidence == "pending"
    assert {link.object_type for link in links} == {"paper", "theorem", "evidence"}


def test_failed_unit_creation_rolls_back_literature_claims(tmp_path, monkeypatch):
    path = tmp_path / "research.db"
    with db.get_connection(path) as connection:
        db.create_tables(connection)
        task_id = db.insert_task(connection, ResearchTask(name="Goal", description="Survey safety"))
    paper = FullTextPaper("Open paper", ("Ada",), 2025, "https://example.org/p",
                          None, None, ("A supported source statement.",))
    monkeypatch.setattr("src.autonomous_research.discover_full_text", lambda *_: paper)
    monkeypatch.setattr("src.autonomous_research.extract_source_claims", lambda *_: [])
    monkeypatch.setattr(db, "insert_research_unit", lambda *_: (_ for _ in ()).throw(ValueError("unit write failed")))
    provider = SequenceProvider([_proposal("literature_review", "Survey source")])
    result = AutonomousResearch(LLMClient(provider=provider), path).run(task_id)
    assert result.status == "stopped"
    with db.get_connection(path) as connection:
        assert db.list_papers(connection, task_id) == []
        assert db.list_research_units(connection, task_id) == []





def test_bounded_experiment_is_linked_to_finished_unit_without_claim_promotion(tmp_path):
    path = tmp_path / "research.db"
    with db.get_connection(path) as connection:
        db.create_tables(connection)
        task_id = db.insert_task(connection, ResearchTask(name="Goal"))
        artifact_id = db.insert_code_artifact(connection, CodeArtifact(
            name="Trusted checker", path="src/experiments/ats_brute_solver.py",
            artifact_type="checker", task_id=task_id, status="tested",
        ))
    proposal = _proposal("bounded_experiment", "Check tiny case")
    proposal["experiment"] = {
        "artifact_id": artifact_id, "kind": "ATS", "process_count": 2,
        "states_per_process": 2, "depth": 0, "seed": 1,
    }
    provider = SequenceProvider([proposal])
    result = AutonomousResearch(LLMClient(provider=provider), path).run(task_id)
    assert result.status == "completed"
    with db.get_connection(path) as connection:
        runs = db.list_experiment_runs(connection, task_id=task_id)
        links = db.list_research_unit_links(connection, result.unit_ids[0])
        assert db.get_research_unit(connection, result.unit_ids[0]).status == "finished"
        assert db.list_theorems(connection, task_id=task_id) == []
    assert len(runs) == 1
    assert any(link.object_type == "experiment_run" and link.object_id == runs[0].run_id for link in links)


def test_failed_unit_creation_rolls_back_bounded_experiment(tmp_path, monkeypatch):
    path = tmp_path / "research.db"
    with db.get_connection(path) as connection:
        db.create_tables(connection)
        task_id = db.insert_task(connection, ResearchTask(name="Goal"))
        artifact_id = db.insert_code_artifact(connection, CodeArtifact(
            name="Trusted checker", path="src/experiments/ats_brute_solver.py",
            artifact_type="checker", task_id=task_id, status="tested",
        ))
    proposal = _proposal("bounded_experiment", "Check tiny case")
    proposal["experiment"] = {"artifact_id": artifact_id, "depth": 0}
    monkeypatch.setattr(db, "insert_research_unit", lambda *_: (_ for _ in ()).throw(ValueError("unit write failed")))
    result = AutonomousResearch(LLMClient(provider=SequenceProvider([proposal])), path).run(task_id)
    assert result.status == "stopped"
    with db.get_connection(path) as connection:
        assert db.list_experiment_runs(connection, task_id=task_id) == []

def test_discovery_rejects_irrelevant_search_result_before_download(monkeypatch):
    monkeypatch.setattr("src.literature.discovery._openalex_works", lambda *_: [
        {"display_name": "Unrelated paper", "publication_year": 2025,
         "best_oa_location": {"pdf_url": "https://example.org/file.pdf"}},
    ])
    monkeypatch.setattr(
        "src.literature.discovery._download_pdf",
        lambda *_: (_ for _ in ()).throw(AssertionError("irrelevant PDF downloaded")),
    )
    client = LLMClient(provider=SequenceProvider([{"index": None, "rationale": "Not related"}]))
    try:
        discover_full_text("ATS global safety", client=client)
    except LiteratureError:
        pass
    else:
        raise AssertionError("irrelevant paper was accepted")



def test_openalex_rate_limit_message_never_leaks_key(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "test-secret")
    def rate_limited(*_args, **_kwargs):
        raise urllib.error.HTTPError("https://api.openalex.org/works", 429, "rate limit", {}, None)
    monkeypatch.setattr("src.literature.discovery.urllib.request.urlopen", rate_limited)
    try:
        _openalex_works("safety games")
    except LiteratureError as exc:
        assert "rate limit" in str(exc)
        assert "test-secret" not in str(exc)
    else:
        raise AssertionError("rate limit was not reported")

def test_exact_page_quote_required_for_source_claims():
    paper = FullTextPaper("P", ("A",), 2025, "https://example.org/p", None, None,
                          ("The main theorem says safety is decidable.",))
    claim = {"kind": "theorem", "title": "Safety", "statement": "Safety is decidable",
             "page": 1, "quote": "The main theorem says safety is decidable.", "proof_note": None}
    provider = SequenceProvider([{"claims": [claim]}])
    assert extract_source_claims(LLMClient(provider=provider), paper, "safety")[0].page == 1
    claim["quote"] = "An invented supporting quotation"
    provider = SequenceProvider([{"claims": [claim]}])
    try:
        extract_source_claims(LLMClient(provider=provider), paper, "safety")
    except LiteratureError:
        pass
    else:
        raise AssertionError("unsupported quotation was accepted")


def test_failed_step_writes_private_sanitized_diagnostic(tmp_path, monkeypatch):
    from src import autonomous_research

    log_path = tmp_path / "research_errors.jsonl"
    monkeypatch.setattr(autonomous_research, "DIAGNOSTIC_LOG", log_path)
    monkeypatch.setenv("LLM_API_KEY", "test-secret-key")
    db_path = tmp_path / "research.db"
    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        task_id = db.insert_task(connection, ResearchTask(name="Goal"))

    provider = SequenceProvider([
        {"research_activities": [{"secret": "test-secret-key"}]}
    ])
    result = AutonomousResearch(LLMClient(provider=provider), db_path).run(task_id)

    assert result.status == "stopped"
    assert result.diagnostic_id
    assert result.error_type == "LLMError"
    assert any(item["field"] == "research_activities" for item in result.error_details)
    record = json.loads(log_path.read_text(encoding="utf-8"))
    assert record["incident_id"] == result.diagnostic_id
    assert record["step_number"] == 1
    assert record["validation"]
    assert record["stack"]
    assert "test-secret-key" not in log_path.read_text(encoding="utf-8")
    assert log_path.stat().st_mode & 0o777 == 0o600
    with db.get_connection(db_path) as connection:
        assert db.list_research_units(connection, task_id) == []


def test_proposal_prompt_specifies_exact_single_object_shape():
    class PromptProvider:
        provider_name = "fake"
        model = "fake"

        def complete(self, request):
            instructions = request.messages[0].content
            assert "exactly one JSON object" in instructions
            assert "not a list or wrapper" in instructions
            assert set(request.response_schema["required"]) >= {
                "kind", "title", "purpose", "outcome", "rationale", "uncertainty_note",
            }
            for field in ("kind", "title", "purpose", "outcome", "rationale", "uncertainty_note", "references"):
                assert field in instructions
            return LLMResponse(
                content=json.dumps(_proposal("analysis", "One bounded step")),
                provider="fake", model="fake",
            )

    proposal = AutonomousResearch(LLMClient(provider=PromptProvider()))._propose(
        {}, "Investigate a goal", None, [], False,
    )
    assert proposal.kind == "analysis"


def test_literature_selection_accepts_index_without_unused_rationale(monkeypatch):
    work = {
        "display_name": "Finite-memory games", "publication_year": 2025,
        "best_oa_location": {"pdf_url": "https://example.org/paper.pdf", "source": {"display_name": "Venue"}},
        "locations": [], "authorships": [{"author": {"display_name": "A"}}],
        "id": "https://openalex.org/W1",
    }
    monkeypatch.setattr("src.literature.discovery._openalex_works", lambda *_: [work])
    monkeypatch.setattr("src.literature.discovery._download_pdf", lambda *_: b"%PDF")
    monkeypatch.setattr("src.literature.discovery.extract_pdf_pages", lambda *_: ("Page text",))

    paper = discover_full_text(
        "finite-memory games", client=LLMClient(provider=SequenceProvider([{"index": 0}])),
    )

    assert paper.title == "Finite-memory games"
    assert paper.pages == ("Page text",)

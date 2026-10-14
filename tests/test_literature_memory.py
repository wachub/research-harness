"""Literature memory regressions: real isolated PostgreSQL, no external APIs."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from src import autonomous_research, db
from src.autonomous_research import AutonomousResearch, StepProposal
from src.literature.discovery import (
    FindingReference, FullTextPaper, LiteratureError, SourceClaim, SourceQuote, extract_source_claims,
)
from src.literature.identity import same_paper, source_identifier
from src.literature.memory import compare_with_existing, completed_ingestion, review_scope, store_distillation
from src.llm import LLMClient
from src.research_context import load_controller_context
from src.schemas import Paper, ResearchTask, ResearchUnit, ResearchUnitLink
from tests.test_autonomous_research import SequenceProvider, _proposal

QUOTE = "Every finite arena has a finite strategy."
PROOF = "The proof constructs the finite product."


@pytest.fixture
def state(tmp_path):
    path = tmp_path / "memory.db"
    with db.get_connection(path) as connection:
        db.create_tables(connection)
        task_id = db.insert_task(connection, ResearchTask(name="Safety", description="Finite memory"))
    return path, task_id


def source(**changes):
    return replace(FullTextPaper(
        title="Finite strategies", authors=("A. Author",), year=2025,
        url="https://doi.org/10.1234/finite", pdf_url=None, venue=None,
        pages=(QUOTE + " " + PROOF + " Unrelated full text must not enter the database.",),
        diagnostics={"review_scope": "initial"},
    ), **changes)


def claim(**changes):
    return SourceClaim(**dict({"kind": "theorem", "title": "Finite strategy",
        "statement": "Every finite arena has a finite strategy.", "page": 1, "quote": QUOTE}, **changes))


def ingest(state, paper, claims):
    path, task_id = state
    with db.get_connection(path) as connection:
        return store_distillation(connection, task_id, paper, claims)


def test_repeat_content_reuses_checkpoint_without_duplicate_rows(state):
    first, _ = ingest(state, source(), [claim()])
    second, outcome = ingest(state, source(), [])
    assert {(k, i) for k, i, _ in first} == {(k, i) for k, i, _ in second}
    assert all(r == "uses" for _, _, r in second)
    assert "No new findings" in outcome
    with db.get_connection(state[0]) as connection:
        assert len(db.list_papers(connection)) == len(db.list_theorems(connection)) == 1
        assert len(db.list_evidence_spans(connection)) == 1
        assert connection.execute("SELECT COUNT(*) AS n FROM research_events WHERE event_type = 'literature_distilled'").fetchone()["n"] == 1


def test_batch_duplicates_merge_finding_but_keep_distinct_support(state):
    links, _ = ingest(state, source(), [claim(), claim(quote=PROOF, proof_note="Product construction")])
    with db.get_connection(state[0]) as connection:
        assert len(db.list_theorems(connection)) == 1
        assert len(db.list_evidence_spans(connection)) == 2
    assert len([link for link in links if link[0] == "theorem"]) == 1
    assert next(link[2] for link in links if link[0] == "theorem") == "produces"


@pytest.mark.parametrize("change", [
    {"statement": "Every finite arena has a Finite strategy."},
    {"assumptions": ["Perfect information"]},
    {"limitations": ["Two processes only"]},
    {"statement": "Every arena with five processes has a finite strategy."},
])
def test_mathematical_differences_are_not_merged(state, change):
    ingest(state, source(), [claim(), claim(**change)])
    with db.get_connection(state[0]) as connection:
        assert len(db.list_theorems(connection)) == 2


def test_whitespace_duplicate_new_source_adds_citation_not_theorem(state):
    ingest(state, source(), [claim()])
    links, _ = ingest(state, source(title="Another proof", url="https://doi.org/10.1234/other"),
                      [claim(statement="Every  finite arena has a finite strategy.")])
    with db.get_connection(state[0]) as connection:
        assert len(db.list_papers(connection)) == 2
        assert len(db.list_theorems(connection)) == 1
        assert len(db.list_evidence_spans(connection)) == 2
    assert next(link[2] for link in links if link[0] == "theorem") == "uses"


def test_additional_distillation_annotates_existing_evidence(state):
    ingest(state, source(), [claim()])
    ingest(state, source(diagnostics={"review_scope": "proof"}), [claim(proof_note="Product construction")])
    with db.get_connection(state[0]) as connection:
        evidence = db.list_evidence_spans(connection)
        assert len(evidence) == 1
        assert json.loads(evidence[0].notes)["additional_distillations"][0]["proof_note"] == "Product construction"


def test_distillation_keeps_ideas_conditions_proof_and_only_excerpts(state):
    ingest(state, source(), [claim(kind="idea", assumptions=["Finite arena"],
        limitations=["No infinite arenas"], proof_note="Use a finite product",
        relevance="Candidate technique", additional_evidence=[SourceQuote(page=1, quote=PROOF)])])
    with db.get_connection(state[0]) as connection:
        result, = db.list_derived_results(connection)
        assert result.status == "draft" and result.proof_sketch == "Use a finite product"
        note = json.loads(result.notes)["distillation"]
        assert note["assumptions"] == ["Finite arena"] and note["limitations"] == ["No infinite arenas"]
        evidence = db.list_evidence_spans(connection)
        assert {row.quote_or_summary for row in evidence} == {QUOTE, PROOF}
        assert all(row.confidence == "pending" for row in evidence)
        assert all(json.loads(row.notes)["source_hash"] for row in evidence)
        assert "Unrelated full text" not in json.dumps([row.model_dump(mode="json") for row in evidence])
    summary = load_controller_context(state[1], state[0]).summary
    assert summary["derived_results"][0]["proof_sketch"] == "Use a finite product"
    assert summary["derived_results"][0]["distillation"]["assumptions"] == ["Finite arena"]
    assert summary["derived_results"][0]["distillation"]["limitations"] == ["No infinite arenas"]


def test_cross_task_reuse_keeps_ownership_and_confidence(state):
    links, _ = ingest(state, source(), [claim()])
    with db.get_connection(state[0]) as connection:
        theorem_id = next(i for k, i, _ in links if k == "theorem")
        connection.execute("UPDATE theorems SET confidence = 'verified' WHERE theorem_id = ?", (theorem_id,))
        other = db.insert_task(connection, ResearchTask(name="Related goal"))
    reused, _ = ingest((state[0], other), source(diagnostics={"review_scope": "other"}), [claim()])
    with db.get_connection(state[0]) as connection:
        unit = db.insert_research_unit(connection, ResearchUnit(task_id=other, title="Reuse", kind="literature_review", purpose="Compare"))
        for kind, ident, relation in reused:
            assert relation == "uses"
            db.link_research_unit(connection, ResearchUnitLink(unit_id=unit, object_type=kind, object_id=ident, relation=relation))
        theorem, = db.list_theorems(connection, other)
        assert theorem.task_id == state[1] and theorem.confidence == "verified"
        assert len(db.list_theorems(connection)) == 1


def test_extensions_and_possible_duplicates_remain_separate_linked_claims(state):
    links, _ = ingest(state, source(), [claim()])
    ident = next(i for k, i, _ in links if k == "theorem")
    for relation in ("extends", "possible_duplicate"):
        links, _ = ingest(state, source(diagnostics={"review_scope": relation}), [claim(
            statement=f"Another statement: {relation}", relationship=relation,
            related_finding=FindingReference(kind="theorem", object_id=ident), relationship_note="Compare scope")])
        assert ("theorem", ident, "uses") in links
    with db.get_connection(state[0]) as connection:
        assert len(db.list_theorems(connection)) == 3
        assert all(t.confidence == "pending" for t in db.list_theorems(connection))


@pytest.mark.parametrize("invalid", [
    {"additional_evidence": [SourceQuote(page=2, quote=PROOF)]},
    {"relationship": "extends", "related_finding": FindingReference(kind="theorem", object_id=999)},
])
def test_bad_evidence_or_relationship_rolls_back_entire_intake(state, invalid):
    with pytest.raises(ValueError):
        ingest(state, source(), [claim(), claim(**invalid)])
    with db.get_connection(state[0]) as connection:
        assert not db.list_papers(connection) and not db.list_theorems(connection)
        assert not db.list_evidence_spans(connection)


def test_existing_paper_notes_preserved_and_empty_extraction_not_checkpointed(state):
    with db.get_connection(state[0]) as connection:
        db.insert_paper(connection, Paper(title=source().title, authors=list(source().authors), year=2025,
            url=source().url, notes="Researcher annotation", task_id=state[1]))
    ingest(state, source(), [])
    with db.get_connection(state[0]) as connection:
        assert completed_ingestion(connection, source(), "initial") is None
        assert db.list_papers(connection)[0].notes == "Researcher annotation"
    ingest(state, source(), [claim()])
    with db.get_connection(state[0]) as connection:
        paper, = db.list_papers(connection)
        assert paper.notes.startswith("Full text read;") and "Researcher annotation" in paper.notes


def test_content_and_scope_changes_invalidate_checkpoint(state):
    ingest(state, source(), [claim()])
    with db.get_connection(state[0]) as connection:
        assert completed_ingestion(connection, source(), "initial")
        assert completed_ingestion(connection, source(), "different purpose") is None
        assert completed_ingestion(connection, source(pages=(QUOTE + " Revised proof.",)), "initial") is None


def test_same_source_concurrent_intake_is_atomic_and_idempotent(state):
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda _: ingest(state, source(), [claim()]), range(2)))
    with db.get_connection(state[0]) as connection:
        assert len(db.list_papers(connection)) == len(db.list_theorems(connection)) == len(db.list_evidence_spans(connection)) == 1


def test_source_identifier_aliases_and_versions():
    base = Paper(title="Title", authors=["A"], year=2025, url="https://doi.org/10.1234/ABC")
    assert same_paper(base, base.model_copy(update={"url": "https://dx.doi.org/10.1234/abc"}))
    assert not same_paper(base, base.model_copy(update={"url": "https://doi.org/10.1234/different"}))
    assert source_identifier("https://arxiv.org/pdf/2501.12345v2.pdf") == "arxiv:2501.12345v2"
    arxiv = base.model_copy(update={"url": "https://arxiv.org/abs/2501.12345v2"})
    assert same_paper(arxiv, arxiv.model_copy(update={"url": "https://arxiv.org/pdf/2501.12345v2.pdf"}))
    assert not same_paper(arxiv, arxiv.model_copy(update={"url": "https://arxiv.org/abs/2501.12345v1"}))
    assert not same_paper(arxiv, base)


def test_repeat_survey_skips_extraction_then_rechecks_revised_text(state, monkeypatch):
    proposal = StepProposal(**_proposal("literature_review", "Read"))
    scope = review_scope("Finite memory", proposal.purpose)
    paper = source(diagnostics={"review_scope": scope})
    ingest(state, paper, [claim()])
    monkeypatch.setattr(autonomous_research, "discover_full_text", lambda *args: paper)
    calls = []
    monkeypatch.setattr(autonomous_research, "extract_source_claims", lambda *args: calls.append(args) or [claim()])
    runner = AutonomousResearch(LLMClient(provider=SequenceProvider([])), state[0])
    assert runner._survey(state[1], proposal)[1] == [] and not calls
    paper = source(pages=(QUOTE + " Revised proof.",))
    assert runner._survey(state[1], proposal)[1] and len(calls) == 1
    assert calls[0][3][0]["statement"] == claim().statement


def test_extraction_rejects_invented_relationship_and_extra_quote():
    for bad in [claim(relationship="extends", related_finding=FindingReference(kind="theorem", object_id=42)),
                claim(additional_evidence=[SourceQuote(page=1, quote="This was never in the source")])]:
        client = LLMClient(provider=SequenceProvider([{"claims": [bad.model_dump()]}]))
        with pytest.raises(LiteratureError):
            extract_source_claims(client, source(), "Find safety results", [])


@pytest.mark.parametrize("verdict,expected", [
    ("equivalent", "possible_duplicate"),
    ("extends", "extends"),
    ("contradicts", "contradicts"),
    ("different", "new"),
])
def test_model_compares_near_matches_without_silently_merging(state, verdict, expected):
    ingest(state, source(), [claim(assumptions=["Finite game"])])
    candidate = claim(title="Finite memory", statement="Finite games admit strategies with finite memory.",
                      assumptions=["Finite game"])
    provider = SequenceProvider([{"comparisons": [{
        "claim_index": 0, "verdict": verdict,
        "object_id": 1 if verdict != "different" else None,
        "rationale": "Compare the game class and strategy memory",
        "material_difference": "Memory bound may differ",
    }]}])
    checked = source(diagnostics={"review_scope": "second"})
    with db.get_connection(state[0]) as connection:
        compared = compare_with_existing(connection, state[1], LLMClient(provider=provider), checked, [candidate])
    assert compared[0].relationship == expected
    if expected != "new":
        assert compared[0].related_finding.object_id == 1
        assert "unverified" in compared[0].relationship_note
    links, _ = ingest(state, checked, compared)
    with db.get_connection(state[0]) as connection:
        assert len(db.list_theorems(connection)) == 2
        notes = json.loads(db.list_theorems(connection)[1].notes)["distillation"]
        assert notes["relationship"] == expected
        assert db.list_theorems(connection)[0].confidence == "pending"
    if expected != "new":
        assert ("theorem", 1, "uses") in links


def test_untrusted_comparison_id_is_ignored_and_claim_remains(state):
    ingest(state, source(), [claim()])
    candidate = claim(statement="Finite memory suffices on a finite arena")
    client = LLMClient(provider=SequenceProvider([{"comparisons": [{
        "claim_index": 0, "verdict": "equivalent", "object_id": 999,
        "rationale": "Looks alike",
    }]}]))
    checked = source(diagnostics={"review_scope": "other"})
    with db.get_connection(state[0]) as connection:
        compared = compare_with_existing(connection, state[1], client, checked, [candidate])
    assert compared == [candidate]
    assert checked.diagnostics["semantic_comparison"]["status"] == "unavailable"
    ingest(state, checked, compared)
    with db.get_connection(state[0]) as connection:
        assert len(db.list_theorems(connection)) == 2


def test_exact_match_skips_comparison_call(state):
    ingest(state, source(), [claim()])
    client = LLMClient(provider=SequenceProvider([]))
    with db.get_connection(state[0]) as connection:
        assert compare_with_existing(connection, state[1], client, source(), [claim()]) == [claim()]
    assert client.call_metadata == []


def test_grounded_extraction_relationship_is_not_overwritten(state):
    ingest(state, source(), [claim()])
    related = claim(statement="A stronger finite strategy statement", relationship="extends",
                    related_finding=FindingReference(kind="theorem", object_id=1),
                    relationship_note="Additional condition from the source")
    client = LLMClient(provider=SequenceProvider([]))
    with db.get_connection(state[0]) as connection:
        checked = compare_with_existing(connection, state[1], client, source(), [related])
    assert checked == [related]
    assert client.call_metadata == []


def test_long_extraction_compares_every_claim_in_bounded_batches(state):
    ingest(state, source(), [claim()])
    claims = [claim(statement=f"Finite strategy formulation {index}") for index in range(7)]
    responses = [{"comparisons": [{"claim_index": index, "verdict": "equivalent",
                                  "object_id": 1, "rationale": "Same finite game"}
                                 for index in indexes]}
                 for indexes in (range(6), range(6, 7))]
    client = LLMClient(provider=SequenceProvider(responses))
    checked = source(diagnostics={"review_scope": "bulk"})
    with db.get_connection(state[0]) as connection:
        compared = compare_with_existing(connection, state[1], client, checked, claims)
    assert len(client.call_metadata) == 2
    assert all(item.relationship == "possible_duplicate" for item in compared)
    assert checked.diagnostics["semantic_comparison"]["flagged"] == 7

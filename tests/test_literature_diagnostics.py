"""Offline regression cases for observable retrieval and grounded extraction."""

import io
import json
import urllib.error

import pytest

from src.literature import discovery as d
from src.llm import LLMClient, LLMError, LLMResponse


def work():
    return {"display_name": "Finite safety", "publication_year": 2025,
            "authorships": [{"author": {"display_name": "Ada"}}],
            "best_oa_location": {"is_oa": True, "pdf_url": "https://example.org/paper?key=private-key",
                                 "landing_page_url": "https://example.org/article"}}


@pytest.mark.parametrize("status", [401, 403, 429, 503])
def test_openalex_failures_identify_search_not_model(monkeypatch, status):
    monkeypatch.setenv("OPENALEX_API_KEY", "private-key")
    def fail(*args, **kwargs):
        raise urllib.error.HTTPError("https://api.openalex.org?api_key=private-key", status,
                                     "private response", {}, io.BytesIO(b"private body"))
    monkeypatch.setattr(d.urllib.request, "urlopen", fail)
    with pytest.raises(d.LiteratureError) as caught:
        d.discover_full_text("private research question")
    data = caught.value.diagnostics
    assert data["literature_stage"] == "search"
    assert data["http_status"] == status
    assert data["literature_trace"][0]["status"] == "failed"
    assert "private" not in str(caught.value) + json.dumps(data)


def test_empty_search_is_not_reported_as_pdf_failure(monkeypatch):
    monkeypatch.setattr(d, "_openalex_works", lambda *_: [])
    with pytest.raises(d.LiteratureError) as caught:
        d.discover_full_text("safety")
    assert caught.value.diagnostics["category"] == "no_search_candidates"
    assert caught.value.diagnostics["literature_trace"][0]["results_returned"] == 0


def test_selection_failure_and_pdf_failure_survive_successful_html_fallback(monkeypatch):
    class Provider:
        provider_name = "fake"
        model = "fake-model"
        def complete(self, request):
            raise LLMError("HTTP 429", diagnostics={"http_status": 429, "category": "quota_exhausted"})
    monkeypatch.setattr(d, "_openalex_works", lambda *_: [work()])
    def blocked(url):
        raise urllib.error.HTTPError(url, 403, "private body", {}, io.BytesIO())
    monkeypatch.setattr(d, "_download_pdf", blocked)
    monkeypatch.setattr(d, "_download_html", lambda *_: b"<article>" + b"Real article text. " * 40 + b"</article>")
    paper = d.discover_full_text("finite safety", client=LLMClient(provider=Provider()))
    trace = paper.diagnostics["literature_trace"]
    assert paper.text_format == "html"
    assert any(e["stage"] == "selection" and e["http_status"] == 429 for e in trace)
    assert any(e["stage"] == "selection_fallback" and e["selected_index"] == 0 for e in trace)
    assert any(e["stage"] == "download_pdf" and e["http_status"] == 403 for e in trace)
    assert trace[-1]["stage"] == "parse_html" and trace[-1]["status"] == "completed"
    assert "private" not in json.dumps(trace)
    assert "Finite safety" not in json.dumps(trace)


def test_parse_failure_distinct_from_download_failure(monkeypatch):
    candidate = work()
    candidate["best_oa_location"].pop("landing_page_url")
    monkeypatch.setattr(d, "_openalex_works", lambda *_: [candidate])
    monkeypatch.setattr(d, "_download_pdf", lambda *_: b"%PDF-invalid")
    with pytest.raises(d.LiteratureError) as caught:
        d.discover_full_text("finite safety")
    trace = caught.value.diagnostics["literature_trace"]
    assert any(e["stage"] == "download_pdf" and e["status"] == "completed" for e in trace)
    assert any(e["stage"] == "parse_pdf" and e["error_code"] == "pdf_parse_error" for e in trace)


def test_missing_local_pdf_is_recorded_before_empty_search(monkeypatch, tmp_path):
    monkeypatch.setattr(d, "_openalex_works", lambda *_: [])
    with pytest.raises(d.LiteratureError) as caught:
        d.discover_full_text("safety", [("Private title", str(tmp_path / "missing.pdf"), ("Ada",), 2025, "")])
    trace = caught.value.diagnostics["literature_trace"]
    assert trace[0]["error_code"] == "local_file_missing"
    assert trace[1]["stage"] == "search"
    assert str(tmp_path) not in json.dumps(trace)


def test_quote_failure_records_chunk_without_storing_claim_text():
    class Provider:
        provider_name = "fake"
        model = "fake-model"
        def complete(self, request):
            return LLMResponse(json.dumps({"claims": [{"kind": "theorem", "title": "Private title",
                "statement": "Private claim", "quote": "An invented quotation", "page": 1}]}), "fake", "fake-model")
    paper = d.FullTextPaper("Private title", ("Ada",), 2025, "", None, None, ("Original source text",))
    with pytest.raises(d.LiteratureError) as caught:
        d.extract_source_claims(LLMClient(provider=Provider()), paper, "private goal")
    data = caught.value.diagnostics
    assert data["literature_stage"] == "validate_quotes"
    assert data["error_code"] == "unsupported_quote"
    assert data["literature_trace"][0]["claims_returned"] == 1
    assert "Private" not in json.dumps(data) and "quotation" not in json.dumps(data)


def test_bad_openalex_envelope_is_not_confused_with_no_results(monkeypatch):
    monkeypatch.setattr(d.urllib.request, "urlopen", lambda *_args, **_kwargs: io.BytesIO(b'{"error":"private"}'))
    with pytest.raises(d.LiteratureError) as caught:
        d.discover_full_text("safety")
    assert caught.value.diagnostics["error_code"] == "invalid_search_response"
    assert caught.value.diagnostics["literature_stage"] == "search"

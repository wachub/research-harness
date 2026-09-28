"""Bounded open-access full-text discovery and page-grounded extraction."""

from __future__ import annotations

import io
import ipaddress
import json
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from pydantic import Field

from pypdf import PdfReader


from ..llm import LLMClient, LLMMessage, LLMRequest
from ..schemas import StrictBase

MAX_PDF_BYTES = 20_000_000
MAX_TEXT_CHARS = 300_000
MAX_PAGES = 100


class LiteratureError(ValueError):
    """No suitable accessible full text or safe extraction was available."""


@dataclass(frozen=True)
class FullTextPaper:
    title: str
    authors: tuple[str, ...]
    year: int
    url: str
    pdf_url: str | None
    venue: str | None
    pages: tuple[str, ...]
    local_path: str | None = None


class SourceClaim(StrictBase):
    kind: str = Field(pattern="^(theorem|conjecture|open_problem)$")
    title: str = Field(min_length=1, max_length=300)
    statement: str = Field(min_length=1)
    page: int = Field(ge=1)
    quote: str = Field(min_length=12)
    proof_note: str | None = None


class SourceClaims(StrictBase):
    claims: list[SourceClaim] = Field(default_factory=list, max_length=12)


class LiteratureChoice(StrictBase):
    index: int | None = Field(default=None, ge=0)
    rationale: str | None = Field(default=None, min_length=1)


def _public_https_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise LiteratureError("literature download requires a public HTTPS URL")
    try:
        addresses = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as exc:
        raise LiteratureError("literature host could not be resolved") from exc
    if not addresses or any(
        not ipaddress.ip_address(item[4][0]).is_global for item in addresses
    ):
        raise LiteratureError("literature download host is not public")
    return url


class _PublicRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        _public_https_url(newurl)
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def _download_pdf(url: str) -> bytes:
    opener = urllib.request.build_opener(_PublicRedirectHandler)
    request = urllib.request.Request(
        _public_https_url(url), headers={"User-Agent": "research-harness/1.0 (open-access literature)"}
    )
    with opener.open(request, timeout=25) as response:
        data = response.read(MAX_PDF_BYTES + 1)
    if len(data) > MAX_PDF_BYTES or not data.startswith(b"%PDF"):
        raise LiteratureError("PDF is invalid or exceeds the 20 MB limit")
    return data


def extract_pdf_pages(data: bytes) -> tuple[str, ...]:
    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted or len(reader.pages) > MAX_PAGES:
            raise LiteratureError("PDF is encrypted or exceeds the 100-page limit")
        pages_list: list[str] = []
        for page in reader.pages:
            stream = page.get_contents()
            if stream is not None and len(stream.get_data()) > 5_000_000:
                raise LiteratureError("PDF page content exceeds the extraction limit")
            pages_list.append((page.extract_text() or "").strip())
        pages = tuple(pages_list)
    except LiteratureError:
        raise
    except Exception as exc:
        raise LiteratureError("PDF text could not be extracted") from exc
    if not any(pages) or sum(map(len, pages)) > MAX_TEXT_CHARS:
        raise LiteratureError("PDF has no extractable text or exceeds the text limit")
    return pages


def _openalex_works(query: str) -> list[dict]:
    parameters = {"search": query[:300], "per-page": 5, "filter": "is_oa:true"}
    api_key = os.getenv("OPENALEX_API_KEY", "").strip()
    if api_key:
        parameters["api_key"] = api_key
    request = urllib.request.Request(
        f"https://api.openalex.org/works?{urllib.parse.urlencode(parameters)}",
        headers={"User-Agent": "research-harness/1.0 (open-access literature)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read(2_000_001)
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise LiteratureError("OpenAlex rate limit reached; configure OPENALEX_API_KEY or retry later") from exc
        raise
    if len(raw) > 2_000_000:
        raise LiteratureError("literature search response exceeds the size limit")
    payload = json.loads(raw)
    return payload.get("results", []) if isinstance(payload, dict) else []


def discover_full_text(
    query: str, local_pdfs: list[tuple[str, str, tuple[str, ...], int, str]] | None = None,
    excluded: set[tuple[str, int]] | None = None,
    client: LLMClient | None = None,
) -> FullTextPaper:
    """Read one existing local PDF or discover a new openly accessible full text."""

    for title, path, authors, year, url in local_pdfs or []:
        candidate = Path(path)
        try:
            if candidate.is_file() and candidate.stat().st_size <= MAX_PDF_BYTES:
                pages = extract_pdf_pages(candidate.read_bytes())
                return FullTextPaper(title, authors, year, url, None, None, pages, str(candidate))
        except (LiteratureError, OSError):
            continue
    try:
        works = _openalex_works(query)
    except LiteratureError:
        raise
    except (OSError, ValueError, TimeoutError) as exc:
        raise LiteratureError("literature discovery failed") from exc
    works = [
        work for work in works
        if (work.get("display_name"), work.get("publication_year")) not in (excluded or set())
    ]
    if client is not None and works:
        candidates = [
            {"index": index, "title": work.get("display_name"),
             "year": work.get("publication_year"),
             "source": ((work.get("primary_location") or {}).get("source") or {}).get("display_name")}
            for index, work in enumerate(works)
        ]
        choice = client.complete_json(
            LLMRequest(messages=(
                LLMMessage(role="system", content="Select one paper whose title and source are genuinely relevant to the research question; return null index if none. Use only supplied indices."),
                LLMMessage(role="user", content=json.dumps({"question": query, "candidates": candidates})),
            ), json_mode=True),
            LiteratureChoice,
        )
        if choice.index is None or choice.index >= len(works):
            raise LiteratureError("No relevant discovered paper was selected")
        works = [works[choice.index]]
    for work in works:
        locations = [work.get("best_oa_location"), *(work.get("locations") or [])]
        for location in locations:
            pdf_url = location.get("pdf_url") if isinstance(location, dict) else None
            if not isinstance(pdf_url, str):
                continue
            try:
                pages = extract_pdf_pages(_download_pdf(pdf_url))
            except (LiteratureError, OSError, ValueError, TimeoutError):
                continue
            title = work.get("display_name") or ""
            authors = tuple(
                item.get("author", {}).get("display_name", "")
                for item in work.get("authorships", [])
                if isinstance(item, dict)
            )
            year = work.get("publication_year")
            if not title or not any(authors) or not isinstance(year, int):
                continue
            source = location.get("source") or {}
            venue = source.get("display_name") if isinstance(source, dict) else None
            return FullTextPaper(title, tuple(author for author in authors if author), year,
                                 work.get("id") or pdf_url, pdf_url, venue, pages)
    raise LiteratureError("No extractable openly accessible PDF was found; no literature claims were stored")


def extract_source_claims(client: LLMClient, paper: FullTextPaper, query: str) -> list[SourceClaim]:
    """Validate each model-reported quotation against the actual extracted page."""

    claims: list[SourceClaim] = []
    # Each page is read; bounded chunks keep requests smaller and preserve page references.
    pages = list(enumerate(paper.pages, start=1))
    chunks: list[list[tuple[int, str]]] = []
    for page, content in pages:
        if not content:
            continue
        if len(content) > 18000:
            raise LiteratureError(f"page {page} exceeds the extraction context limit")
        if not chunks or sum(len(text) for _, text in chunks[-1]) + len(content) > 18000:
            chunks.append([])
        chunks[-1].append((page, content))
    for chunk in chunks:
        prompt = {
            "paper": {"title": paper.title, "authors": paper.authors, "year": paper.year},
            "research_question": query,
            "pages": [{"page": page, "text": content} for page, content in chunk],
            "instruction": (
                "Extract only relevant explicitly stated theorems, conjectures, or open problems. "
                "Copy a contiguous exact supporting quote from the indicated page. "
                "Summarize any proof as a note, never claim it was verified. "
                "Return JSON with claims; use an empty list if none are supported."
            ),
        }
        result = client.complete_json(
            LLMRequest(messages=(
                LLMMessage(role="system", content="Extract source-attributed, unverified claims from supplied PDF text only."),
                LLMMessage(role="user", content=json.dumps(prompt)),
            ), json_mode=True),
            SourceClaims,
        )
        page_map = dict(chunk)
        for claim in result.claims:
            if claim.page not in page_map or claim.quote not in page_map[claim.page]:
                raise LiteratureError("LLM supplied an unsupported quotation or page")
            claims.append(claim)
    return claims

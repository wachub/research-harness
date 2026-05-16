"""LLM-assisted extraction into the manual approval queue."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is declared in requirements.
    def load_dotenv() -> bool:
        return False

from . import db
from .schemas import PendingEntry


class LLMProvider(Protocol):
    """Provider interface for replaceable LLM backends."""

    def extract_candidates(self, text: str) -> list[dict[str, Any]]:
        """Return candidate entries with entry_type and payload fields."""


@dataclass
class PlaceholderProvider:
    """Deterministic placeholder provider for the MVP.

    The class preserves the provider boundary without making network calls.
    It uses simple textual cues so the rest of the harness can be tested and
    curated exactly as real LLM output would be.
    """

    model: str
    dry_run: bool = True

    def extract_candidates(self, text: str) -> list[dict[str, Any]]:
        candidates = _heuristic_extract(text)
        if candidates:
            return candidates
        title = _short_title(text, "Candidate theorem")
        return [
            {
                "entry_type": "theorem",
                "payload": {
                    "title": title,
                    "statement": text.strip()[:1000] or "Empty extraction input",
                    "assumptions": [],
                    "conclusion": None,
                    "paper_id": None,
                    "tags": ["dry-run" if self.dry_run else "placeholder"],
                },
            }
        ]


class LLMClient:
    """Small facade that reads LLM configuration from environment variables."""

    def __init__(self, provider: LLMProvider | None = None) -> None:
        load_dotenv()
        self.provider_name = os.getenv("LLM_PROVIDER", "placeholder")
        self.model = os.getenv("LLM_MODEL", "placeholder-model")
        self._api_key_present = bool(os.getenv("LLM_API_KEY"))
        self.dry_run = not self._api_key_present
        self.provider = provider or PlaceholderProvider(model=self.model, dry_run=self.dry_run)

    def extract_json(self, text: str) -> str:
        """Return extracted candidates as formatted JSON."""

        candidates = self.provider.extract_candidates(text)
        return json.dumps(candidates, indent=2, sort_keys=True)

    def extract_pending_entries(self, text: str) -> list[PendingEntry]:
        """Extract and validate candidates as pending entries."""

        raw_candidates = json.loads(self.extract_json(text))
        entries: list[PendingEntry] = []
        for candidate in raw_candidates:
            entry_type = candidate.get("entry_type")
            payload = candidate.get("payload", {})
            warnings = list(candidate.get("warnings", []))
            if self.dry_run and "dry-run extraction" not in warnings:
                warnings.append("dry-run extraction")
            entries.append(
                PendingEntry(
                    entry_type=entry_type,
                    payload=payload,
                    source_text=text,
                    warnings=warnings,
                )
            )
        return entries


def extract_from_text(
    text: str,
    db_path: str | Path | None = None,
    client: LLMClient | None = None,
) -> list[int]:
    """Extract candidate entries from text and write them to pending_entries."""

    llm_client = client or LLMClient()
    entries = llm_client.extract_pending_entries(text)
    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        return [db.insert_pending_entry(connection, entry) for entry in entries]


def _heuristic_extract(text: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]
    if not chunks and text.strip():
        chunks = [text.strip()]

    for chunk in chunks:
        lowered = chunk.lower()
        if "open problem" in lowered or "open question" in lowered:
            candidates.append(
                {
                    "entry_type": "open_problem",
                    "payload": {
                        "title": _short_title(chunk, "Open problem"),
                        "statement": _strip_label(chunk),
                        "context": None,
                        "status": "active",
                        "paper_id": None,
                        "tags": _tags_from_text(chunk),
                    },
                }
            )
        elif "reduction" in lowered or "reduces to" in lowered:
            candidates.append(
                {
                    "entry_type": "reduction",
                    "payload": {
                        "title": _short_title(chunk, "Reduction"),
                        "source_problem": "unknown source problem",
                        "target_problem": "unknown target problem",
                        "statement": _strip_label(chunk),
                        "assumptions": [],
                        "paper_id": None,
                        "tags": _tags_from_text(chunk),
                    },
                }
            )
        elif "theorem" in lowered or "lemma" in lowered or "proposition" in lowered:
            candidates.append(
                {
                    "entry_type": "theorem",
                    "payload": {
                        "title": _short_title(chunk, "Theorem"),
                        "statement": _strip_label(chunk),
                        "assumptions": [],
                        "conclusion": None,
                        "paper_id": None,
                        "tags": _tags_from_text(chunk),
                    },
                }
            )
        elif any(token in lowered for token in ("ats", "cdm", "2dm", "control game")) and "model" in lowered:
            candidates.append(
                {
                    "entry_type": "model",
                    "payload": {
                        "name": _short_title(chunk, "Model"),
                        "model_type": _model_type_from_text(chunk),
                        "description": _strip_label(chunk),
                        "data": {},
                        "source_paper_id": None,
                    },
                }
            )
    return candidates


def _strip_label(text: str) -> str:
    return re.sub(
        r"^\s*(theorem|lemma|proposition|reduction|open problem|open question|model)\s*[:.\-]\s*",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )


def _short_title(text: str, fallback: str) -> str:
    cleaned = _strip_label(text).replace("\n", " ").strip()
    if not cleaned:
        return fallback
    sentence = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0]
    return sentence[:80].rstrip(" .") or fallback


def _tags_from_text(text: str) -> list[str]:
    lowered = text.lower()
    tags = []
    for token in ("ats", "cdm", "2dm", "control", "distributed synthesis", "safety"):
        if token in lowered:
            tags.append(token.replace(" ", "_"))
    return tags


def _model_type_from_text(text: str) -> str:
    lowered = text.lower()
    if "ats" in lowered:
        return "ATS"
    if "cdm" in lowered:
        return "CDM"
    if "2dm" in lowered:
        return "2DM"
    if "control" in lowered:
        return "control"
    if "distributed" in lowered:
        return "distributed_synthesis"
    return "other"


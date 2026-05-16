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


EXTRACTION_INSTRUCTIONS = """
Extract candidate research-memory entries as JSON. Allowed entry_type values are:
paper_summary, concept, model, theorem, reduction, open_problem, conjecture_seed.

For theorem-like results, capture exact assumptions, model family, objective type,
number of processes or players, architecture assumptions, information structure,
strategy type, complexity and memory bounds, source location, proof technique, and
whether the result is an upper bound, lower bound, completeness result,
equivalence, characterization, algorithm, decidability, or undecidability result.

Never treat extracted candidates as approved facts. The caller must write them
only to pending_entries for human curation.
""".strip()


class LLMProvider(Protocol):
    """Provider interface for replaceable LLM backends."""

    def extract_candidates(self, text: str, instructions: str) -> list[dict[str, Any]]:
        """Return candidate entries with entry_type and payload fields."""


@dataclass
class PlaceholderProvider:
    """Deterministic placeholder provider for tests and offline dry runs."""

    model: str
    dry_run: bool = True

    def extract_candidates(self, text: str, instructions: str) -> list[dict[str, Any]]:
        candidates = _heuristic_extract(text)
        if candidates:
            return candidates
        return [
            {
                "entry_type": "paper_summary",
                "payload": {
                    "summary": text.strip()[:1000] or "Empty extraction input",
                    "model_families": _model_families(text),
                    "objectives": _objective_families(text),
                    "notes": "dry-run placeholder extraction" if self.dry_run else "placeholder extraction",
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

        candidates = self.provider.extract_candidates(text, EXTRACTION_INSTRUCTIONS)
        return json.dumps(candidates, indent=2, sort_keys=True)

    def extract_pending_entries(self, text: str) -> list[PendingEntry]:
        """Extract and validate candidates as pending entries."""

        raw_candidates = json.loads(self.extract_json(text))
        entries: list[PendingEntry] = []
        for candidate in raw_candidates:
            warnings = list(candidate.get("warnings", []))
            if self.dry_run and "dry-run extraction" not in warnings:
                warnings.append("dry-run extraction")
            entries.append(
                PendingEntry(
                    entry_type=candidate.get("entry_type"),
                    payload=candidate.get("payload", {}),
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
    """Extract candidate entries from text and write them only to pending_entries."""

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
        if "concept:" in lowered or lowered.startswith("concept"):
            candidates.append(_concept_candidate(chunk))
        elif "conjecture" in lowered:
            candidates.append(_conjecture_candidate(chunk))
        elif "open problem" in lowered or "open question" in lowered:
            candidates.append(_open_problem_candidate(chunk))
        elif "reduction" in lowered or "reduces to" in lowered:
            candidates.append(_reduction_candidate(chunk))
        elif any(token in lowered for token in ("theorem", "lemma", "proposition", "decidable", "undecidable", "complete", "hard")):
            candidates.append(_theorem_candidate(chunk))
        elif any(token in lowered for token in ("ats", "cdm", "2dm", "control game", "petri game", "asynchronous automata")) and "model" in lowered:
            candidates.append(_model_candidate(chunk))
        elif "summary" in lowered or "paper" in lowered:
            candidates.append(
                {
                    "entry_type": "paper_summary",
                    "payload": {
                        "summary": _strip_label(chunk),
                        "model_families": _model_families(chunk),
                        "objectives": _objective_families(chunk),
                    },
                }
            )
    return candidates


def _theorem_candidate(text: str) -> dict[str, Any]:
    return {
        "entry_type": "theorem",
        "payload": {
            "title": _short_title(text, "Theorem"),
            "statement": _strip_label(text),
            "theorem_type": _theorem_type(text),
            "model_family": _first_or_none(_model_families(text)),
            "objective_family": _first_or_none(_objective_families(text)),
            "architecture_assumptions": _architecture_assumptions(text),
            "information_assumptions": _information_assumptions(text),
            "strategy_assumptions": _strategy_assumptions(text),
            "process_bound": _process_bound(text),
            "complexity_upper": _complexity_bound(text, upper=True),
            "complexity_lower": _complexity_bound(text, upper=False),
            "memory_upper": _memory_bound(text, upper=True),
            "memory_lower": _memory_bound(text, upper=False),
            "source_paper_id": None,
            "source_location": _source_location(text),
            "proof_technique": _proof_technique(text),
            "confidence": "needs_review",
            "cluster_id": None,
            "notes": None,
            "assumptions": [],
            "conclusion": None,
            "paper_id": None,
            "tags": _tags_from_text(text),
        },
    }


def _concept_candidate(text: str) -> dict[str, Any]:
    name = _strip_label(text)
    return {
        "entry_type": "concept",
        "payload": {
            "name": _short_title(name, "Concept"),
            "concept_type": _concept_type(text),
            "description": name,
            "aliases": [],
            "notes": None,
        },
    }


def _model_candidate(text: str) -> dict[str, Any]:
    return {
        "entry_type": "model",
        "payload": {
            "name": _short_title(text, "Model"),
            "model_type": _model_type_from_text(text),
            "description": _strip_label(text),
            "data": {},
            "source_paper_id": None,
            "cluster_id": None,
        },
    }


def _reduction_candidate(text: str) -> dict[str, Any]:
    return {
        "entry_type": "reduction",
        "payload": {
            "title": _short_title(text, "Reduction"),
            "source_problem": "unknown source problem",
            "target_problem": "unknown target problem",
            "statement": _strip_label(text),
            "assumptions": [],
            "paper_id": None,
            "source_paper_id": None,
            "source_location": _source_location(text),
            "proof_technique": _proof_technique(text),
            "cluster_id": None,
            "tags": _tags_from_text(text),
            "notes": None,
        },
    }


def _open_problem_candidate(text: str) -> dict[str, Any]:
    return {
        "entry_type": "open_problem",
        "payload": {
            "title": _short_title(text, "Open problem"),
            "statement": _strip_label(text),
            "context": None,
            "status": "active",
            "paper_id": None,
            "source_paper_id": None,
            "source_location": _source_location(text),
            "cluster_id": None,
            "tags": _tags_from_text(text),
            "notes": None,
        },
    }


def _conjecture_candidate(text: str) -> dict[str, Any]:
    return {
        "entry_type": "conjecture_seed",
        "payload": {
            "title": _short_title(text, "Conjecture"),
            "statement": _strip_label(text),
            "cluster_id": None,
            "motivation": None,
            "related_theorems": [],
            "expected_status": "unknown",
            "confidence": "needs_review",
            "attack_plan": None,
            "possible_counterexamples": [],
            "status": "active",
            "notes": None,
        },
    }


def _strip_label(text: str) -> str:
    return re.sub(
        r"^\s*(theorem|lemma|proposition|reduction|open problem|open question|model|concept|conjecture|summary)\s*[:.\-]\s*",
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
    for token in (
        "ats",
        "cdm",
        "2dm",
        "control",
        "petri",
        "distributed synthesis",
        "asynchronous automata",
        "trace",
        "partial information",
        "safety",
        "reachability",
        "parity",
    ):
        if token in lowered:
            tags.append(token.replace(" ", "_"))
    return tags


def _model_families(text: str) -> list[str]:
    lowered = text.lower()
    mapping = {
        "ats": "ATS games",
        "cdm": "CDM games",
        "2dm": "2DM games",
        "control game": "control games",
        "petri": "Petri games",
        "asynchronous automata": "asynchronous automata",
        "zielonka": "asynchronous automata",
        "trace": "Mazurkiewicz traces",
        "partial information": "partial-information games",
        "imperfect information": "partial-information games",
    }
    return [label for token, label in mapping.items() if token in lowered]


def _objective_families(text: str) -> list[str]:
    lowered = text.lower()
    objectives = []
    for token, label in (
        ("safety", "safety"),
        ("reachability", "reachability"),
        ("parity", "parity"),
        ("liveness", "liveness"),
        ("global objective", "global"),
        ("local objective", "local"),
    ):
        if token in lowered:
            objectives.append(label)
    return objectives


def _theorem_type(text: str) -> str:
    lowered = text.lower()
    if "undecidable" in lowered or "undecidability" in lowered:
        return "undecidability"
    if "complete" in lowered:
        return "completeness"
    if "hard" in lowered or "lower bound" in lowered:
        return "complexity_lower"
    if "upper bound" in lowered or "algorithm" in lowered:
        return "complexity_upper"
    if "equivalent" in lowered or "equivalence" in lowered:
        return "equivalence"
    if "characterization" in lowered or "characterizes" in lowered:
        return "characterization"
    if "decidable" in lowered or "decidability" in lowered:
        return "decidability"
    return "characterization"


def _architecture_assumptions(text: str) -> list[str]:
    assumptions = []
    lowered = text.lower()
    for token in ("pipeline", "tree", "ring", "hierarchical", "two-process", "bounded processes"):
        if token in lowered:
            assumptions.append(token)
    return assumptions


def _information_assumptions(text: str) -> list[str]:
    lowered = text.lower()
    assumptions = []
    for token in ("perfect information", "partial information", "imperfect information", "local observation", "causal memory"):
        if token in lowered:
            assumptions.append(token)
    return assumptions


def _strategy_assumptions(text: str) -> list[str]:
    lowered = text.lower()
    assumptions = []
    for token in ("memoryless", "finite-state", "distributed strategy", "uniform strategy", "memory automaton"):
        if token in lowered:
            assumptions.append(token)
    return assumptions


def _process_bound(text: str) -> str | None:
    match = re.search(r"\b(\d+)[-\s]*(process|player)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    if "two-process" in text.lower() or "two process" in text.lower():
        return "2"
    return None


def _complexity_bound(text: str, upper: bool) -> str | None:
    lowered = text.lower()
    classes = ["2exptime", "nexptime", "exptime", "pspace", "np", "ptime"]
    for cls in classes:
        if cls in lowered:
            if upper and any(token in lowered for token in ("upper", "in ", "decidable", "algorithm")):
                return cls.upper()
            if not upper and any(token in lowered for token in ("hard", "lower", "complete")):
                return f"{cls.upper()}-hard" if "hard" in lowered else cls.upper()
    return None


def _memory_bound(text: str, upper: bool) -> str | None:
    lowered = text.lower()
    if "memoryless" in lowered:
        return "memoryless"
    if "finite-state" in lowered:
        return "finite-state"
    if "unbounded memory" in lowered and not upper:
        return "unbounded"
    return None


def _source_location(text: str) -> str | None:
    match = re.search(r"\b(page|p\.|theorem|section|sec\.)\s*([A-Za-z0-9_.-]+)", text, flags=re.IGNORECASE)
    if match:
        return f"{match.group(1)} {match.group(2)}"
    return None


def _proof_technique(text: str) -> str | None:
    lowered = text.lower()
    for token in ("reduction", "fixed-point", "automata construction", "linearization", "gossip automaton", "knowledge"):
        if token in lowered:
            return token
    return None


def _concept_type(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("objective", "safety", "reachability", "parity")):
        return "objective"
    if any(token in lowered for token in ("strategy", "memory")):
        return "strategy"
    if any(token in lowered for token in ("exptime", "pspace", "nexptime", "undecidability")):
        return "complexity_class"
    if any(token in lowered for token in ("mso", "logic")):
        return "logic"
    if any(token in lowered for token in ("reduction", "linearization", "fixed-point")):
        return "proof_technique"
    return "model"


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
    if "petri" in lowered:
        return "petri_game"
    if "partial" in lowered or "imperfect" in lowered:
        return "partial_information_game"
    if "automata" in lowered:
        return "automata"
    if "trace" in lowered:
        return "trace"
    if "distributed" in lowered:
        return "distributed_synthesis"
    return "other"


def _first_or_none(values: list[str]) -> str | None:
    return values[0] if values else None

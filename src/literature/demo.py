"""Rudimentary local literature workflow using approved seed artifacts."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .. import db
from ..llm import LLMClient, LLMError, LLMMessage, LLMRequest
from ..schemas import EvidenceSpan, LiteratureNote, LiteratureSummary, Paper, ResearchCluster, ResearchTopic


TARGET_TOPIC = "causally ordered two-decision-maker ATS/CDM games and decidability of distributed safety synthesis"
TARGET_CLUSTER = "Restricted multi-decision-maker synthesis"

PAPER_SEEDS: dict[int, Paper] = {
    1: Paper(
        title="Distributed Games with a Central Decision Maker",
        authors=["Bharat Adsul", "Nehul Jain"],
        year=2025,
        venue="FSTTCS",
        url="https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.FSTTCS.2025.5",
    ),
    2: Paper(
        title="Non-deterministic asynchronous automata games and their undecidability",
        authors=["Bharat Adsul", "Nehul Jain"],
        year=2024,
        venue="arXiv",
        url="https://arxiv.org/abs/2410.04420",
    ),
    3: Paper(
        title="Petri Games: Synthesis of Distributed Systems with Causal Memory",
        authors=["Bernd Finkbeiner", "Ernst-Ruediger Olderog"],
        year=2014,
        venue="EPTCS",
        url="https://doi.org/10.4204/EPTCS.161.19",
    ),
    4: Paper(
        title="Global Winning Conditions in Synthesis of Distributed Systems with Causal Memory",
        authors=["Bernd Finkbeiner", "Manuel Gieseking", "Jesko Hecking-Harbusch", "Ernst-Ruediger Olderog"],
        year=2022,
        venue="CSL",
        url="https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.CSL.2022.20",
    ),
    5: Paper(
        title="Distributed synthesis for acyclic architectures",
        authors=["Anca Muscholl", "Igor Walukiewicz"],
        year=2014,
        venue="Draft",
        url="https://www.labri.fr/perso/igw/Papers/igw-dist-acyclic.pdf",
    ),
}


@dataclass(frozen=True)
class ResearchDemoResult:
    """Counts and output path from a local research demo run."""

    topic_id: int
    topics_created: int
    notes_loaded: int
    summaries_created: int
    evidence_spans_linked: int
    output_files_written: int
    report_path: Path


@dataclass(frozen=True)
class LiteratureQueryResult:
    """A deterministic answer from stored literature summaries and evidence."""

    answer: str
    evidence_count: int
    insufficient: bool


@dataclass(frozen=True)
class ResearchMemoResult:
    """Output metadata from a stored-evidence research memo."""

    topic_id: int
    memo_path: Path
    evidence_count: int
    used_llm: bool


@dataclass(frozen=True)
class QualityCheckResult:
    """Output metadata from a literature quality report."""

    topic_id: int
    report_path: Path
    issue_count: int


@dataclass(frozen=True)
class VerificationTasksResult:
    """Output metadata from a verification-task report."""

    topic_id: int
    tasks_path: Path
    task_count: int


class MemoOrganization(BaseModel):
    """The constrained JSON shape accepted from an LLM memo organizer."""

    model_config = ConfigDict(extra="forbid")

    memo: str = Field(min_length=1)


def run_research_demo(
    dry_run: bool = True,
    db_path: str | Path | None = None,
    approved_dir: str | Path | None = None,
    output_path: str | Path | None = None,
) -> ResearchDemoResult:
    """Run the smallest local literature workflow against approved seed JSON."""

    approved_root = Path(approved_dir) if approved_dir else db.PROJECT_ROOT / "results" / "approved"
    report_path = Path(output_path) if output_path else db.PROJECT_ROOT / "results" / "literature" / "demo_literature_map.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        cluster_id = _ensure_cluster(connection)
        topic_id = db.insert_research_topic(connection, _topic())
        paper_ids = _ensure_seed_papers(connection, cluster_id)

        evidence_count = 0
        notes_loaded = 0
        summaries_created = 0
        evidence_records: list[dict[str, Any]] = []

        for seed_path in sorted(approved_root.glob("*.json")):
            payload = json.loads(seed_path.read_text(encoding="utf-8"))
            source_paper_id = int(payload.get("source_paper_id") or 0)
            paper_id = paper_ids.get(source_paper_id) or next(iter(paper_ids.values()))
            note_id = db.insert_literature_note(
                connection,
                _note_from_seed(topic_id, paper_id, seed_path, payload),
            )
            notes_loaded += 1

            evidence_id = db.insert_evidence_span(
                connection,
                EvidenceSpan(
                    paper_id=paper_id,
                    entry_type="literature_note",
                    entry_id=note_id,
                    quote_or_summary=_evidence_summary(payload),
                    confidence=payload.get("confidence") or "needs_review",
                    notes=f"source_path={_relative(seed_path)}",
                ),
            )
            evidence_count += 1

            summary = _summary_from_seed(topic_id, note_id, paper_id, payload, seed_path, evidence_id)
            db.insert_literature_summary(connection, summary)
            summaries_created += 1
            evidence_records.append(
                {
                    "evidence_id": evidence_id,
                    "note_id": note_id,
                    "paper_id": paper_id,
                    "source_path": _relative(seed_path),
                    "payload": payload,
                }
            )

        topic = db.get_research_topic(connection, topic_id)
        summaries = db.list_literature_summaries(connection, topic_id=topic_id)
        papers = {paper.id: paper for paper in db.list_papers(connection) if paper.id is not None}

    report = _build_literature_map(topic, summaries, papers, evidence_records, dry_run=dry_run)
    report_path.write_text(report, encoding="utf-8")

    return ResearchDemoResult(
        topic_id=topic_id,
        topics_created=1,
        notes_loaded=notes_loaded,
        summaries_created=summaries_created,
        evidence_spans_linked=evidence_count,
        output_files_written=1,
        report_path=report_path,
    )


def query_literature(
    topic_id: int,
    question: str,
    db_path: str | Path | None = None,
    max_results: int = 5,
) -> LiteratureQueryResult:
    """Answer a question from stored literature summaries and evidence only."""

    terms = _query_terms(question)
    if not terms:
        return LiteratureQueryResult(
            answer="insufficient stored evidence: no searchable terms in question",
            evidence_count=0,
            insufficient=True,
        )

    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        topic = db.get_research_topic(connection, topic_id)
        if topic is None:
            return LiteratureQueryResult(
                answer=f"insufficient stored evidence: topic {topic_id} does not exist",
                evidence_count=0,
                insufficient=True,
            )
        summaries = db.list_literature_summaries(connection, topic_id=topic_id)
        papers = {paper.id: paper for paper in db.list_papers(connection) if paper.id is not None}
        scored = []
        for summary in summaries:
            haystack = f"{summary.markdown_summary}\n{json.dumps(summary.summary_json, sort_keys=True)}"
            score = _score(haystack, terms)
            if score == 0:
                continue
            evidence = []
            if summary.note_id is not None:
                evidence = db.list_evidence_spans(connection, entry_type="literature_note", entry_id=summary.note_id)
            scored.append((score, summary, evidence))

    if not scored:
        return LiteratureQueryResult(
            answer="insufficient stored evidence for this question",
            evidence_count=0,
            insufficient=True,
        )

    scored.sort(key=lambda item: item[0], reverse=True)
    lines = [f"Answer from stored evidence for topic {topic_id}:"]
    evidence_count = 0
    for _, summary, evidence_items in scored[:max_results]:
        statement = summary.summary_json.get("statement") or _snippet(summary.markdown_summary, terms)
        paper = papers.get(summary.paper_id)
        paper_title = paper.title if paper else f"paper {summary.paper_id}"
        citations = []
        for evidence in evidence_items:
            evidence_count += 1
            source_path = _source_path_from_notes(evidence.notes)
            citations.append(f"evidence_id={evidence.evidence_id}, source={source_path or 'stored evidence'}")
        citation_text = "; ".join(citations) if citations else "stored summary without evidence span"
        lines.append(f"- {statement} ({paper_title}; {citation_text})")

    return LiteratureQueryResult(answer="\n".join(lines), evidence_count=evidence_count, insufficient=False)


def write_research_memo(
    topic_id: int,
    question: str,
    db_path: str | Path | None = None,
    output_path: str | Path | None = None,
    use_llm: bool = False,
) -> ResearchMemoResult:
    """Write a research memo from stored literature evidence only."""

    memo_path = Path(output_path) if output_path else db.PROJECT_ROOT / "results" / "literature" / f"topic_{topic_id}_research_memo.md"
    memo_path.parent.mkdir(parents=True, exist_ok=True)

    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        topic = db.get_research_topic(connection, topic_id)
        if topic is None:
            raise ValueError(f"research topic {topic_id} does not exist")
        notes = db.list_literature_notes(connection, topic_id=topic_id)
        summaries = db.list_literature_summaries(connection, topic_id=topic_id)
        papers = {paper.id: paper for paper in db.list_papers(connection) if paper.id is not None}
        evidence_by_note: dict[int, list[EvidenceSpan]] = {}
        for note in notes:
            if note.id is None:
                continue
            evidence_by_note[note.id] = db.list_evidence_spans(
                connection,
                entry_type="literature_note",
                entry_id=note.id,
            )

    evidence_records = _memo_evidence_records(summaries, evidence_by_note, papers)
    memo = _build_research_memo(topic, question, evidence_records)
    used_llm = False
    if use_llm:
        organized = _organize_memo_with_llm(question, memo, evidence_records)
        if organized:
            memo = organized
            used_llm = True

    memo_path.write_text(memo, encoding="utf-8")
    evidence_count = sum(len(record["evidence"]) for record in evidence_records)
    return ResearchMemoResult(topic_id=topic_id, memo_path=memo_path, evidence_count=evidence_count, used_llm=used_llm)


def quality_check_literature(
    topic_id: int,
    db_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> QualityCheckResult:
    """Write a local quality report for stored literature evidence and memo outputs."""

    report_path = Path(output_path) if output_path else db.PROJECT_ROOT / "results" / "literature" / f"topic_{topic_id}_quality_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    context = _load_literature_context(topic_id, db_path)

    notes = context["notes"]
    summaries = context["summaries"]
    evidence_by_note = context["evidence_by_note"]
    source_files = sorted({note.source_path for note in notes})
    duplicate_sources = sorted(source for source, count in Counter(note.source_path for note in notes).items() if count > 1)
    missing_seed_sources = _missing_seed_sources(source_files)
    memo_path = db.PROJECT_ROOT / "results" / "literature" / f"topic_{topic_id}_research_memo.md"
    map_path = db.PROJECT_ROOT / "results" / "literature" / "demo_literature_map.md"
    memo_text = memo_path.read_text(encoding="utf-8") if memo_path.exists() else ""
    map_text = map_path.read_text(encoding="utf-8") if map_path.exists() else ""

    issues: list[str] = []
    if not notes:
        issues.append("No literature notes are stored for this topic.")
    if not summaries:
        issues.append("No literature summaries are stored for this topic.")
    if sum(len(items) for items in evidence_by_note.values()) == 0:
        issues.append("No evidence spans are linked to literature notes.")
    if duplicate_sources:
        issues.append("Duplicate literature notes use the same source file.")
    if missing_seed_sources:
        issues.append("Some approved seed results are not represented in this topic.")
    if not memo_text:
        issues.append("Research memo is missing; run research-memo for this topic.")
    if memo_text and "## Conjecture" not in memo_text:
        issues.append("Memo does not clearly label a conjecture section.")
    if memo_text and "needs verification" not in memo_text:
        issues.append("Memo does not visibly mark missing facts as needs verification.")
    unsupported_known = _unsupported_known_result_lines(memo_text)
    if unsupported_known:
        issues.append("Some known-result memo lines do not cite evidence.")

    query_checks = _quality_query_checks(topic_id, db_path)
    if not query_checks[-1]["insufficient"]:
        issues.append("Unrelated query did not return insufficient stored evidence.")

    report = _build_quality_report(
        topic_id=topic_id,
        notes=notes,
        summaries=summaries,
        evidence_by_note=evidence_by_note,
        source_files=source_files,
        duplicate_sources=duplicate_sources,
        missing_seed_sources=missing_seed_sources,
        memo_path=memo_path,
        map_path=map_path,
        memo_text=memo_text,
        map_text=map_text,
        unsupported_known=unsupported_known,
        query_checks=query_checks,
        issues=issues,
    )
    report_path.write_text(report, encoding="utf-8")
    return QualityCheckResult(topic_id=topic_id, report_path=report_path, issue_count=len(issues))


def generate_verification_tasks(
    topic_id: int,
    db_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> VerificationTasksResult:
    """Write literature/theory verification tasks derived from stored evidence and memo text."""

    tasks_path = Path(output_path) if output_path else db.PROJECT_ROOT / "results" / "literature" / f"topic_{topic_id}_verification_tasks.md"
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    context = _load_literature_context(topic_id, db_path)
    evidence_records = _memo_evidence_records(context["summaries"], context["evidence_by_note"], context["papers"])
    tasks = _verification_task_specs(evidence_records)
    tasks_path.write_text(_build_verification_tasks_markdown(tasks), encoding="utf-8")
    return VerificationTasksResult(topic_id=topic_id, tasks_path=tasks_path, task_count=len(tasks))


def _ensure_cluster(connection) -> int:
    return db.insert_cluster(
        connection,
        ResearchCluster(
            name=TARGET_CLUSTER,
            description="ATS/CDM/2DM frontier for causal memory, decision-maker restrictions, and safety synthesis.",
            status="active",
            priority=10,
        ),
        ignore_existing=True,
    )


def _topic() -> ResearchTopic:
    clarified = (
        "Focus on causally ordered two-decision-maker ATS/CDM games, global safety objectives, "
        "causal-memory distributed strategies, and what the approved seed results do or do not settle."
    )
    return ResearchTopic(
        title="Causally ordered 2DM ATS/CDM safety synthesis",
        raw_topic=TARGET_TOPIC,
        clarified_topic=clarified,
        clarification_json={
            "sanitized_topic": TARGET_TOPIC.strip(),
            "included_models": ["ATS games", "CDM games", "2DM games"],
            "included_objectives": ["global safety", "distributed safety synthesis"],
            "excluded": ["new web crawling", "unapproved theorem insertion"],
        },
    )


def _ensure_seed_papers(connection, cluster_id: int) -> dict[int, int]:
    paper_ids: dict[int, int] = {}
    for source_id, seed in PAPER_SEEDS.items():
        paper = Paper(
            title=seed.title,
            authors=seed.authors,
            year=seed.year,
            venue=seed.venue,
            url=seed.url,
            cluster_id=cluster_id if source_id in {1, 2} else seed.cluster_id,
            notes="Loaded by local research-demo from approved seed artifacts.",
        )
        paper_ids[source_id] = db.insert_paper(connection, paper)
    return paper_ids


def _note_from_seed(topic_id: int, paper_id: int, seed_path: Path, payload: dict[str, Any]) -> LiteratureNote:
    return LiteratureNote(
        topic_id=topic_id,
        paper_id=paper_id,
        source_path=_relative(seed_path),
        note_type="approved_seed_result",
        title=_title(payload),
        content_json=payload,
        markdown_note=_markdown_note(payload, seed_path),
    )


def _summary_from_seed(
    topic_id: int,
    note_id: int,
    paper_id: int,
    payload: dict[str, Any],
    seed_path: Path,
    evidence_id: int,
) -> LiteratureSummary:
    summary_json = {
        "statement": payload.get("statement"),
        "theorem_type": payload.get("theorem_type"),
        "model_family": payload.get("model_family"),
        "objective_family": payload.get("objective_family"),
        "architecture_assumptions": payload.get("architecture_assumptions") or [],
        "information_assumptions": payload.get("information_assumptions") or [],
        "strategy_assumptions": payload.get("strategy_assumptions") or [],
        "process_bound": payload.get("process_bound"),
        "complexity_upper": payload.get("complexity_upper"),
        "complexity_lower": payload.get("complexity_lower"),
        "memory_upper": payload.get("memory_upper"),
        "memory_lower": payload.get("memory_lower"),
        "source_location": payload.get("source_location"),
        "source_path": _relative(seed_path),
        "evidence_id": evidence_id,
        "relevance_to_topic": _relevance(payload),
        "verification_status": "from existing approved seed JSON; source-paper evidence should still be checked for publication-grade use",
    }
    return LiteratureSummary(
        topic_id=topic_id,
        note_id=note_id,
        paper_id=paper_id,
        summary_json=summary_json,
        markdown_summary=_markdown_summary(summary_json),
    )


def _build_literature_map(
    topic: ResearchTopic | None,
    summaries: list[LiteratureSummary],
    papers: dict[int, Paper],
    evidence_records: list[dict[str, Any]],
    dry_run: bool,
) -> str:
    topic_text = topic.clarified_topic if topic else TARGET_TOPIC
    relevant = [_summary_payload(summary) for summary in summaries]
    target_relevant = [item for item in relevant if _is_target_relevant(item)]

    lines = [
        "# Literature Map",
        "",
        "## Topic",
        topic_text,
        "",
        "## Known Results",
    ]
    for item in target_relevant or relevant[:5]:
        lines.append(f"- {item.get('statement')} [evidence {item.get('evidence_id')}; {item.get('source_path')}]")

    lines.extend(["", "## Important Definitions"])
    definition_lines = _definition_lines(target_relevant or relevant)
    lines.extend(definition_lines or ["- needs verification: no explicit definitions were stored in the local seed summaries."])

    lines.extend(["", "## Decidability / Complexity Claims"])
    for item in relevant:
        if item.get("complexity_upper") or item.get("complexity_lower") or item.get("theorem_type") in {"decidability", "undecidability"}:
            lines.append(
                "- "
                f"{item.get('model_family')}: {item.get('theorem_type')}; "
                f"upper={item.get('complexity_upper')}; lower={item.get('complexity_lower')} "
                f"[evidence {item.get('evidence_id')}]"
            )

    lines.extend(["", "## Relevance to the Proposed Direction"])
    cdm_safety_evidence = _evidence_id_for(evidence_records, "t01_cdm_global_safety")
    two_dm_evidence = _evidence_id_for(evidence_records, "t04_2dm_ats")
    lines.append(
        "- Stored seed evidence says global safety CDM games are EXPTIME-complete, giving a decidable baseline for one central decision maker. "
        f"This directly frames the proposed two-decision-maker restriction. [evidence {cdm_safety_evidence}]"
    )
    lines.append(
        "- Stored seed evidence says ATS games with two decision makers are undecidable when the reduction uses truly concurrent decisions. "
        f"The causally ordered two-decision-maker fragment is therefore a focused gap, not a settled theorem in this demo. [evidence {two_dm_evidence}]"
    )
    lines.append("- needs verification: whether causal ordering alone recovers decidability is not established by the stored approved seed JSON.")

    lines.extend(["", "## Gaps / Unclear Points"])
    lines.append("- needs verification: exact formal definition of 'causally ordered decision events' for 2DM ATS/CDM games.")
    lines.append("- needs verification: whether the two-DM undecidability proof can be blocked by forbidding concurrent decision events.")
    lines.append("- needs verification: whether the CDM sequential-game reduction extends to a causally ordered two-DM architecture.")

    lines.extend(["", "## Evidence Used"])
    for record in evidence_records:
        paper = papers.get(record["paper_id"])
        paper_title = paper.title if paper else f"paper {record['paper_id']}"
        payload = record["payload"]
        lines.append(
            f"- evidence {record['evidence_id']}: {paper_title}; {payload.get('source_location')}; {record['source_path']}"
        )

    if dry_run:
        lines.extend(["", "Dry-run note: this report used only local approved seed JSON artifacts."])
    return "\n".join(lines) + "\n"


def _memo_evidence_records(
    summaries: list[LiteratureSummary],
    evidence_by_note: dict[int, list[EvidenceSpan]],
    papers: dict[int, Paper],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for summary in summaries:
        evidence_items = evidence_by_note.get(summary.note_id or -1, [])
        paper = papers.get(summary.paper_id)
        records.append(
            {
                "summary": summary.summary_json,
                "markdown": summary.markdown_summary,
                "paper_title": paper.title if paper else f"paper {summary.paper_id}",
                "evidence": [
                    {
                        "evidence_id": evidence.evidence_id,
                        "source_path": _source_path_from_notes(evidence.notes),
                        "quote_or_summary": evidence.quote_or_summary,
                    }
                    for evidence in evidence_items
                ],
            }
        )
    return records


def _build_research_memo(
    topic: ResearchTopic,
    question: str,
    evidence_records: list[dict[str, Any]],
) -> str:
    target_records = [record for record in evidence_records if _is_target_relevant(record["summary"])]
    records = target_records or evidence_records
    cdm_safety = _first_record(records, "t01_cdm_global_safety")
    two_dm = _first_record(records, "t04_2dm_ats")
    finite_memory = _first_record(records, "t03_cdm_finite_state")
    ats_undecidable = _first_record(records, "t05_ats_undecidable")

    lines = [
        "# Research Memo",
        "",
        "## Question",
        question,
        "",
        "## Topic",
        topic.clarified_topic,
        "",
        "## Known Results",
    ]
    for record in records[:6]:
        summary = record["summary"]
        lines.append(f"- {summary.get('statement')} {_citation_text(record)}")

    lines.extend(["", "## Definitions / Assumptions Being Used"])
    definitions = _memo_definition_lines(records)
    lines.extend(definitions or ["- needs verification: no stored definition or assumption evidence was found for this topic."])

    lines.extend(["", "## Unclear Or Missing Facts"])
    lines.append("- needs verification: a formal definition of causal ordering for two-decision-maker ATS/CDM decision events.")
    lines.append("- needs verification: whether the two-decision-maker undecidability reduction still works when concurrent decision events are forbidden.")
    lines.append("- needs verification: whether the CDM sequentialization proof can be generalized beyond one central decision maker.")

    lines.extend(["", "## Conjecture"])
    lines.append(
        "- conjecture: causal ordering in two-decision-maker ATS/CDM safety games plausibly identifies a decidable fragment of distributed safety synthesis, "
        "but this is not established by the stored evidence."
    )

    lines.extend(["", "## Evidence For The Conjecture"])
    if cdm_safety:
        lines.append(
            "- A nearby one-decision-maker baseline is decidable/EXPTIME-complete for global safety, so causal-memory safety synthesis is not uniformly hopeless in the restricted ATS/CDM setting. "
            f"{_citation_text(cdm_safety)}"
        )
    if finite_memory:
        lines.append(
            "- The CDM seed includes a finite-state/gossip-memory construction, suggesting that structured causal information can support finite controllers in the restricted setting. "
            f"{_citation_text(finite_memory)}"
        )
    if two_dm:
        lines.append(
            "- The stored two-DM undecidability seed records that the reduction uses truly concurrent decisions; this makes the causally ordered variant a plausible boundary to test, not an established theorem. "
            f"{_citation_text(two_dm)}"
        )
    if not any((cdm_safety, finite_memory, two_dm)):
        lines.append("- needs verification: no stored evidence directly supports the conjecture.")

    lines.extend(["", "## Reasons for Caution"])
    if two_dm:
        lines.append(
            "- Two-decision-maker ATS games are stored as undecidable in the seed evidence, so adding a second decision maker is a serious source of hardness. "
            f"{_citation_text(two_dm)}"
        )
    if ats_undecidable:
        lines.append(
            "- General ATS games are also stored as undecidable, so a decidability claim requires precise architectural and information assumptions. "
            f"{_citation_text(ats_undecidable)}"
        )
    lines.append("- needs verification: the stored evidence does not prove that causal ordering blocks the known undecidability mechanism.")
    lines.append("- needs verification: the stored CDM results assume a single central decision maker, which is stronger than the conjectured two-decision-maker setting.")

    lines.extend(["", "## Next Verification Step"])
    lines.append(
        "- Formalize the causally ordered two-decision-maker restriction, then check the proof of the stored two-DM undecidability result to isolate the first step requiring truly concurrent decision events."
    )
    lines.append(
        "- In parallel, test whether the CDM sequential-game construction can be parameterized by a causal order over the two decision makers; any claim here remains needs verification until checked against source proofs."
    )

    lines.extend(["", "## Evidence Used"])
    for record in evidence_records:
        summary = record["summary"]
        for evidence in record["evidence"]:
            lines.append(
                f"- evidence {evidence['evidence_id']}: {record['paper_title']}; "
                f"{summary.get('source_location')}; {evidence.get('source_path')}"
            )
    return "\n".join(lines) + "\n"


def _memo_definition_lines(records: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    lines: list[str] = []
    for record in records:
        summary = record["summary"]
        parts = [
            f"model={summary.get('model_family')}",
            f"objective={summary.get('objective_family')}",
            "architecture=" + "; ".join(summary.get("architecture_assumptions") or []),
            "information=" + "; ".join(summary.get("information_assumptions") or []),
            "strategy=" + "; ".join(summary.get("strategy_assumptions") or []),
        ]
        text = "; ".join(part for part in parts if part and not part.endswith("="))
        if text in seen:
            continue
        seen.add(text)
        lines.append(f"- {text} {_citation_text(record)}")
    return lines[:6]


def _first_record(records: list[dict[str, Any]], source_fragment: str) -> dict[str, Any] | None:
    for record in records:
        if source_fragment in _record_source_text(record):
            return record
    return None


def _record_source_text(record: dict[str, Any]) -> str:
    return " ".join(str(evidence.get("source_path") or "") for evidence in record.get("evidence", []))


def _citation_text(record: dict[str, Any]) -> str:
    citations = []
    for evidence in record.get("evidence", []):
        source = evidence.get("source_path") or "stored evidence"
        citations.append(f"evidence {evidence.get('evidence_id')}; {source}")
    return "[" + "; ".join(citations) + "]" if citations else "[needs verification: no evidence span]"


def _organize_memo_with_llm(
    question: str,
    deterministic_memo: str,
    evidence_records: list[dict[str, Any]],
) -> str | None:
    client = LLMClient()
    if not client.available:
        return None
    prompt = {
        "instruction": (
            "Reorganize the provided research memo for clarity. Use only the memo and evidence records. "
            "Do not add facts, citations, or evidence. Keep unsupported points labelled 'needs verification' "
            "and keep any conjecture labelled 'conjecture'. Return JSON with exactly one key: memo."
        ),
        "question": question,
        "memo": deterministic_memo,
        "evidence_records": evidence_records,
    }
    try:
        organization = client.complete_json(
            LLMRequest(
                messages=(
                    LLMMessage(
                        role="system",
                        content="You organize stored-evidence research memos without inventing facts or citations.",
                    ),
                    LLMMessage(role="user", content=json.dumps(prompt, sort_keys=True)),
                ),
                temperature=0.0,
                json_mode=True,
            ),
            MemoOrganization,
        )
    except LLMError:
        return None
    content = organization.memo
    if "Known Results" not in content or "Conjecture" not in content:
        return None
    if not _uses_only_stored_citations(content, evidence_records):
        return None
    return content if content.endswith("\n") else content + "\n"


def _uses_only_stored_citations(content: str, evidence_records: list[dict[str, Any]]) -> bool:
    """Reject organizer output that cites evidence or local artifacts it was not given."""

    known_evidence_ids = {
        str(item["evidence_id"])
        for record in evidence_records
        for item in record["evidence"]
        if item.get("evidence_id") is not None
    }
    cited_evidence_ids = set(re.findall(r"\bevidence(?:_id=|\s+)(\d+)\b", content, flags=re.IGNORECASE))
    if known_evidence_ids and not cited_evidence_ids:
        return False
    if not cited_evidence_ids.issubset(known_evidence_ids):
        return False

    known_paths = {
        str(item["source_path"])
        for record in evidence_records
        for item in record["evidence"]
        if item.get("source_path")
    }
    cited_paths = set(re.findall(r"results/approved/[A-Za-z0-9_.-]+\.json", content))
    return cited_paths.issubset(known_paths)


def _load_literature_context(topic_id: int, db_path: str | Path | None) -> dict[str, Any]:
    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        topic = db.get_research_topic(connection, topic_id)
        if topic is None:
            raise ValueError(f"research topic {topic_id} does not exist")
        notes = db.list_literature_notes(connection, topic_id=topic_id)
        summaries = db.list_literature_summaries(connection, topic_id=topic_id)
        papers = {paper.id: paper for paper in db.list_papers(connection) if paper.id is not None}
        evidence_by_note: dict[int, list[EvidenceSpan]] = {}
        for note in notes:
            if note.id is not None:
                evidence_by_note[note.id] = db.list_evidence_spans(
                    connection,
                    entry_type="literature_note",
                    entry_id=note.id,
                )
    return {
        "topic": topic,
        "notes": notes,
        "summaries": summaries,
        "papers": papers,
        "evidence_by_note": evidence_by_note,
    }


def _missing_seed_sources(source_files: list[str]) -> list[str]:
    approved_dir = db.PROJECT_ROOT / "results" / "approved"
    expected = sorted(path.relative_to(db.PROJECT_ROOT).as_posix() for path in approved_dir.glob("*.json"))
    return [source for source in expected if source not in source_files]


def _unsupported_known_result_lines(memo_text: str) -> list[str]:
    if not memo_text:
        return []
    lines = memo_text.splitlines()
    in_known_results = False
    unsupported: list[str] = []
    for line in lines:
        if line.startswith("## "):
            in_known_results = line == "## Known Results"
            continue
        if in_known_results and line.startswith("- ") and "evidence " not in line and "needs verification" not in line:
            unsupported.append(line)
    return unsupported


def _quality_query_checks(topic_id: int, db_path: str | Path | None) -> list[dict[str, Any]]:
    queries = [
        "What is known about global safety in CDM or ATS games?",
        "What evidence weakens the conjecture?",
        "Which exact result blocks decidability?",
        "What does this say about protein folding?",
    ]
    checks: list[dict[str, Any]] = []
    for question in queries:
        result = query_literature(topic_id, question, db_path=db_path)
        checks.append(
            {
                "question": question,
                "insufficient": result.insufficient,
                "evidence_count": result.evidence_count,
                "answer": result.answer,
            }
        )
    return checks


def _build_quality_report(
    topic_id: int,
    notes: list[LiteratureNote],
    summaries: list[LiteratureSummary],
    evidence_by_note: dict[int, list[EvidenceSpan]],
    source_files: list[str],
    duplicate_sources: list[str],
    missing_seed_sources: list[str],
    memo_path: Path,
    map_path: Path,
    memo_text: str,
    map_text: str,
    unsupported_known: list[str],
    query_checks: list[dict[str, Any]],
    issues: list[str],
) -> str:
    evidence_count = sum(len(items) for items in evidence_by_note.values())
    lines = [
        "# Quality Report",
        "",
        "## Coverage",
        f"- Notes: {len(notes)}",
        f"- Summaries: {len(summaries)}",
        f"- Evidence spans: {evidence_count}",
        f"- Literature map present: {'yes' if map_text else 'no'} ({_display_path(map_path)})",
        f"- Research memo present: {'yes' if memo_text else 'no'} ({_display_path(memo_path)})",
        "- Source files used:",
    ]
    lines.extend(f"  - {source}" for source in source_files)
    if duplicate_sources:
        lines.append("- Duplicate notes detected:")
        lines.extend(f"  - {source}" for source in duplicate_sources)
    else:
        lines.append("- Duplicate notes detected: none")
    if missing_seed_sources:
        lines.append("- Important seed results missing:")
        lines.extend(f"  - {source}" for source in missing_seed_sources)
    else:
        lines.append("- Important seed results missing: none detected")

    lines.extend(["", "## Citation / Evidence Quality"])
    if memo_text:
        lines.append(f"- Memo evidence-id citations: {memo_text.count('evidence ')}")
        lines.append(f"- Memo source-path citations: {memo_text.count('results/approved/')}")
        if unsupported_known:
            lines.append("- Unsupported known-result lines:")
            lines.extend(f"  - {line}" for line in unsupported_known)
        else:
            lines.append("- Unsupported known-result lines: none detected")
        lines.append(f"- Conjectures clearly labelled: {'yes' if 'conjecture:' in memo_text.lower() else 'no'}")
    else:
        lines.append("- Memo citation check: memo not found")

    lines.extend(["", "## Caution / Overclaiming Check"])
    if memo_text:
        lines.append(f"- Claims a theorem without local evidence: {'yes' if unsupported_known else 'no'}")
        lines.append(f"- Distinguishes known results from conjectures: {'yes' if '## Known Results' in memo_text and '## Conjecture' in memo_text else 'no'}")
        lines.append(f"- Marks missing facts as needs verification: {'yes' if 'needs verification' in memo_text else 'no'}")
    else:
        lines.append("- Memo caution check: memo not found")

    lines.extend(["", "## Query Robustness"])
    for check in query_checks:
        status = "insufficient stored evidence" if check["insufficient"] else f"answered with {check['evidence_count']} evidence citations"
        lines.append(f"- {check['question']}: {status}")
        lines.append(f"  - {check['answer'].splitlines()[0]}")

    lines.extend(["", "## Recommended Fixes"])
    if issues:
        lines.extend(f"- {issue}" for issue in issues)
    else:
        lines.append("- No required fixes detected for the current local vertical slice.")
    lines.append("- Continue to verify source-paper theorem statements before using memo claims as publication-grade facts.")
    return "\n".join(lines) + "\n"


def _verification_task_specs(evidence_records: list[dict[str, Any]]) -> list[dict[str, str]]:
    cdm_safety = _first_record(evidence_records, "t01_cdm_global_safety")
    two_dm = _first_record(evidence_records, "t04_2dm_ats")
    ats_undecidable = _first_record(evidence_records, "t05_ats_undecidable")
    acyclic = _first_record(evidence_records, "t09_acyclic_zielonka")
    petri_global = _first_record(evidence_records, "t07_petri_global_bad")
    return [
        {
            "question": "Verify the precise model difference between CDM, ATS, and 2DM in the stored sources.",
            "why": "The conjecture depends on whether the CDM restriction and two-decision-maker restriction preserve enough causal structure for synthesis.",
            "available": _task_evidence(cdm_safety, two_dm, ats_undecidable),
            "missing": "needs verification: exact source definitions and whether the stored summaries preserve all architecture assumptions.",
            "check": "Read the model definitions around the cited CDM/ATS/2DM theorem locations and compare action participation and decision-maker assumptions.",
        },
        {
            "question": "Verify whether causal ordering blocks the known two-decision-maker undecidability construction.",
            "why": "This is the central plausibility test for the conjecture.",
            "available": _task_evidence(two_dm),
            "missing": "needs verification: the proof step where truly concurrent decision events are required.",
            "check": "Inspect the proof cited by the two-DM undecidability seed and isolate every use of concurrent decisions.",
        },
        {
            "question": "Check whether acyclicity/Zielonka-style restrictions are comparable to causal ordering.",
            "why": "Acyclic architectures are a known decidability island, but comparability to causal ordering is not established by stored evidence.",
            "available": _task_evidence(acyclic),
            "missing": "needs verification: a formal translation or separation between acyclic communication architecture and causal ordering of decision events.",
            "check": "Compare the acyclic Zielonka assumptions against the 2DM ATS/CDM assumptions in the stored seed summaries.",
        },
        {
            "question": "Check whether global safety is being used as an objective, information pattern, or both.",
            "why": "The query target is distributed safety synthesis; confusing objective and information assumptions would overstate the result.",
            "available": _task_evidence(cdm_safety, petri_global),
            "missing": "needs verification: exact objective definition in the CDM source and whether global safety changes observation assumptions.",
            "check": "Read the cited global safety theorem statements and definitions; record objective family separately from information assumptions.",
        },
        {
            "question": "Identify the exact theorem that gives EXPTIME, undecidability, and decidability in the seed material.",
            "why": "The memo should be traceable to theorem labels before any claim is reused.",
            "available": _task_evidence(cdm_safety, two_dm, acyclic),
            "missing": "needs verification: page/theorem labels should be checked against source PDFs.",
            "check": "Open each cited seed source location and confirm theorem label, model family, objective family, and complexity status.",
        },
    ]


def _task_evidence(*records: dict[str, Any] | None) -> str:
    citations = []
    for record in records:
        if record is not None:
            citations.append(_citation_text(record))
    return "; ".join(citations) if citations else "needs verification: no stored evidence currently linked"


def _build_verification_tasks_markdown(tasks: list[dict[str, str]]) -> str:
    lines = ["# Verification Tasks", ""]
    for index, task in enumerate(tasks, start=1):
        lines.extend(
            [
                f"## Task {index}",
                "",
                f"* Question: {task['question']}",
                f"* Why it matters: {task['why']}",
                f"* Evidence currently available: {task['available']}",
                f"* Missing evidence: {task['missing']}",
                f"* Suggested next reading/check: {task['check']}",
                "* Status: open",
                "",
            ]
        )
    return "\n".join(lines)


def _definition_lines(items: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    lines: list[str] = []
    for item in items:
        parts = [
            f"model={item.get('model_family')}",
            f"objective={item.get('objective_family')}",
            "architecture=" + "; ".join(item.get("architecture_assumptions") or []),
            "information=" + "; ".join(item.get("information_assumptions") or []),
        ]
        text = "; ".join(part for part in parts if part and not part.endswith("="))
        if text and text not in seen:
            seen.add(text)
            lines.append(f"- {text} [evidence {item.get('evidence_id')}]")
    return lines[:8]


def _evidence_id_for(evidence_records: list[dict[str, Any]], source_fragment: str) -> int | str:
    for record in evidence_records:
        if source_fragment in str(record.get("source_path", "")):
            return record["evidence_id"]
    return "needs verification"


def _markdown_note(payload: dict[str, Any], seed_path: Path) -> str:
    return "\n".join(
        [
            f"# {_title(payload)}",
            "",
            f"Source seed: {_relative(seed_path)}",
            f"Statement: {payload.get('statement')}",
            f"Type: {payload.get('theorem_type')}",
            f"Model: {payload.get('model_family')}",
            f"Objective: {payload.get('objective_family')}",
            f"Source location: {payload.get('source_location')}",
        ]
    )


def _markdown_summary(summary_json: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"Statement: {summary_json.get('statement')}",
            f"Type: {summary_json.get('theorem_type')}",
            f"Model family: {summary_json.get('model_family')}",
            f"Objective family: {summary_json.get('objective_family')}",
            f"Complexity upper: {summary_json.get('complexity_upper')}",
            f"Complexity lower: {summary_json.get('complexity_lower')}",
            f"Relevance: {summary_json.get('relevance_to_topic')}",
            f"Evidence: {summary_json.get('evidence_id')} from {summary_json.get('source_path')}",
        ]
    )


def _evidence_summary(payload: dict[str, Any]) -> str:
    statement = str(payload.get("statement") or "").strip()
    location = payload.get("source_location")
    return f"{statement} Source location: {location}".strip()


def _title(payload: dict[str, Any]) -> str:
    statement = str(payload.get("statement") or "Untitled seed result")
    return statement[:90].rstrip(" .")


def _relevance(payload: dict[str, Any]) -> str:
    text = " ".join(
        str(payload.get(key) or "")
        for key in ("statement", "model_family", "objective_family", "process_bound", "complexity_upper", "complexity_lower")
    ).lower()
    if "cdm" in text and "safety" in text:
        return "central-decision-maker safety baseline"
    if "2dm" in text or "two decision" in text:
        return "direct two-decision-maker frontier evidence"
    if "ats" in text and "undecidable" in text:
        return "ATS undecidability baseline"
    return "background contrast evidence"


def _summary_payload(summary: LiteratureSummary) -> dict[str, Any]:
    return summary.summary_json


def _is_target_relevant(item: dict[str, Any]) -> bool:
    text = json.dumps(item, sort_keys=True).lower()
    return any(term in text for term in ("cdm", "ats", "2dm", "two decision"))


def _query_terms(question: str) -> list[str]:
    stop = {"what", "about", "known", "with", "from", "this", "that", "games", "game", "and", "the", "for", "does", "say"}
    lowered = question.lower()
    terms = [term for term in re.findall(r"[a-zA-Z0-9_]+", lowered) if len(term) >= 3 and term not in stop]
    if any(token in lowered for token in ("weakens", "against", "caution", "blocks decidability", "block decidability")):
        terms.extend(["undecidable", "undecidability", "2dm", "ats"])
    if "global safety" in lowered:
        terms.extend(["global_safety", "safety", "cdm", "ats"])
    return terms


def _score(text: str, terms: list[str]) -> int:
    token_counts = Counter(re.findall(r"[a-zA-Z0-9_]+", text.lower()))
    return sum(token_counts[term] for term in terms)


def _snippet(text: str, terms: list[str], width: int = 220) -> str:
    lowered = text.lower()
    first = min((lowered.find(term) for term in terms if term in lowered), default=0)
    start = max(first - 60, 0)
    return " ".join(text[start : start + width].split())


def _source_path_from_notes(notes: str | None) -> str | None:
    if not notes:
        return None
    prefix = "source_path="
    for part in notes.splitlines():
        if part.startswith(prefix):
            return part[len(prefix) :]
    return None


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(db.PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(db.PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()

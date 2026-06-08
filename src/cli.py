"""CLI for decidability, complexity, and synthesis research workflows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import db
from .code_artifacts import (
    get_code_artifact,
    list_code_artifacts,
    register_code_artifact,
    update_code_artifact_status,
)
from .curate import approve_pending, curate_pending, flag_pending, reject_pending
from .experiment_manager import run_experiment
from .extract import LLMClient, extract_from_text
from .experiments.ats_brute_solver import find_memoryless_safety_strategy
from .experiments.ats_generator import generate_tiny_game
from .experiments.ats_models import SafetyGame
from .ingest import add_paper
from .literature import (
    generate_verification_tasks,
    quality_check_literature,
    query_literature,
    run_research_demo,
    write_research_memo,
)
from .orchestrator import run_pipeline
from .schemas import Concept, ConceptLink, Conjecture, ResearchCluster


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Research harness for decidability, complexity, and strategy synthesis "
            "in distributed games and automata-theoretic synthesis."
        )
    )
    parser.add_argument("--db", default=str(db.DEFAULT_DB_PATH), help="SQLite database path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init-db", help="Create or migrate database tables")

    add_paper_parser = subparsers.add_parser("add-paper", help="Add a paper manually")
    add_paper_parser.add_argument("--title", required=True)
    add_paper_parser.add_argument("--authors", required=True, help="Semicolon-separated author list")
    add_paper_parser.add_argument("--year", required=True, type=int)
    add_paper_parser.add_argument("--venue")
    add_paper_parser.add_argument("--pdf-path")
    add_paper_parser.add_argument("--url")
    add_paper_parser.add_argument("--notes")
    add_paper_parser.add_argument("--cluster-id", type=int)

    list_papers_parser = subparsers.add_parser("list-papers", help="List papers")
    list_papers_parser.add_argument("--cluster-id", type=int)

    demo_parser = subparsers.add_parser(
        "research-demo",
        help="Run a local dry-run literature workflow from approved seed assets",
    )
    demo_parser.add_argument("--dry-run", action="store_true", help="Use only local seed assets")
    demo_parser.add_argument("--output", help="Markdown report path")
    demo_parser.add_argument("--approved-dir", help="Directory containing approved seed JSON files")

    query_parser = subparsers.add_parser(
        "query-literature",
        help="Answer from stored literature summaries and evidence for a topic",
    )
    query_parser.add_argument("--topic-id", required=True, type=int)
    query_parser.add_argument("--question", required=True)
    query_parser.add_argument("--max-results", default=5, type=int)

    memo_parser = subparsers.add_parser(
        "research-memo",
        help="Write a stored-evidence research memo for one topic and question",
    )
    memo_parser.add_argument("--topic-id", required=True, type=int)
    memo_parser.add_argument("--question", required=True)
    memo_parser.add_argument("--output", help="Markdown memo path")
    memo_parser.add_argument("--llm", action="store_true", help="Use OPENAI_API_KEY only to organize stored evidence")

    quality_parser = subparsers.add_parser(
        "quality-check-literature",
        help="Write a quality report for stored literature evidence and memo outputs",
    )
    quality_parser.add_argument("--topic-id", required=True, type=int)
    quality_parser.add_argument("--output", help="Markdown quality report path")

    verification_parser = subparsers.add_parser(
        "generate-verification-tasks",
        help="Write literature/theory verification tasks from stored evidence",
    )
    verification_parser.add_argument("--topic-id", required=True, type=int)
    verification_parser.add_argument("--output", help="Markdown verification task path")

    extract_parser = subparsers.add_parser("extract-from-text", help="Extract entries into pending queue")
    extract_parser.add_argument("--text", help="Text to extract from")
    extract_parser.add_argument("--file", help="Text file to extract from")
    extract_parser.add_argument("--prompt-file", help="Prompt file to prepend to the extraction text")
    extract_parser.add_argument("--paper-id", type=int, help="Source paper id to attach to extracted candidates")
    extract_parser.add_argument("--output-json", help="Write extraction metadata to this JSON path")

    pdf_extract_parser = subparsers.add_parser("extract-from-pdf", help="Extract entries from a local PDF into pending queue")
    pdf_extract_parser.add_argument("--paper-id", type=int, required=True)
    pdf_extract_parser.add_argument("--pdf", required=True)
    pdf_extract_parser.add_argument("--prompt-file")
    pdf_extract_parser.add_argument("--output-json")

    pending_parser = subparsers.add_parser("list-pending", help="List pending entries")
    pending_parser.add_argument("--status", default="pending", help="pending, flagged, approved, rejected, or all")
    pending_parser.add_argument("--entry-type", help="Filter by entry type")

    curate_parser = subparsers.add_parser("curate-pending", help="Analyze pending entries and print warnings")
    curate_parser.add_argument("--entry-type", help="Filter by entry type")
    curate_parser.add_argument("--output-report", help="Write a markdown curation report")

    detail_parser = subparsers.add_parser("show-pending-detail", help="Show one pending entry as JSON")
    detail_parser.add_argument("entry_id", nargs="?", type=int)
    detail_parser.add_argument("--pending-id", type=int)

    show_pending_parser = subparsers.add_parser("show-pending", help="Show one pending entry as JSON")
    show_pending_parser.add_argument("entry_id", nargs="?", type=int)
    show_pending_parser.add_argument("--pending-id", type=int)

    approve_parser = subparsers.add_parser("approve-pending", help="Approve a pending entry")
    approve_parser.add_argument("entry_id", nargs="?", type=int)
    approve_parser.add_argument("--pending-id", type=int)
    approve_parser.add_argument("--edits-file", help="JSON file whose payload replaces the pending payload before approval")
    approve_parser.add_argument("--reject", action="store_true", help="Compatibility: reject instead of approving")
    approve_parser.add_argument("--reason", help="Compatibility rejection reason")

    reject_parser = subparsers.add_parser("reject-pending", help="Reject a pending entry")
    reject_parser.add_argument("entry_id", nargs="?", type=int)
    reject_parser.add_argument("--pending-id", type=int)
    reject_parser.add_argument("--reason")

    flag_parser = subparsers.add_parser("flag-pending", help="Flag a pending entry for later review")
    flag_parser.add_argument("entry_id", nargs="?", type=int)
    flag_parser.add_argument("--pending-id", type=int)
    flag_parser.add_argument("--reason", required=True)

    cluster_parser = subparsers.add_parser("add-cluster", help="Add a research cluster")
    cluster_parser.add_argument("--name", required=True)
    cluster_parser.add_argument("--description")
    cluster_parser.add_argument("--status", default="active", choices=["active", "watchlist", "archived"])
    cluster_parser.add_argument("--priority", default=0, type=int)
    cluster_parser.add_argument("--notes")

    list_clusters_parser = subparsers.add_parser("list-clusters", help="List research clusters")
    list_clusters_parser.add_argument("--status")

    concept_parser = subparsers.add_parser("add-concept", help="Add an ontology concept")
    concept_parser.add_argument("--name", required=True)
    concept_parser.add_argument(
        "--type",
        required=True,
        choices=[
            "model",
            "objective",
            "strategy",
            "architecture",
            "complexity_class",
            "logic",
            "proof_technique",
            "reduction_type",
        ],
    )
    concept_parser.add_argument("--description")
    concept_parser.add_argument("--aliases", default="", help="Semicolon-separated aliases")
    concept_parser.add_argument("--notes")

    list_concepts_parser = subparsers.add_parser("list-concepts", help="List ontology concepts")
    list_concepts_parser.add_argument("--type")

    link_parser = subparsers.add_parser("link-concepts", help="Add a typed relation between concepts")
    link_parser.add_argument("--source", required=True, type=int)
    link_parser.add_argument("--target", required=True, type=int)
    link_parser.add_argument(
        "--relation",
        required=True,
        choices=[
            "generalizes",
            "specializes",
            "equivalent_to",
            "reduces_to",
            "uses",
            "conflicts_with",
            "related_to",
        ],
    )
    link_parser.add_argument("--notes")

    by_cluster_parser = subparsers.add_parser("theorems-by-cluster", help="List theorems for a cluster")
    by_cluster_parser.add_argument("cluster_id", type=int)

    by_model_parser = subparsers.add_parser("theorems-by-model", help="List theorems by model family")
    by_model_parser.add_argument("model_family")

    by_objective_parser = subparsers.add_parser("theorems-by-objective", help="List theorems by objective family")
    by_objective_parser.add_argument("objective_family")

    op_cluster_parser = subparsers.add_parser("open-problems-by-cluster", help="List open problems for a cluster")
    op_cluster_parser.add_argument("cluster_id", type=int)

    subparsers.add_parser("show-research-map", help="Print compact research map summary")

    conjecture_parser = subparsers.add_parser("add-conjecture", help="Add a conjecture")
    conjecture_parser.add_argument("--statement", required=True)
    conjecture_parser.add_argument("--title")
    conjecture_parser.add_argument("--description")
    conjecture_parser.add_argument("--priority", type=int)
    conjecture_parser.add_argument("--cluster-id", type=int)
    conjecture_parser.add_argument("--motivation")
    conjecture_parser.add_argument("--expected-status", default="unknown", choices=["true", "false", "unknown"])
    conjecture_parser.add_argument("--confidence", default="needs_review", choices=["pending", "verified", "rejected", "needs_review"])
    conjecture_parser.add_argument("--attack-plan")
    conjecture_parser.add_argument("--possible-counterexamples", default="", help="Semicolon-separated notes")
    conjecture_parser.add_argument("--status", default="active", choices=["active", "paused", "refuted", "proved", "abandoned"])
    conjecture_parser.add_argument("--notes")

    list_conjectures_parser = subparsers.add_parser("list-conjectures", help="List conjectures")
    list_conjectures_parser.add_argument("--cluster-id", type=int)

    show_conjecture_parser = subparsers.add_parser("show-conjecture", help="Show a conjecture")
    show_conjecture_parser.add_argument("conjecture_id", nargs="?", type=int)
    show_conjecture_parser.add_argument("--conjecture-id", type=int, dest="conjecture_id_flag")

    update_conjecture_parser = subparsers.add_parser("update-conjecture-status", help="Update conjecture status")
    update_conjecture_parser.add_argument("conjecture_id", type=int)
    update_conjecture_parser.add_argument("--status", required=True, choices=["active", "paused", "refuted", "proved", "abandoned"])

    generate_parser = subparsers.add_parser("generate-game", help="Generate a tiny ATS/CDM/2DM safety game")
    generate_parser.add_argument("--kind", default="ATS", choices=["ATS", "CDM", "2DM"])
    generate_parser.add_argument("--processes", type=int, default=2)
    generate_parser.add_argument("--states", type=int, default=2)
    generate_parser.add_argument("--depth", type=int, default=5, help="Included for workflow symmetry")
    generate_parser.add_argument("--seed", type=int)
    generate_parser.add_argument("--output", help="Write generated JSON to this path")

    brute_parser = subparsers.add_parser("brute-check", help="Run bounded memoryless distributed safety check")
    brute_parser.add_argument("--input", help="Game JSON file; if omitted, a simple ATS game is generated")
    brute_parser.add_argument("--depth", type=int, default=5)

    pipeline_parser = subparsers.add_parser("run-pipeline", help="Run one bounded manual pipeline step")
    pipeline_parser.add_argument("--cluster-id", required=True, type=int)
    pipeline_parser.add_argument("--mode", required=True, choices=["literature", "experiments"])

    artifact_parser = subparsers.add_parser("register-code-artifact", help="Register reusable code metadata")
    artifact_parser.add_argument("--name", required=True)
    artifact_parser.add_argument("--path", required=True)
    artifact_parser.add_argument(
        "--artifact-type",
        required=True,
        choices=["library", "python_module", "solver", "generator", "reduction", "checker", "proof_check", "experiment_script"],
    )
    artifact_parser.add_argument("--entrypoint")
    artifact_parser.add_argument("--language")
    artifact_parser.add_argument("--description")
    artifact_parser.add_argument("--cluster-id", type=int)
    artifact_parser.add_argument("--related-concepts", default="", help="Semicolon-separated concept ids or names")
    artifact_parser.add_argument("--related-conjectures", default="", help="Semicolon-separated conjecture ids")
    artifact_parser.add_argument("--tests-path")
    artifact_parser.add_argument("--status", default="draft", choices=["draft", "tested", "deprecated"])
    artifact_parser.add_argument("--notes")

    list_artifact_parser = subparsers.add_parser("list-code-artifacts", help="List registered code artifacts")
    list_artifact_parser.add_argument("--artifact-type")
    list_artifact_parser.add_argument("--status")

    show_artifact_parser = subparsers.add_parser("show-code-artifact", help="Show one code artifact")
    show_artifact_parser.add_argument("--artifact-id", required=True, type=int)

    update_artifact_parser = subparsers.add_parser("update-code-artifact-status", help="Update artifact status")
    update_artifact_parser.add_argument("--artifact-id", required=True, type=int)
    update_artifact_parser.add_argument("--status", required=True, choices=["draft", "tested", "deprecated"])

    run_experiment_parser = subparsers.add_parser("run-experiment", help="Run a command and store experiment metadata")
    run_experiment_parser.add_argument("--artifact-id", required=True, type=int)
    run_experiment_parser.add_argument("--command", required=True, dest="run_command")
    run_experiment_parser.add_argument("--input-path")
    run_experiment_parser.add_argument("--output-path")
    run_experiment_parser.add_argument("--cluster-id", type=int)
    run_experiment_parser.add_argument("--conjecture-id", type=int)
    run_experiment_parser.add_argument("--experiment-type")
    run_experiment_parser.add_argument("--notes")

    subparsers.add_parser("list-experiment-runs", help="List experiment runs")
    subparsers.add_parser("list-experiments", help="List experiment runs")

    show_run_parser = subparsers.add_parser("show-experiment-run", help="Show one experiment run")
    show_run_parser.add_argument("--run-id", required=True, type=int)

    show_experiment_parser = subparsers.add_parser("show-experiment", help="Show one experiment run")
    show_experiment_parser.add_argument("--run-id", required=True, type=int)

    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_output_encoding()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "init-db":
        db.initialize_database(args.db)
        print(f"Initialized database at {args.db}")
        return 0

    if args.command == "add-paper":
        paper_id = add_paper(
            title=args.title,
            authors=_split_semicolon(args.authors),
            year=args.year,
            venue=args.venue,
            pdf_path=args.pdf_path,
            url=args.url,
            notes=args.notes,
            cluster_id=args.cluster_id,
            db_path=args.db,
        )
        print(f"Added paper {paper_id}")
        return 0

    if args.command == "list-papers":
        with db.get_connection(args.db) as connection:
            db.create_tables(connection)
            papers = db.list_papers(connection, cluster_id=args.cluster_id)
        for paper in papers:
            authors = ", ".join(paper.authors)
            cluster = f" cluster={paper.cluster_id}" if paper.cluster_id else ""
            venue = f", {paper.venue}" if paper.venue else ""
            print(f"{paper.id}: {paper.title} ({paper.year}{venue}) - {authors}{cluster}")
        return 0

    if args.command == "research-demo":
        result = run_research_demo(
            dry_run=args.dry_run,
            approved_dir=args.approved_dir,
            output_path=args.output,
            db_path=args.db,
        )
        print(
            "Research demo complete: "
            f"topic_id={result.topic_id} "
            f"topics_created={result.topics_created} "
            f"papers_notes_loaded={result.notes_loaded} "
            f"summaries_created={result.summaries_created} "
            f"evidence_spans_linked={result.evidence_spans_linked} "
            f"output_files_written={result.output_files_written} "
            f"report={result.report_path}"
        )
        return 0

    if args.command == "query-literature":
        result = query_literature(
            topic_id=args.topic_id,
            question=args.question,
            max_results=args.max_results,
            db_path=args.db,
        )
        print(result.answer)
        return 0

    if args.command == "research-memo":
        result = write_research_memo(
            topic_id=args.topic_id,
            question=args.question,
            output_path=args.output,
            use_llm=args.llm,
            db_path=args.db,
        )
        mode = "llm-organized" if result.used_llm else "stored-evidence"
        print(
            f"Wrote research memo topic_id={result.topic_id} "
            f"evidence_count={result.evidence_count} mode={mode} path={result.memo_path}"
        )
        return 0

    if args.command == "quality-check-literature":
        result = quality_check_literature(
            topic_id=args.topic_id,
            output_path=args.output,
            db_path=args.db,
        )
        print(
            f"Wrote quality report topic_id={result.topic_id} "
            f"issues={result.issue_count} path={result.report_path}"
        )
        return 0

    if args.command == "generate-verification-tasks":
        result = generate_verification_tasks(
            topic_id=args.topic_id,
            output_path=args.output,
            db_path=args.db,
        )
        print(
            f"Wrote verification tasks topic_id={result.topic_id} "
            f"tasks={result.task_count} path={result.tasks_path}"
        )
        return 0

    if args.command == "extract-from-text":
        text = _with_prompt(_read_text_arg(args.text, args.file), args.prompt_file)
        client = LLMClient()
        entry_ids = extract_from_text(text, db_path=args.db, client=client)
        if args.paper_id:
            _annotate_pending_paper(args.db, entry_ids, args.paper_id)
        if args.output_json:
            _write_json(Path(args.output_json), {"pending_entry_ids": entry_ids, "paper_id": args.paper_id})
        mode = "dry-run" if client.dry_run else "provider"
        print(f"Inserted pending entries ({mode}): {', '.join(str(entry_id) for entry_id in entry_ids)}")
        return 0

    if args.command == "extract-from-pdf":
        text = _with_prompt(_read_pdfish_text(Path(args.pdf)), args.prompt_file)
        client = LLMClient()
        entry_ids = extract_from_text(text, db_path=args.db, client=client)
        _annotate_pending_paper(args.db, entry_ids, args.paper_id)
        if args.output_json:
            _write_json(
                Path(args.output_json),
                {"pending_entry_ids": entry_ids, "paper_id": args.paper_id, "pdf": args.pdf},
            )
        mode = "dry-run" if client.dry_run else "provider"
        print(f"Inserted pending entries from PDF ({mode}): {', '.join(str(entry_id) for entry_id in entry_ids)}")
        return 0

    if args.command == "list-pending":
        status = None if args.status == "all" else args.status
        with db.get_connection(args.db) as connection:
            db.create_tables(connection)
            entries = db.list_pending_entries(connection, status=status)
        if args.entry_type:
            entries = [entry for entry in entries if entry.entry_type == args.entry_type]
        for entry in entries:
            title = entry.payload.get("title") or entry.payload.get("name") or entry.payload.get("summary") or "(untitled)"
            warning_text = f" warnings={entry.warnings}" if entry.warnings else ""
            print(f"{entry.id}: {entry.entry_type} [{entry.status}] {str(title)[:100]}{warning_text}")
        return 0

    if args.command == "curate-pending":
        analyses = curate_pending(db_path=args.db)
        if args.entry_type:
            analyses = [analysis for analysis in analyses if analysis.entry.entry_type == args.entry_type]
        if args.output_report:
            _write_curation_report(Path(args.output_report), analyses)
        for analysis in analyses:
            title = analysis.entry.payload.get("title") or analysis.entry.payload.get("name") or "(untitled)"
            print(f"{analysis.entry.id}: {analysis.entry.entry_type} {title}")
            if analysis.warnings:
                print("  warnings: " + "; ".join(analysis.warnings))
            if analysis.duplicates:
                dupes = ", ".join(f"{d.table}:{d.entry_id}@{d.score:.2f}" for d in analysis.duplicates)
                print("  duplicates: " + dupes)
        return 0

    if args.command in {"show-pending-detail", "show-pending"}:
        entry_id = _resolve_id(args.entry_id, args.pending_id, "pending id")
        with db.get_connection(args.db) as connection:
            db.create_tables(connection)
            entry = db.get_pending_entry(connection, entry_id)
        if entry is None:
            raise SystemExit(f"pending entry {entry_id} does not exist")
        print(json.dumps(entry.model_dump(), indent=2, sort_keys=True))
        return 0

    if args.command == "approve-pending":
        entry_id = _resolve_id(args.entry_id, args.pending_id, "pending id")
        if args.reject:
            reject_pending(entry_id, reason=args.reason, db_path=args.db)
            print(f"Rejected pending entry {entry_id}")
            return 0
        if args.edits_file:
            _replace_pending_payload_from_file(args.db, entry_id, Path(args.edits_file))
        result = approve_pending(entry_id, db_path=args.db)
        print(f"Approved pending entry {result.pending_id} into {result.inserted_table}:{result.inserted_id}")
        if result.warnings:
            print("Warnings: " + "; ".join(result.warnings))
        return 0

    if args.command == "reject-pending":
        entry_id = _resolve_id(args.entry_id, args.pending_id, "pending id")
        reject_pending(entry_id, reason=args.reason, db_path=args.db)
        print(f"Rejected pending entry {entry_id}")
        return 0

    if args.command == "flag-pending":
        entry_id = _resolve_id(args.entry_id, args.pending_id, "pending id")
        flag_pending(entry_id, reason=args.reason, db_path=args.db)
        print(f"Flagged pending entry {entry_id}")
        return 0

    if args.command == "add-cluster":
        with db.get_connection(args.db) as connection:
            db.create_tables(connection)
            cluster_id = db.insert_cluster(
                connection,
                ResearchCluster(
                    name=args.name,
                    description=args.description,
                    status=args.status,
                    priority=args.priority,
                    notes=args.notes,
                ),
            )
        print(f"Added cluster {cluster_id}")
        return 0

    if args.command == "list-clusters":
        with db.get_connection(args.db) as connection:
            db.create_tables(connection)
            clusters = db.list_clusters(connection, status=args.status)
        for cluster in clusters:
            print(f"{cluster.cluster_id}: [{cluster.status}] p={cluster.priority} {cluster.name}")
        return 0

    if args.command == "add-concept":
        with db.get_connection(args.db) as connection:
            db.create_tables(connection)
            concept_id = db.insert_concept(
                connection,
                Concept(
                    name=args.name,
                    concept_type=args.type,
                    description=args.description,
                    aliases=_split_semicolon(args.aliases),
                    notes=args.notes,
                ),
            )
        print(f"Added concept {concept_id}")
        return 0

    if args.command == "list-concepts":
        with db.get_connection(args.db) as connection:
            db.create_tables(connection)
            concepts = db.list_concepts(connection, concept_type=args.type)
        for concept in concepts:
            aliases = f" aliases={concept.aliases}" if concept.aliases else ""
            print(f"{concept.concept_id}: [{concept.concept_type}] {concept.name}{aliases}")
        return 0

    if args.command == "link-concepts":
        with db.get_connection(args.db) as connection:
            db.create_tables(connection)
            db.insert_concept_link(
                connection,
                ConceptLink(
                    source_concept_id=args.source,
                    target_concept_id=args.target,
                    relation_type=args.relation,
                    notes=args.notes,
                ),
            )
        print(f"Linked concept {args.source} {args.relation} {args.target}")
        return 0

    if args.command == "theorems-by-cluster":
        _print_theorems(args.db, cluster_id=args.cluster_id)
        return 0

    if args.command == "theorems-by-model":
        _print_theorems(args.db, model_family=args.model_family)
        return 0

    if args.command == "theorems-by-objective":
        _print_theorems(args.db, objective_family=args.objective_family)
        return 0

    if args.command == "open-problems-by-cluster":
        with db.get_connection(args.db) as connection:
            db.create_tables(connection)
            problems = db.list_open_problems(connection, cluster_id=args.cluster_id)
        for problem in problems:
            print(f"{problem.id}: {problem.title} [{problem.status}]")
        return 0

    if args.command == "show-research-map":
        _print_research_map(args.db)
        return 0

    if args.command == "add-conjecture":
        with db.get_connection(args.db) as connection:
            db.create_tables(connection)
            conjecture_id = db.insert_conjecture(
                connection,
                Conjecture(
                    title=args.title,
                    statement=args.statement,
                    cluster_id=args.cluster_id,
                    motivation=args.motivation or args.description,
                    expected_status=args.expected_status,
                    confidence=args.confidence,
                    attack_plan=args.attack_plan,
                    possible_counterexamples=_split_semicolon(args.possible_counterexamples),
                    status=args.status,
                    notes=_combine_notes(args.notes, f"priority={args.priority}" if args.priority is not None else None),
                ),
            )
        print(f"Added conjecture {conjecture_id}")
        return 0

    if args.command == "list-conjectures":
        with db.get_connection(args.db) as connection:
            db.create_tables(connection)
            conjectures = db.list_conjectures(connection, cluster_id=args.cluster_id)
        for conjecture in conjectures:
            cluster = f" cluster={conjecture.cluster_id}" if conjecture.cluster_id else ""
            print(f"{conjecture.id}: [{conjecture.status}] {conjecture.title}{cluster}")
        return 0

    if args.command == "show-conjecture":
        conjecture_id = _resolve_id(args.conjecture_id, args.conjecture_id_flag, "conjecture id")
        with db.get_connection(args.db) as connection:
            db.create_tables(connection)
            conjecture = db.get_conjecture(connection, conjecture_id)
        if conjecture is None:
            raise SystemExit(f"conjecture {conjecture_id} does not exist")
        print(json.dumps(conjecture.model_dump(), indent=2, sort_keys=True))
        return 0

    if args.command == "update-conjecture-status":
        with db.get_connection(args.db) as connection:
            db.create_tables(connection)
            db.update_conjecture_status(connection, args.conjecture_id, args.status)
        print(f"Updated conjecture {args.conjecture_id} to {args.status}")
        return 0

    if args.command == "generate-game":
        game = generate_tiny_game(
            kind=args.kind,
            process_count=args.processes,
            states_per_process=args.states,
            seed=args.seed,
        )
        data = json.dumps(game.to_dict(), indent=2, sort_keys=True)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(data + "\n", encoding="utf-8")
            print(f"Wrote game to {output_path}")
        else:
            print(data)
        return 0

    if args.command == "brute-check":
        game = _load_game(args.input)
        result = find_memoryless_safety_strategy(game, depth=args.depth)
        print(f"winning={result.winning} checked_strategies={result.checked_strategies} depth={result.depth}")
        if result.strategy is not None:
            print(json.dumps(result.strategy, indent=2, sort_keys=True))
        if result.counterexample is not None:
            print("counterexample=" + json.dumps(result.counterexample))
        return 0

    if args.command == "run-pipeline":
        result = run_pipeline(cluster_id=args.cluster_id, mode=args.mode, db_path=args.db)
        print(result.summary)
        return 0

    if args.command == "register-code-artifact":
        artifact_id = register_code_artifact(
            name=args.name,
            path=args.path,
            artifact_type=args.artifact_type,
            entrypoint=args.entrypoint,
            language=args.language,
            description=args.description,
            cluster_id=args.cluster_id,
            related_concepts=_split_semicolon(args.related_concepts),
            related_conjectures=_split_ints(args.related_conjectures),
            tests_path=args.tests_path,
            status=args.status,
            notes=args.notes,
            db_path=args.db,
        )
        print(f"Registered code artifact {artifact_id}")
        return 0

    if args.command == "list-code-artifacts":
        artifacts = list_code_artifacts(
            artifact_type=args.artifact_type,
            status=args.status,
            db_path=args.db,
        )
        for artifact in artifacts:
            print(f"{artifact.artifact_id}: [{artifact.status}] {artifact.artifact_type} {artifact.name} -> {artifact.path}")
        return 0

    if args.command == "show-code-artifact":
        artifact = get_code_artifact(args.artifact_id, db_path=args.db)
        if artifact is None:
            raise SystemExit(f"code artifact {args.artifact_id} does not exist")
        print(json.dumps(artifact.model_dump(), indent=2, sort_keys=True))
        return 0

    if args.command == "update-code-artifact-status":
        update_code_artifact_status(args.artifact_id, args.status, db_path=args.db)
        print(f"Updated code artifact {args.artifact_id} to {args.status}")
        return 0

    if args.command == "run-experiment":
        execution = run_experiment(
            artifact_id=args.artifact_id,
            command=args.run_command,
            input_path=args.input_path,
            output_path=args.output_path,
            cluster_id=args.cluster_id,
            conjecture_id=args.conjecture_id,
            experiment_type=args.experiment_type,
            notes=args.notes,
            db_path=args.db,
        )
        print(
            f"Recorded experiment run {execution.run_id}: "
            f"{execution.result_summary} output={execution.result_file}"
        )
        return 0

    if args.command in {"list-experiment-runs", "list-experiments"}:
        with db.get_connection(args.db) as connection:
            db.create_tables(connection)
            runs = db.list_experiment_runs(connection)
        for run in runs:
            artifact = f" artifact={run.artifact_id}" if run.artifact_id else ""
            print(f"{run.run_id}: {run.experiment_type} {run.result_summary or ''}{artifact}")
        return 0

    if args.command in {"show-experiment-run", "show-experiment"}:
        with db.get_connection(args.db) as connection:
            db.create_tables(connection)
            run = db.get_experiment_run(connection, args.run_id)
        if run is None:
            raise SystemExit(f"experiment run {args.run_id} does not exist")
        print(json.dumps(run.model_dump(), indent=2, sort_keys=True))
        return 0

    parser.error(f"unknown command {args.command}")
    return 2


def _print_theorems(
    db_path: str,
    cluster_id: int | None = None,
    model_family: str | None = None,
    objective_family: str | None = None,
) -> None:
    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        theorems = db.list_theorems(
            connection,
            cluster_id=cluster_id,
            model_family=model_family,
            objective_family=objective_family,
        )
    for theorem in theorems:
        bounds = " ".join(
            part
            for part in (
                f"upper={theorem.complexity_upper}" if theorem.complexity_upper else "",
                f"lower={theorem.complexity_lower}" if theorem.complexity_lower else "",
            )
            if part
        )
        print(f"{theorem.id}: [{theorem.theorem_type}] {theorem.title} {bounds}".rstrip())


def _print_research_map(db_path: str) -> None:
    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        clusters = db.list_clusters(connection, status="active")
        papers = db.list_papers(connection)
        theorems = db.list_theorems(connection)
        problems = db.list_open_problems(connection)
        conjectures = db.list_conjectures(connection)
    print("Active clusters")
    for cluster in clusters[:10]:
        print(f"- {cluster.cluster_id}: {cluster.name}")
    print("Key papers")
    for paper in papers[:10]:
        print(f"- {paper.id}: {paper.title} ({paper.year})")
    print("Key theorems")
    for theorem in theorems[:10]:
        print(f"- {theorem.id}: {theorem.title} [{theorem.theorem_type}]")
    print("Known upper/lower bounds")
    for theorem in theorems:
        if theorem.complexity_upper or theorem.complexity_lower:
            print(f"- {theorem.title}: upper={theorem.complexity_upper} lower={theorem.complexity_lower}")
    print("Open gaps")
    for problem in problems[:10]:
        print(f"- {problem.id}: {problem.title}")
    print("Candidate conjectures")
    for conjecture in conjectures[:10]:
        print(f"- {conjecture.id}: {conjecture.title} [{conjecture.status}]")


def _split_semicolon(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(";") if item.strip()]


def _split_ints(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(";") if item.strip()]


def _print_warnings(warnings: list[str]) -> None:
    for warning in warnings:
        print(f"warning: {warning}")


def _resolve_id(positional: int | None, named: int | None, label: str) -> int:
    resolved = named if named is not None else positional
    if resolved is None:
        raise SystemExit(f"missing {label}")
    return resolved


def _with_prompt(text: str, prompt_file: str | None) -> str:
    if prompt_file is None:
        return text
    prompt = Path(prompt_file).read_text(encoding="utf-8")
    return f"{prompt}\n\n--- SOURCE TEXT ---\n{text}"


def _read_pdfish_text(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"PDF does not exist: {path}")
    data = path.read_bytes()
    # This is a dependency-free MVP fallback. Real PDF extraction can be swapped
    # in later; pending entries still require human curation.
    return data.decode("utf-8", errors="replace")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _annotate_pending_paper(db_path: str, entry_ids: list[int], paper_id: int) -> None:
    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        for entry_id in entry_ids:
            entry = db.get_pending_entry(connection, entry_id)
            if entry is None:
                continue
            payload = dict(entry.payload)
            payload.setdefault("paper_id", paper_id)
            payload.setdefault("source_paper_id", paper_id)
            db.update_pending_payload(connection, entry_id, payload)


def _replace_pending_payload_from_file(db_path: str, entry_id: int, edits_file: Path) -> None:
    payload = json.loads(edits_file.read_text(encoding="utf-8"))
    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        entry = db.get_pending_entry(connection, entry_id)
        if entry is None:
            raise SystemExit(f"pending entry {entry_id} does not exist")
        warnings = [warning for warning in entry.warnings if warning != "dry-run extraction"]
        db.update_pending_payload(connection, entry_id, payload, warnings=warnings)


def _write_curation_report(path: Path, analyses) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Pending Entry Curation Report", ""]
    for analysis in analyses:
        title = analysis.entry.payload.get("title") or analysis.entry.payload.get("name") or "(untitled)"
        lines.append(f"## {analysis.entry.id}: {analysis.entry.entry_type} - {title}")
        if analysis.warnings:
            lines.append("")
            lines.append("Warnings:")
            lines.extend(f"- {warning}" for warning in analysis.warnings)
        if analysis.duplicates:
            lines.append("")
            lines.append("Possible duplicates:")
            lines.extend(
                f"- {duplicate.table}:{duplicate.entry_id} score={duplicate.score:.2f} {duplicate.title}"
                for duplicate in analysis.duplicates
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _combine_notes(*parts: str | None) -> str | None:
    values = [part for part in parts if part]
    return "\n".join(values) if values else None


def _read_text_arg(text: str | None, file_path: str | None) -> str:
    if text is not None:
        return text
    if file_path is not None:
        return Path(file_path).read_text(encoding="utf-8")
    return sys.stdin.read()


def _load_game(file_path: str | None) -> SafetyGame:
    if file_path is None:
        return generate_tiny_game(kind="ATS", process_count=2, states_per_process=2, seed=0)
    return SafetyGame.from_dict(json.loads(Path(file_path).read_text(encoding="utf-8")))


def _configure_output_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    raise SystemExit(main())

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
from .experiment_manager import run_experiment
from .extract import LLMClient as ExtractionLLMClient, extract_from_text
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
from .llm import LLMClient as ProviderLLMClient
from .llm import LLMError
from .autonomous_research import AutonomousResearch
from .orchestrator import run_pipeline
from .research_units import add_paper_for_unit, record_partial_result
from .research_planning import plan_research
from .schemas import Concept, ConceptLink, Conjecture, ResearchTask, ResearchUnit, ResearchUnitLink


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Research harness for decidability, complexity, and strategy synthesis "
            "in distributed games and automata-theoretic synthesis."
        )
    )
    parser.add_argument("--db", default=db.DEFAULT_DATABASE_URL, help="PostgreSQL database URL")
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
    add_paper_parser.add_argument("--task-id", dest="task_id", type=int)
    add_paper_parser.add_argument("--unit-id", type=int, help="Literature activity that found this paper")
    list_papers_parser = subparsers.add_parser("list-papers", help="List papers")
    list_papers_parser.add_argument("--task-id", dest="task_id", type=int)

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
    memo_parser.add_argument("--llm", action="store_true", help="Use the configured LLM only to organize stored evidence")

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

    plan_parser = subparsers.add_parser(
        "plan-research",
        help="Propose reviewable research directions from stored state; never executes them",
    )
    plan_parser.add_argument("--goal", required=True, help="Free-form research objective")
    plan_parser.add_argument("--task-id", dest="task_id", type=int, help="Restrict planning to an existing research task")
    plan_parser.add_argument("--llm", action="store_true", help="Use the configured remote LLM provider")

    controller_parser = subparsers.add_parser(
        "research-loop",
        help="Run bounded autonomous research steps; each successful step creates a unit",
    )
    controller_parser.add_argument("--unit-id", type=int, help="Build the first step from this unit; otherwise choose automatically")
    controller_parser.add_argument("--task-id", dest="task_id", required=True, type=int)
    controller_parser.add_argument("--llm", action="store_true", help="Use the configured remote LLM provider")
    controller_parser.add_argument("--steps", default=1, type=int, help="Number of completed research units to create")
    controller_parser.add_argument("--literature-from-unit", action="store_true", help="Force full-text survey as the first step from --unit-id")

    extract_parser = subparsers.add_parser("extract-from-text", help="Extract entries into pending queue")
    extract_parser.add_argument("--text", help="Text to extract from")
    extract_parser.add_argument("--file", help="Text file to extract from")
    extract_parser.add_argument("--prompt-file", help="Prompt file to prepend to the extraction text")
    extract_parser.add_argument("--paper-id", type=int, help="Source paper id to attach to extracted candidates")
    extract_parser.add_argument("--output-json", help="Write extraction metadata to this JSON path")
    extract_parser.add_argument("--unit-id", type=int, help="Link pending candidates to a research activity")
    extract_parser.add_argument(
        "--llm",
        action="store_true",
        help="Use the explicitly configured remote LLM provider; otherwise use deterministic extraction",
    )

    pdf_extract_parser = subparsers.add_parser("extract-from-pdf", help="Extract entries from a local PDF into pending queue")
    pdf_extract_parser.add_argument("--paper-id", type=int, required=True)
    pdf_extract_parser.add_argument("--pdf", required=True)
    pdf_extract_parser.add_argument("--prompt-file")
    pdf_extract_parser.add_argument("--output-json")
    pdf_extract_parser.add_argument("--unit-id", type=int, help="Link pending candidates to a research activity")
    pdf_extract_parser.add_argument(
        "--llm",
        action="store_true",
        help="Use the explicitly configured remote LLM provider; otherwise use deterministic extraction",
    )


    task_parser = subparsers.add_parser("add-task", aliases=["add-research-task"], help="Add a research task")
    task_parser.add_argument("--name", required=True)
    task_parser.add_argument("--description", required=True, help="Fixed research objective")

    unit_parser = subparsers.add_parser("add-research-unit", help="Create one research activity in a task")
    unit_parser.add_argument("--task-id", dest="task_id", required=True, type=int)
    unit_parser.add_argument("--kind", default="investigation")
    unit_parser.add_argument("--title", required=True)
    unit_parser.add_argument("--purpose", required=True)
    unit_parser.add_argument("--parent-unit-id", type=int)
    unit_parser.add_argument("--priority", type=int, default=0)
    unit_parser.add_argument("--status", choices=["proposed", "ready", "active"], default="proposed")

    units_parser = subparsers.add_parser("list-research-units", help="List activities in creation order")
    units_parser.add_argument("--task-id", dest="task_id", type=int)
    units_parser.add_argument("--status")

    next_unit_parser = subparsers.add_parser("next-research-unit", help="Show the current frontier activity")
    next_unit_parser.add_argument("--task-id", dest="task_id", required=True, type=int)

    show_unit_parser = subparsers.add_parser("show-research-unit", help="Show an activity and its links")
    show_unit_parser.add_argument("--unit-id", required=True, type=int)

    update_unit_parser = subparsers.add_parser("update-research-unit", help="Update activity status and outcome")
    update_unit_parser.add_argument("--unit-id", required=True, type=int)
    update_unit_parser.add_argument("--status", required=True,
        choices=["proposed", "ready", "active", "blocked", "finished", "abandoned"])
    update_unit_parser.add_argument("--outcome-note")

    link_unit_parser = subparsers.add_parser("link-research-unit", help="Link an existing research object")
    link_unit_parser.add_argument("--unit-id", required=True, type=int)
    link_unit_parser.add_argument("--relation", required=True,
        choices=["investigates", "uses", "produces", "supports", "challenges"])
    link_unit_parser.add_argument("--object-type", required=True,
        choices=["paper", "concept", "model", "theorem", "reduction", "open_problem",
                 "conjecture", "derived_result", "proof_attempt", "evidence",
                 "literature_note", "literature_summary", "experiment_run",
                 "code_artifact", "pending_entry"])
    link_unit_parser.add_argument("--object-id", required=True, type=int)

    result_parser = subparsers.add_parser("record-unit-result", help="Record a draft finding and optional follow-up")
    result_parser.add_argument("--unit-id", required=True, type=int)
    result_parser.add_argument("--title", required=True)
    result_parser.add_argument("--statement", required=True)
    result_parser.add_argument("--notes")
    result_parser.add_argument("--followup-title")
    result_parser.add_argument("--followup-purpose")
    result_parser.add_argument("--followup-kind", default="investigation")

    subparsers.add_parser("list-research-tasks", aliases=["list-tasks"], help="List research tasks")

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

    by_task_parser = subparsers.add_parser("theorems-by-task", help="List theorems for a task")
    by_task_parser.add_argument("task_id", type=int)

    by_model_parser = subparsers.add_parser("theorems-by-model", help="List theorems by model family")
    by_model_parser.add_argument("model_family")

    by_objective_parser = subparsers.add_parser("theorems-by-objective", help="List theorems by objective family")
    by_objective_parser.add_argument("objective_family")

    op_task_parser = subparsers.add_parser("open-problems-by-task", help="List open problems for a task")
    op_task_parser.add_argument("task_id", type=int)

    subparsers.add_parser("show-research-map", help="Print compact research map summary")

    conjecture_parser = subparsers.add_parser("add-conjecture", help="Add a conjecture")
    conjecture_parser.add_argument("--statement", required=True)
    conjecture_parser.add_argument("--title")
    conjecture_parser.add_argument("--description")
    conjecture_parser.add_argument("--priority", type=int)
    conjecture_parser.add_argument("--task-id", dest="task_id", type=int)
    conjecture_parser.add_argument("--motivation")
    conjecture_parser.add_argument("--expected-status", default="unknown", choices=["true", "false", "unknown"])
    conjecture_parser.add_argument("--confidence", default="needs_review", choices=["pending", "verified", "rejected", "needs_review"])
    conjecture_parser.add_argument("--attack-plan")
    conjecture_parser.add_argument("--possible-counterexamples", default="", help="Semicolon-separated notes")
    conjecture_parser.add_argument("--status", default="active", choices=["active", "paused", "refuted", "proved", "abandoned"])
    conjecture_parser.add_argument("--notes")

    list_conjectures_parser = subparsers.add_parser("list-conjectures", help="List conjectures")
    list_conjectures_parser.add_argument("--task-id", dest="task_id", type=int)

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
    pipeline_parser.add_argument("--task-id", dest="task_id", required=True, type=int)
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
    artifact_parser.add_argument("--task-id", dest="task_id", type=int)
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
    run_experiment_parser.add_argument("--task-id", dest="task_id", type=int)
    run_experiment_parser.add_argument("--conjecture-id", type=int)
    run_experiment_parser.add_argument("--unit-id", type=int, help="Activity that produced this run")
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
        details = dict(
            title=args.title,
            authors=_split_semicolon(args.authors),
            year=args.year,
            venue=args.venue,
            pdf_path=args.pdf_path,
            url=args.url,
            notes=args.notes,
            db_path=args.db,
        )
        if args.unit_id is not None:
            unit = _require_unit(args.db, args.unit_id)
            if args.task_id is not None and args.task_id != unit.task_id:
                raise SystemExit("paper task and research unit task must match")
            paper_id = add_paper_for_unit(args.unit_id, **details)
        else:
            paper_id = add_paper(task_id=args.task_id, **details)
        print(f"Added paper {paper_id}")
        return 0

    if args.command == "list-papers":
        with db.get_connection(args.db) as connection:
            db.create_tables(connection)
            papers = db.list_papers(connection, task_id=args.task_id)
        for paper in papers:
            authors = ", ".join(paper.authors)
            task = f" task={paper.task_id}" if paper.task_id else ""
            venue = f", {paper.venue}" if paper.venue else ""
            print(f"{paper.id}: {paper.title} ({paper.year}{venue}) - {authors}{task}")
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

    if args.command == "plan-research":
        try:
            result = plan_research(
                args.goal,
                task_id=args.task_id,
                use_llm=args.llm,
                db_path=args.db,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if not result.available or result.plan is None:
            print(result.message)
            return 2
        payload = result.plan.model_dump()
        payload["provider"] = result.provider_metadata
        payload["message"] = result.message
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "research-loop":
        if not args.llm:
            print("research-loop requires --llm and a configured remote LLM provider; nothing was written.")
            return 2
        try:
            result = AutonomousResearch(ProviderLLMClient(), args.db).run(
                args.task_id, steps=args.steps, unit_id=args.unit_id,
                literature_from_unit=args.literature_from_unit,
            )
        except ValueError as exc:
            print(str(exc))
            return 2
        print(json.dumps({
            "status": result.status, "message": result.message,
            "task_id": result.task_id, "steps_completed": len(result.unit_ids),
            "research_unit_ids": result.unit_ids,
        }, indent=2, sort_keys=True))
        return 0 if result.status == "completed" else 2

    if args.command == "extract-from-text":
        if args.unit_id is not None:
            _require_unit(args.db, args.unit_id)
        text = _with_prompt(_read_text_arg(args.text, args.file), args.prompt_file)
        client = ExtractionLLMClient(use_configured_provider=args.llm)
        try:
            entry_ids = extract_from_text(text, db_path=args.db, client=client)
        except LLMError:
            print("LLM extraction failed validation; no entries were written.")
            return 2
        if args.paper_id:
            _annotate_pending_paper(args.db, entry_ids, args.paper_id)
        if args.unit_id is not None:
            _link_pending_to_unit(args.db, args.unit_id, entry_ids)
        if args.output_json:
            payload = {"pending_entry_ids": entry_ids, "paper_id": args.paper_id}
            if args.llm:
                payload["llm"] = client.metadata
            _write_json(Path(args.output_json), payload)
        mode = "provider" if client.used_remote else "dry-run"
        print(f"Inserted pending entries ({mode}): {', '.join(str(entry_id) for entry_id in entry_ids)}")
        return 0

    if args.command == "extract-from-pdf":
        if args.unit_id is not None:
            _require_unit(args.db, args.unit_id)
        text = _with_prompt(_read_pdfish_text(Path(args.pdf)), args.prompt_file)
        client = ExtractionLLMClient(use_configured_provider=args.llm)
        try:
            entry_ids = extract_from_text(text, db_path=args.db, client=client)
        except LLMError:
            print("LLM extraction failed validation; no entries were written.")
            return 2
        _annotate_pending_paper(args.db, entry_ids, args.paper_id)
        if args.unit_id is not None:
            _link_pending_to_unit(args.db, args.unit_id, entry_ids)
        if args.output_json:
            payload = {"pending_entry_ids": entry_ids, "paper_id": args.paper_id, "pdf": args.pdf}
            if args.llm:
                payload["llm"] = client.metadata
            _write_json(
                Path(args.output_json),
                payload,
            )
        mode = "provider" if client.used_remote else "dry-run"
        print(f"Inserted pending entries from PDF ({mode}): {', '.join(str(entry_id) for entry_id in entry_ids)}")
        return 0


    if args.command in {"add-task", "add-research-task"}:
        with db.get_connection(args.db) as connection:
            db.create_tables(connection)
            task_id = db.insert_task(
                connection,
                ResearchTask(name=args.name, description=args.description),
            )
        print(f"Added task {task_id}")
        return 0

    if args.command == "add-research-unit":
        with db.get_connection(args.db) as connection:
            db.create_tables(connection)
            unit_id = db.insert_research_unit(
                connection,
                ResearchUnit(
                    task_id=args.task_id,
                    kind=args.kind,
                    title=args.title,
                    purpose=args.purpose,
                    parent_unit_id=args.parent_unit_id,
                    priority=args.priority,
                    status=args.status,
                ),
            )
        print(f"Added research unit {unit_id}")
        return 0

    if args.command == "list-research-units":
        with db.get_connection(args.db) as connection:
            db.create_tables(connection)
            units = db.list_research_units(connection, task_id=args.task_id, status=args.status)
        for unit in units:
            parent = f" parent={unit.parent_unit_id}" if unit.parent_unit_id else ""
            print(f"{unit.unit_id}: task={unit.task_id} [{unit.status}] {unit.kind}: {unit.title}{parent}")
        return 0

    if args.command == "next-research-unit":
        with db.get_connection(args.db) as connection:
            db.create_tables(connection)
            if db.get_task(connection, args.task_id) is None:
                raise SystemExit(f"research task {args.task_id} does not exist")
            unit = db.next_research_unit(connection, args.task_id)
        if unit is None:
            print("No open research units in this task.")
            return 2
        print(json.dumps(unit.model_dump(), indent=2, sort_keys=True))
        return 0

    if args.command == "show-research-unit":
        with db.get_connection(args.db) as connection:
            db.create_tables(connection)
            unit = db.get_research_unit(connection, args.unit_id)
            if unit is None:
                raise SystemExit(f"research unit {args.unit_id} does not exist")
            links = db.list_research_unit_links(connection, args.unit_id)
            children = db.list_research_unit_children(connection, args.unit_id)
        print(json.dumps({
            "unit": unit.model_dump(),
            "links": [link.model_dump() for link in links],
            "children": [child.model_dump() for child in children],
        }, indent=2, sort_keys=True))
        return 0

    if args.command == "update-research-unit":
        with db.get_connection(args.db) as connection:
            db.create_tables(connection)
            unit = db.update_research_unit(
                connection, args.unit_id, args.status, outcome_note=args.outcome_note,
            )
        print(f"Updated research unit {unit.unit_id} to {unit.status}")
        return 0

    if args.command == "link-research-unit":
        with db.get_connection(args.db) as connection:
            db.create_tables(connection)
            db.link_research_unit(
                connection,
                ResearchUnitLink(
                    unit_id=args.unit_id,
                    relation=args.relation,
                    object_type=args.object_type,
                    object_id=args.object_id,
                ),
            )
        print(f"Linked {args.object_type} {args.object_id} to research unit {args.unit_id}")
        return 0

    if args.command == "record-unit-result":
        try:
            result_id, followup_id = record_partial_result(
                args.unit_id,
                args.title,
                args.statement,
                notes=args.notes,
                followup_title=args.followup_title,
                followup_purpose=args.followup_purpose,
                followup_kind=args.followup_kind,
                db_path=args.db,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"Recorded draft derived result {result_id}")
        if followup_id is not None:
            print(f"Created follow-up research unit {followup_id}")
        return 0

    if args.command in {"list-tasks", "list-research-tasks"}:
        with db.get_connection(args.db) as connection:
            db.create_tables(connection)
            tasks = db.list_tasks(connection)
        for task in tasks:
            print(f"{task.task_id}: {task.name} — {task.description}")
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

    if args.command == "theorems-by-task":
        _print_theorems(args.db, task_id=args.task_id)
        return 0

    if args.command == "theorems-by-model":
        _print_theorems(args.db, model_family=args.model_family)
        return 0

    if args.command == "theorems-by-objective":
        _print_theorems(args.db, objective_family=args.objective_family)
        return 0

    if args.command == "open-problems-by-task":
        with db.get_connection(args.db) as connection:
            db.create_tables(connection)
            problems = db.list_open_problems(connection, task_id=args.task_id)
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
                    task_id=args.task_id,
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
            conjectures = db.list_conjectures(connection, task_id=args.task_id)
        for conjecture in conjectures:
            task = f" task={conjecture.task_id}" if conjecture.task_id else ""
            print(f"{conjecture.id}: [{conjecture.status}] {conjecture.title}{task}")
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
        result = run_pipeline(task_id=args.task_id, mode=args.mode, db_path=args.db)
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
            task_id=args.task_id,
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
        unit = _require_unit(args.db, args.unit_id) if args.unit_id is not None else None
        if unit is not None and args.task_id is not None and args.task_id != unit.task_id:
            raise SystemExit("experiment task and research unit task must match")
        execution = run_experiment(
            artifact_id=args.artifact_id,
            command=args.run_command,
            input_path=args.input_path,
            output_path=args.output_path,
            task_id=unit.task_id if unit is not None else args.task_id,
            conjecture_id=args.conjecture_id,
            experiment_type=args.experiment_type,
            notes=args.notes,
            db_path=args.db,
        )
        if unit is not None:
            with db.get_connection(args.db) as connection:
                db.create_tables(connection)
                db.link_research_unit(
                    connection,
                    ResearchUnitLink(
                        unit_id=unit.unit_id,
                        relation="produces",
                        object_type="experiment_run",
                        object_id=execution.run_id,
                    ),
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
    task_id: int | None = None,
    model_family: str | None = None,
    objective_family: str | None = None,
) -> None:
    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        theorems = db.list_theorems(
            connection,
            task_id=task_id,
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
        tasks = db.list_tasks(connection)
        papers = db.list_papers(connection)
        theorems = db.list_theorems(connection)
        problems = db.list_open_problems(connection)
        conjectures = db.list_conjectures(connection)
    print("Research tasks")
    for task in tasks[:10]:
        print(f"- {task.task_id}: {task.name}")
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


def _require_unit(db_path: str, unit_id: int) -> ResearchUnit:
    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        unit = db.get_research_unit(connection, unit_id)
    if unit is None:
        raise SystemExit(f"research unit {unit_id} does not exist")
    return unit


def _link_pending_to_unit(db_path: str, unit_id: int, entry_ids: list[int]) -> None:
    with db.get_connection(db_path) as connection:
        db.create_tables(connection)
        unit = db.get_research_unit(connection, unit_id)
        if unit is None:
            raise ValueError(f"research unit {unit_id} does not exist")
        for entry_id in entry_ids:
            entry = db.get_pending_entry(connection, entry_id)
            if entry is None:
                raise ValueError(f"pending entry {entry_id} does not exist")
            if entry.entry_type in {"model", "theorem", "reduction", "open_problem", "conjecture_seed"}:
                payload = dict(entry.payload)
                payload["task_id"] = unit.task_id
                db.update_pending_payload(connection, entry_id, payload)
            db.link_research_unit(
                connection,
                ResearchUnitLink(
                    unit_id=unit_id,
                    relation="produces",
                    object_type="pending_entry",
                    object_id=entry_id,
                ),
            )


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

"""Argparse command-line interface for the research harness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import db
from .brute_solver import find_memoryless_safety_strategy
from .curate import approve_pending, reject_pending
from .extract import LLMClient, extract_from_text
from .generate_examples import generate_tiny_game
from .ingest import add_paper
from .models import SafetyGame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local research harness CLI")
    parser.add_argument("--db", default=str(db.DEFAULT_DB_PATH), help="SQLite database path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init-db", help="Create database tables")

    add_paper_parser = subparsers.add_parser("add-paper", help="Add a paper manually")
    add_paper_parser.add_argument("--title", required=True)
    add_paper_parser.add_argument("--authors", required=True, help="Semicolon-separated author list")
    add_paper_parser.add_argument("--year", required=True, type=int)
    add_paper_parser.add_argument("--venue")
    add_paper_parser.add_argument("--pdf-path")
    add_paper_parser.add_argument("--notes")

    subparsers.add_parser("list-papers", help="List papers")

    extract_parser = subparsers.add_parser("extract-from-text", help="Extract entries into pending queue")
    extract_parser.add_argument("--text", help="Text to extract from")
    extract_parser.add_argument("--file", help="Text file to extract from")

    pending_parser = subparsers.add_parser("list-pending", help="List pending entries")
    pending_parser.add_argument("--status", default="pending", help="pending, approved, rejected, or all")

    approve_parser = subparsers.add_parser("approve-pending", help="Approve or reject a pending entry")
    approve_parser.add_argument("entry_id", type=int)
    approve_parser.add_argument("--reject", action="store_true", help="Reject instead of approving")
    approve_parser.add_argument("--reason", help="Rejection reason")

    generate_parser = subparsers.add_parser("generate-game", help="Generate a tiny safety game")
    generate_parser.add_argument("--kind", default="ATS", choices=["ATS", "CDM", "2DM"])
    generate_parser.add_argument("--processes", type=int, default=2)
    generate_parser.add_argument("--states", type=int, default=2)
    generate_parser.add_argument("--depth", type=int, default=5, help="Included for workflow symmetry")
    generate_parser.add_argument("--seed", type=int)
    generate_parser.add_argument("--output", help="Write generated JSON to this path")

    brute_parser = subparsers.add_parser("brute-check", help="Run bounded brute-force safety check")
    brute_parser.add_argument("--input", help="Game JSON file; if omitted, a simple ATS game is generated")
    brute_parser.add_argument("--depth", type=int, default=5)

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
        authors = [author.strip() for author in args.authors.split(";") if author.strip()]
        paper_id = add_paper(
            title=args.title,
            authors=authors,
            year=args.year,
            venue=args.venue,
            pdf_path=args.pdf_path,
            notes=args.notes,
            db_path=args.db,
        )
        print(f"Added paper {paper_id}")
        return 0

    if args.command == "list-papers":
        with db.get_connection(args.db) as connection:
            db.create_tables(connection)
            papers = db.list_papers(connection)
        for paper in papers:
            authors = ", ".join(paper.authors)
            venue = f", {paper.venue}" if paper.venue else ""
            print(f"{paper.id}: {paper.title} ({paper.year}{venue}) - {authors}")
        return 0

    if args.command == "extract-from-text":
        text = _read_text_arg(args.text, args.file)
        client = LLMClient()
        entry_ids = extract_from_text(text, db_path=args.db, client=client)
        mode = "dry-run" if client.dry_run else "provider"
        print(f"Inserted pending entries ({mode}): {', '.join(str(entry_id) for entry_id in entry_ids)}")
        return 0

    if args.command == "list-pending":
        status = None if args.status == "all" else args.status
        with db.get_connection(args.db) as connection:
            db.create_tables(connection)
            entries = db.list_pending_entries(connection, status=status)
        for entry in entries:
            title = entry.payload.get("title") or entry.payload.get("name") or "(untitled)"
            warning_text = f" warnings={entry.warnings}" if entry.warnings else ""
            print(f"{entry.id}: {entry.entry_type} [{entry.status}] {title}{warning_text}")
        return 0

    if args.command == "approve-pending":
        if args.reject:
            reject_pending(args.entry_id, reason=args.reason, db_path=args.db)
            print(f"Rejected pending entry {args.entry_id}")
        else:
            result = approve_pending(args.entry_id, db_path=args.db)
            print(
                f"Approved pending entry {result.pending_id} into "
                f"{result.inserted_table}:{result.inserted_id}"
            )
            if result.warnings:
                print("Warnings: " + "; ".join(result.warnings))
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

    parser.error(f"unknown command {args.command}")
    return 2


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

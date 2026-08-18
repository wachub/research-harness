"""SQLite persistence helpers for the research harness."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from .schemas import (
    CodeArtifact,
    Concept,
    ConceptLink,
    Conjecture,
    DerivedResult,
    EvidenceSpan,
    ExperimentRun,
    LiteratureNote,
    LiteratureSummary,
    Model,
    OpenProblem,
    Paper,
    PendingEntry,
    ProofAttempt,
    Reduction,
    ResearchCluster,
    ResearchEvent,
    ResearchTopic,
    Theorem,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "research.db"

EXPLORABLE_TABLES: tuple[str, ...] = (
    "research_clusters",
    "concepts",
    "concept_links",
    "papers",
    "research_topics",
    "literature_notes",
    "literature_summaries",
    "models",
    "theorems",
    "reductions",
    "open_problems",
    "pending_entries",
    "derived_results",
    "conjectures",
    "proof_attempts",
    "evidence_spans",
    "code_artifacts",
    "experiment_runs",
    "research_events",
)


SEED_CLUSTERS: tuple[ResearchCluster, ...] = (
    ResearchCluster(
        name="Restricted multi-decision-maker synthesis",
        description="ATS/CDM/2DM frontier: decidability, complexity, and memory under causal memory; focus on central-decision-maker restrictions and small decidable extensions.",
        status="active",
        priority=1,
    ),
    ResearchCluster(
        name="Global objectives in causal-memory games",
        description="Petri/control/ATS results for global safety, good-and-bad markings, parity, and reductions that separate decidable safety from undecidable liveness.",
        status="active",
        priority=2,
    ),
    ResearchCluster(
        name="Acyclic architectures and automata-theoretic transfers",
        description="Zielonka automata, acyclic/tree architectures, decomposable games, and transfers between control-game and Petri-game formulations.",
        status="active",
        priority=3,
    ),
    ResearchCluster(
        name="Petri games and control games",
        description="Petri game, control-game, and causal memory formulations of synthesis.",
        status="watchlist",
        priority=6,
    ),
    ResearchCluster(
        name="Asynchronous automata and trace theory",
        description="Zielonka automata, Mazurkiewicz traces, trace languages, and distributed controllers.",
        status="active",
        priority=7,
    ),
    ResearchCluster(
        name="Games with imperfect information",
        description="Partial-information games, observation structures, and knowledge-based synthesis.",
        status="active",
        priority=7,
    ),
    ResearchCluster(
        name="Two-process distributed reachability",
        description="Decidability and complexity around two-process architectures and reachability objectives.",
        status="watchlist",
        priority=5,
    ),
    ResearchCluster(
        name="Logical/automata characterizations of distributed strategies",
        description="MSO, automata-theoretic, and strategy-language characterizations.",
        status="watchlist",
        priority=5,
    ),
)


SEED_CONCEPTS: tuple[Concept, ...] = (
    Concept(name="ATS games", concept_type="model", aliases=["asynchronous team synthesis"]),
    Concept(name="CDM games", concept_type="model", aliases=["concurrent decision-maker games"]),
    Concept(name="2DM games", concept_type="model", aliases=["two-decision-maker games"]),
    Concept(name="control games", concept_type="model"),
    Concept(name="Petri games", concept_type="model"),
    Concept(name="asynchronous automata", concept_type="model", aliases=["Zielonka automata"]),
    Concept(name="Mazurkiewicz traces", concept_type="model", aliases=["trace theory"]),
    Concept(name="distributed strategy", concept_type="strategy"),
    Concept(name="finite-state strategy", concept_type="strategy"),
    Concept(name="memory automaton", concept_type="strategy"),
    Concept(name="safety objective", concept_type="objective"),
    Concept(name="reachability objective", concept_type="objective"),
    Concept(name="parity objective", concept_type="objective"),
    Concept(name="global objective", concept_type="objective"),
    Concept(name="local objective", concept_type="objective"),
    Concept(name="decidability", concept_type="proof_technique"),
    Concept(name="EXPTIME-complete", concept_type="complexity_class"),
    Concept(name="PSPACE-hard", concept_type="complexity_class"),
    Concept(name="NEXPTIME upper bound", concept_type="complexity_class"),
    Concept(name="undecidability", concept_type="complexity_class"),
    Concept(name="reduction", concept_type="reduction_type"),
    Concept(name="fixed-point algorithm", concept_type="proof_technique"),
    Concept(name="linearization", concept_type="proof_technique"),
    Concept(name="gossip automaton", concept_type="model"),
)


def resolve_db_path(db_path: str | Path | None = None) -> Path:
    """Return a concrete database path and ensure its parent exists."""

    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open a SQLite connection with row dictionaries and foreign keys enabled."""

    connection = sqlite3.connect(resolve_db_path(db_path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(db_path: str | Path | None = None) -> None:
    """Create or migrate all harness tables and seed baseline ontology data."""

    with get_connection(db_path) as connection:
        create_tables(connection)


def create_tables(connection: sqlite3.Connection) -> None:
    """Create the database schema and apply lightweight additive migrations."""

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS research_clusters (
            cluster_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            description TEXT,
            status TEXT NOT NULL,
            priority INTEGER NOT NULL DEFAULT 0,
            notes TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS concepts (
            concept_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            concept_type TEXT NOT NULL,
            description TEXT,
            aliases_json TEXT NOT NULL,
            notes TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS concept_links (
            source_concept_id INTEGER NOT NULL,
            target_concept_id INTEGER NOT NULL,
            relation_type TEXT NOT NULL,
            notes TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (source_concept_id, target_concept_id, relation_type),
            FOREIGN KEY(source_concept_id) REFERENCES concepts(concept_id),
            FOREIGN KEY(target_concept_id) REFERENCES concepts(concept_id)
        );

        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            authors_json TEXT NOT NULL,
            year INTEGER NOT NULL,
            venue TEXT,
            pdf_path TEXT,
            url TEXT,
            notes TEXT,
            cluster_id INTEGER,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(cluster_id) REFERENCES research_clusters(cluster_id)
        );

        CREATE TABLE IF NOT EXISTS research_topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            raw_topic TEXT NOT NULL,
            clarified_topic TEXT NOT NULL,
            clarification_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS literature_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic_id INTEGER NOT NULL,
            paper_id INTEGER,
            source_path TEXT NOT NULL,
            note_type TEXT NOT NULL,
            title TEXT NOT NULL,
            content_json TEXT NOT NULL,
            markdown_note TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(topic_id) REFERENCES research_topics(id),
            FOREIGN KEY(paper_id) REFERENCES papers(id)
        );

        CREATE TABLE IF NOT EXISTS literature_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic_id INTEGER NOT NULL,
            note_id INTEGER,
            paper_id INTEGER,
            summary_json TEXT NOT NULL,
            markdown_summary TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(topic_id) REFERENCES research_topics(id),
            FOREIGN KEY(note_id) REFERENCES literature_notes(id),
            FOREIGN KEY(paper_id) REFERENCES papers(id)
        );

        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            model_type TEXT NOT NULL,
            description TEXT,
            data_json TEXT NOT NULL,
            source_paper_id INTEGER,
            cluster_id INTEGER,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(source_paper_id) REFERENCES papers(id),
            FOREIGN KEY(cluster_id) REFERENCES research_clusters(cluster_id)
        );

        CREATE TABLE IF NOT EXISTS theorems (
            theorem_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            statement TEXT NOT NULL,
            theorem_type TEXT NOT NULL,
            model_family TEXT,
            objective_family TEXT,
            architecture_assumptions_json TEXT NOT NULL,
            information_assumptions_json TEXT NOT NULL,
            strategy_assumptions_json TEXT NOT NULL,
            process_bound TEXT,
            complexity_upper TEXT,
            complexity_lower TEXT,
            memory_upper TEXT,
            memory_lower TEXT,
            source_paper_id INTEGER,
            source_location TEXT,
            proof_technique TEXT,
            confidence TEXT NOT NULL,
            cluster_id INTEGER,
            notes TEXT,
            assumptions_json TEXT NOT NULL,
            conclusion TEXT,
            paper_id INTEGER,
            tags_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(source_paper_id) REFERENCES papers(id),
            FOREIGN KEY(cluster_id) REFERENCES research_clusters(cluster_id)
        );

        CREATE TABLE IF NOT EXISTS reductions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            source_problem TEXT NOT NULL,
            target_problem TEXT NOT NULL,
            statement TEXT NOT NULL,
            assumptions_json TEXT NOT NULL,
            paper_id INTEGER,
            source_paper_id INTEGER,
            source_location TEXT,
            proof_technique TEXT,
            cluster_id INTEGER,
            tags_json TEXT NOT NULL,
            notes TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(source_paper_id) REFERENCES papers(id),
            FOREIGN KEY(cluster_id) REFERENCES research_clusters(cluster_id)
        );

        CREATE TABLE IF NOT EXISTS open_problems (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            statement TEXT NOT NULL,
            context TEXT,
            status TEXT NOT NULL,
            paper_id INTEGER,
            source_paper_id INTEGER,
            source_location TEXT,
            cluster_id INTEGER,
            tags_json TEXT NOT NULL,
            notes TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(source_paper_id) REFERENCES papers(id),
            FOREIGN KEY(cluster_id) REFERENCES research_clusters(cluster_id)
        );

        CREATE TABLE IF NOT EXISTS pending_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_type TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            source_text TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            duplicate_of TEXT,
            warnings_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            reviewed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS derived_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            statement TEXT NOT NULL,
            dependencies_json TEXT NOT NULL,
            proof_sketch TEXT,
            status TEXT NOT NULL,
            cluster_id INTEGER,
            notes TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(cluster_id) REFERENCES research_clusters(cluster_id)
        );

        CREATE TABLE IF NOT EXISTS conjectures (
            conjecture_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            statement TEXT NOT NULL,
            cluster_id INTEGER,
            motivation TEXT,
            related_theorems_json TEXT NOT NULL,
            expected_status TEXT NOT NULL,
            confidence TEXT NOT NULL,
            attack_plan TEXT,
            possible_counterexamples_json TEXT NOT NULL,
            status TEXT NOT NULL,
            notes TEXT,
            rationale TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(cluster_id) REFERENCES research_clusters(cluster_id)
        );

        CREATE TABLE IF NOT EXISTS proof_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            target_type TEXT NOT NULL,
            target_id INTEGER NOT NULL,
            strategy TEXT NOT NULL,
            notes TEXT,
            status TEXT NOT NULL,
            cluster_id INTEGER,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(cluster_id) REFERENCES research_clusters(cluster_id)
        );

        CREATE TABLE IF NOT EXISTS evidence_spans (
            evidence_id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id INTEGER NOT NULL,
            entry_type TEXT NOT NULL,
            entry_id INTEGER NOT NULL,
            page_start INTEGER,
            page_end INTEGER,
            quote_or_summary TEXT NOT NULL,
            confidence TEXT NOT NULL,
            notes TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(paper_id) REFERENCES papers(id)
        );

        CREATE TABLE IF NOT EXISTS code_artifacts (
            artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            path TEXT NOT NULL,
            artifact_type TEXT NOT NULL,
            entrypoint TEXT,
            language TEXT,
            description TEXT,
            cluster_id INTEGER,
            related_concepts TEXT NOT NULL,
            related_conjectures TEXT NOT NULL,
            tests_path TEXT,
            status TEXT NOT NULL,
            git_commit_hash TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            notes TEXT,
            FOREIGN KEY(cluster_id) REFERENCES research_clusters(cluster_id)
        );

        CREATE TABLE IF NOT EXISTS experiment_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            artifact_id INTEGER,
            cluster_id INTEGER,
            conjecture_id INTEGER,
            experiment_type TEXT NOT NULL,
            input_path TEXT,
            output_path TEXT,
            input_json TEXT NOT NULL,
            output_json TEXT NOT NULL,
            result_summary TEXT,
            command_run TEXT,
            git_commit_hash TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            notes TEXT,
            FOREIGN KEY(artifact_id) REFERENCES code_artifacts(artifact_id),
            FOREIGN KEY(cluster_id) REFERENCES research_clusters(cluster_id),
            FOREIGN KEY(conjecture_id) REFERENCES conjectures(conjecture_id)
        );

        CREATE TABLE IF NOT EXISTS research_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            cluster_id INTEGER,
            event_type TEXT NOT NULL,
            object_type TEXT NOT NULL,
            object_id INTEGER,
            summary TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(cluster_id) REFERENCES research_clusters(cluster_id)
        );

        CREATE INDEX IF NOT EXISTS idx_research_events_cluster_created
            ON research_events (cluster_id, created_at, event_id);
        """
    )
    _migrate_existing_tables(connection)
    _seed_defaults(connection)


def _migrate_existing_tables(connection: sqlite3.Connection) -> None:
    _add_missing_columns(
        connection,
        "papers",
        {"cluster_id": "INTEGER", "url": "TEXT"},
    )
    _add_missing_columns(
        connection,
        "research_topics",
        {
            "title": "TEXT NOT NULL DEFAULT 'Untitled topic'",
            "raw_topic": "TEXT NOT NULL DEFAULT ''",
            "clarified_topic": "TEXT NOT NULL DEFAULT ''",
            "clarification_json": "TEXT NOT NULL DEFAULT '{}'",
            "created_at": "TEXT",
        },
    )
    _add_missing_columns(
        connection,
        "literature_notes",
        {
            "topic_id": "INTEGER",
            "paper_id": "INTEGER",
            "source_path": "TEXT NOT NULL DEFAULT ''",
            "note_type": "TEXT NOT NULL DEFAULT 'note'",
            "title": "TEXT NOT NULL DEFAULT 'Untitled note'",
            "content_json": "TEXT NOT NULL DEFAULT '{}'",
            "markdown_note": "TEXT NOT NULL DEFAULT ''",
            "created_at": "TEXT",
        },
    )
    _add_missing_columns(
        connection,
        "literature_summaries",
        {
            "topic_id": "INTEGER",
            "note_id": "INTEGER",
            "paper_id": "INTEGER",
            "summary_json": "TEXT NOT NULL DEFAULT '{}'",
            "markdown_summary": "TEXT NOT NULL DEFAULT ''",
            "created_at": "TEXT",
        },
    )
    _add_missing_columns(
        connection,
        "models",
        {"cluster_id": "INTEGER"},
    )
    _add_missing_columns(
        connection,
        "theorems",
        {
            "title": "TEXT",
            "theorem_type": "TEXT NOT NULL DEFAULT 'characterization'",
            "model_family": "TEXT",
            "objective_family": "TEXT",
            "architecture_assumptions_json": "TEXT NOT NULL DEFAULT '[]'",
            "information_assumptions_json": "TEXT NOT NULL DEFAULT '[]'",
            "strategy_assumptions_json": "TEXT NOT NULL DEFAULT '[]'",
            "process_bound": "TEXT",
            "complexity_upper": "TEXT",
            "complexity_lower": "TEXT",
            "memory_upper": "TEXT",
            "memory_lower": "TEXT",
            "source_paper_id": "INTEGER",
            "source_location": "TEXT",
            "proof_technique": "TEXT",
            "confidence": "TEXT NOT NULL DEFAULT 'pending'",
            "cluster_id": "INTEGER",
            "notes": "TEXT",
            "assumptions_json": "TEXT NOT NULL DEFAULT '[]'",
            "conclusion": "TEXT",
            "paper_id": "INTEGER",
            "tags_json": "TEXT NOT NULL DEFAULT '[]'",
        },
    )
    _add_missing_columns(
        connection,
        "reductions",
        {
            "source_paper_id": "INTEGER",
            "source_location": "TEXT",
            "proof_technique": "TEXT",
            "cluster_id": "INTEGER",
            "notes": "TEXT",
        },
    )
    _add_missing_columns(
        connection,
        "open_problems",
        {
            "source_paper_id": "INTEGER",
            "source_location": "TEXT",
            "cluster_id": "INTEGER",
            "notes": "TEXT",
        },
    )
    _add_missing_columns(
        connection,
        "derived_results",
        {"cluster_id": "INTEGER", "notes": "TEXT"},
    )
    _add_missing_columns(
        connection,
        "conjectures",
        {
            "title": "TEXT",
            "cluster_id": "INTEGER",
            "motivation": "TEXT",
            "related_theorems_json": "TEXT NOT NULL DEFAULT '[]'",
            "expected_status": "TEXT NOT NULL DEFAULT 'unknown'",
            "confidence": "TEXT NOT NULL DEFAULT 'needs_review'",
            "attack_plan": "TEXT",
            "possible_counterexamples_json": "TEXT NOT NULL DEFAULT '[]'",
            "notes": "TEXT",
            "rationale": "TEXT",
        },
    )
    _add_missing_columns(connection, "proof_attempts", {"cluster_id": "INTEGER"})
    _add_missing_columns(
        connection,
        "experiment_runs",
        {
            "artifact_id": "INTEGER",
            "conjecture_id": "INTEGER",
            "input_path": "TEXT",
            "output_path": "TEXT",
            "command_run": "TEXT",
            "git_commit_hash": "TEXT",
            "input_json": "TEXT NOT NULL DEFAULT '{}'",
            "output_json": "TEXT NOT NULL DEFAULT '{}'",
        },
    )
    _add_missing_columns(
        connection,
        "code_artifacts",
        {
            "entrypoint": "TEXT",
            "language": "TEXT",
            "cluster_id": "INTEGER",
        },
    )


def _add_missing_columns(connection: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = set(_column_names(connection, table))
    for name, definition in columns.items():
        if name not in existing:
            try:
                connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise


def _column_names(connection: sqlite3.Connection, table: str) -> list[str]:
    rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    return [str(row["name"]) for row in rows]


def _pk_column(connection: sqlite3.Connection, table: str, preferred: str = "id") -> str:
    columns = set(_column_names(connection, table))
    if preferred in columns:
        return preferred
    return "id"


def _seed_defaults(connection: sqlite3.Connection) -> None:
    for cluster in SEED_CLUSTERS:
        insert_cluster(connection, cluster, ignore_existing=True)
    for concept in SEED_CONCEPTS:
        insert_concept(connection, concept, ignore_existing=True)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _json_loads(value: str | None, default: Any) -> Any:
    if value is None:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _last_insert_id(connection: sqlite3.Connection) -> int:
    row = connection.execute("SELECT last_insert_rowid() AS id").fetchone()
    return int(row["id"])


def insert_cluster(
    connection: sqlite3.Connection,
    cluster: ResearchCluster,
    ignore_existing: bool = False,
) -> int:
    """Insert a research cluster and return its id."""

    verb = "INSERT OR IGNORE" if ignore_existing else "INSERT"
    created = False
    try:
        cursor = connection.execute(
            f"""
            {verb} INTO research_clusters (name, description, status, priority, notes)
            VALUES (?, ?, ?, ?, ?)
            """,
            (cluster.name, cluster.description, cluster.status, cluster.priority, cluster.notes),
        )
        created = cursor.rowcount > 0
    except sqlite3.IntegrityError:
        connection.execute(
            """
            UPDATE research_clusters
            SET description = COALESCE(?, description),
                status = ?,
                priority = ?,
                notes = COALESCE(?, notes)
            WHERE name = ?
            """,
            (cluster.description, cluster.status, cluster.priority, cluster.notes, cluster.name),
        )
    row = connection.execute(
        "SELECT cluster_id FROM research_clusters WHERE name = ?",
        (cluster.name,),
    ).fetchone()
    cluster_id = int(row["cluster_id"])
    if created:
        _record_event(
            connection,
            cluster_id,
            "cluster_created",
            "research_cluster",
            cluster_id,
            f"Created research cluster: {cluster.name}",
        )
    return cluster_id


def list_clusters(connection: sqlite3.Connection, status: str | None = None) -> list[ResearchCluster]:
    """List research clusters ordered by priority and name."""

    if status:
        rows = connection.execute(
            "SELECT * FROM research_clusters WHERE status = ? ORDER BY priority DESC, name",
            (status,),
        ).fetchall()
    else:
        rows = connection.execute(
            "SELECT * FROM research_clusters ORDER BY priority DESC, name"
        ).fetchall()
    return [
        ResearchCluster(
            cluster_id=row["cluster_id"],
            name=row["name"],
            description=row["description"],
            status=row["status"],
            priority=row["priority"],
            notes=row["notes"],
        )
        for row in rows
    ]


def get_cluster(connection: sqlite3.Connection, cluster_id: int) -> ResearchCluster | None:
    row = connection.execute(
        "SELECT * FROM research_clusters WHERE cluster_id = ?",
        (cluster_id,),
    ).fetchone()
    if row is None:
        return None
    return ResearchCluster(
        cluster_id=row["cluster_id"],
        name=row["name"],
        description=row["description"],
        status=row["status"],
        priority=row["priority"],
        notes=row["notes"],
    )


def insert_concept(
    connection: sqlite3.Connection,
    concept: Concept,
    ignore_existing: bool = False,
) -> int:
    """Insert an ontology concept and return its id."""

    verb = "INSERT OR IGNORE" if ignore_existing else "INSERT"
    connection.execute(
        f"""
        {verb} INTO concepts (name, concept_type, description, aliases_json, notes)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            concept.name,
            concept.concept_type,
            concept.description,
            _json_dumps(concept.aliases),
            concept.notes,
        ),
    )
    row = connection.execute(
        "SELECT concept_id FROM concepts WHERE name = ?",
        (concept.name,),
    ).fetchone()
    return int(row["concept_id"])


def list_concepts(connection: sqlite3.Connection, concept_type: str | None = None) -> list[Concept]:
    """List ontology concepts."""

    if concept_type:
        rows = connection.execute(
            "SELECT * FROM concepts WHERE concept_type = ? ORDER BY name",
            (concept_type,),
        ).fetchall()
    else:
        rows = connection.execute("SELECT * FROM concepts ORDER BY concept_type, name").fetchall()
    return [_row_to_concept(row) for row in rows]


def get_concept(connection: sqlite3.Connection, concept_id: int) -> Concept | None:
    row = connection.execute("SELECT * FROM concepts WHERE concept_id = ?", (concept_id,)).fetchone()
    return _row_to_concept(row) if row else None


def find_concept_by_name_or_alias(connection: sqlite3.Connection, name: str) -> Concept | None:
    needle = name.strip().lower()
    for concept in list_concepts(connection):
        aliases = [alias.lower() for alias in concept.aliases]
        if concept.name.lower() == needle or needle in aliases:
            return concept
    return None


def insert_concept_link(connection: sqlite3.Connection, link: ConceptLink) -> None:
    """Insert or replace a typed concept relation."""

    connection.execute(
        """
        INSERT OR REPLACE INTO concept_links
            (source_concept_id, target_concept_id, relation_type, notes)
        VALUES (?, ?, ?, ?)
        """,
        (link.source_concept_id, link.target_concept_id, link.relation_type, link.notes),
    )


def list_concept_links(connection: sqlite3.Connection) -> list[ConceptLink]:
    rows = connection.execute(
        "SELECT * FROM concept_links ORDER BY source_concept_id, target_concept_id, relation_type"
    ).fetchall()
    return [
        ConceptLink(
            source_concept_id=row["source_concept_id"],
            target_concept_id=row["target_concept_id"],
            relation_type=row["relation_type"],
            notes=row["notes"],
        )
        for row in rows
    ]


def insert_paper(connection: sqlite3.Connection, paper: Paper) -> int:
    """Insert a paper and return its id."""

    existing = connection.execute(
        "SELECT id FROM papers WHERE title = ? AND year = ? ORDER BY id LIMIT 1",
        (paper.title, paper.year),
    ).fetchone()
    if existing:
        return int(existing["id"])

    connection.execute(
        """
        INSERT INTO papers (title, authors_json, year, venue, pdf_path, url, notes, cluster_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            paper.title,
            _json_dumps(paper.authors),
            paper.year,
            paper.venue,
            paper.pdf_path,
            paper.url,
            paper.notes,
            paper.cluster_id,
        ),
    )
    paper_id = _last_insert_id(connection)
    _record_event(
        connection,
        paper.cluster_id,
        "paper_added",
        "paper",
        paper_id,
        f"Added paper: {paper.title}",
    )
    return paper_id


def list_papers(connection: sqlite3.Connection, cluster_id: int | None = None) -> list[Paper]:
    """List papers ordered by insertion id."""

    if cluster_id is None:
        rows = connection.execute("SELECT * FROM papers ORDER BY id").fetchall()
    else:
        rows = connection.execute(
            "SELECT * FROM papers WHERE cluster_id = ? ORDER BY id",
            (cluster_id,),
        ).fetchall()
    return [_row_to_paper(row) for row in rows]


def get_paper(connection: sqlite3.Connection, paper_id: int) -> Paper | None:
    """Fetch one paper by id."""

    row = connection.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
    return _row_to_paper(row) if row else None


def find_paper_by_normalized_title(connection: sqlite3.Connection, title: str) -> Paper | None:
    """Find a paper by a conservative normalized title comparison."""

    needle = _normalize_title(title)
    for paper in list_papers(connection):
        if _normalize_title(paper.title) == needle:
            return paper
    return None


def update_paper_pdf_path(connection: sqlite3.Connection, paper_id: int, pdf_path: str) -> None:
    """Attach a local PDF path to an existing paper record."""

    connection.execute("UPDATE papers SET pdf_path = ? WHERE id = ?", (pdf_path, paper_id))


def insert_research_topic(connection: sqlite3.Connection, topic: ResearchTopic) -> int:
    """Insert a clarified literature review topic and return its id."""

    connection.execute(
        """
        INSERT INTO research_topics (title, raw_topic, clarified_topic, clarification_json)
        VALUES (?, ?, ?, ?)
        """,
        (
            topic.title,
            topic.raw_topic,
            topic.clarified_topic,
            _json_dumps(topic.clarification_json),
        ),
    )
    return _last_insert_id(connection)


def get_research_topic(connection: sqlite3.Connection, topic_id: int) -> ResearchTopic | None:
    """Fetch one literature review topic."""

    row = connection.execute("SELECT * FROM research_topics WHERE id = ?", (topic_id,)).fetchone()
    return _row_to_research_topic(row) if row else None


def list_research_topics(connection: sqlite3.Connection) -> list[ResearchTopic]:
    """List literature review topics ordered by insertion id."""

    rows = connection.execute("SELECT * FROM research_topics ORDER BY id").fetchall()
    return [_row_to_research_topic(row) for row in rows]


def insert_literature_note(connection: sqlite3.Connection, note: LiteratureNote) -> int:
    """Insert a local literature note and return its id."""

    connection.execute(
        """
        INSERT INTO literature_notes (
            topic_id, paper_id, source_path, note_type, title, content_json, markdown_note
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            note.topic_id,
            note.paper_id,
            note.source_path,
            note.note_type,
            note.title,
            _json_dumps(note.content_json),
            note.markdown_note,
        ),
    )
    return _last_insert_id(connection)


def list_literature_notes(
    connection: sqlite3.Connection,
    topic_id: int | None = None,
    paper_id: int | None = None,
) -> list[LiteratureNote]:
    """List local literature notes."""

    query = "SELECT * FROM literature_notes"
    clauses: list[str] = []
    params: list[Any] = []
    if topic_id is not None:
        clauses.append("topic_id = ?")
        params.append(topic_id)
    if paper_id is not None:
        clauses.append("paper_id = ?")
        params.append(paper_id)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY id"
    return [_row_to_literature_note(row) for row in connection.execute(query, params).fetchall()]


def insert_literature_summary(connection: sqlite3.Connection, summary: LiteratureSummary) -> int:
    """Insert a deterministic topic-specific literature summary and return its id."""

    connection.execute(
        """
        INSERT INTO literature_summaries (
            topic_id, note_id, paper_id, summary_json, markdown_summary
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            summary.topic_id,
            summary.note_id,
            summary.paper_id,
            _json_dumps(summary.summary_json),
            summary.markdown_summary,
        ),
    )
    return _last_insert_id(connection)


def list_literature_summaries(
    connection: sqlite3.Connection,
    topic_id: int | None = None,
    paper_id: int | None = None,
) -> list[LiteratureSummary]:
    """List deterministic topic-specific literature summaries."""

    query = "SELECT * FROM literature_summaries"
    clauses: list[str] = []
    params: list[Any] = []
    if topic_id is not None:
        clauses.append("topic_id = ?")
        params.append(topic_id)
    if paper_id is not None:
        clauses.append("paper_id = ?")
        params.append(paper_id)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY id"
    return [_row_to_literature_summary(row) for row in connection.execute(query, params).fetchall()]


def insert_model(connection: sqlite3.Connection, model: Model) -> int:
    """Insert a model record and return its id."""

    connection.execute(
        """
        INSERT INTO models (name, model_type, description, data_json, source_paper_id, cluster_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            model.name,
            model.model_type,
            model.description,
            _json_dumps(model.data),
            model.source_paper_id,
            model.cluster_id,
        ),
    )
    model_id = _last_insert_id(connection)
    _record_event(connection, model.cluster_id, "model_created", "model", model_id, f"Added model: {model.name}")
    return model_id


def list_models(connection: sqlite3.Connection) -> list[Model]:
    rows = connection.execute("SELECT * FROM models ORDER BY id").fetchall()
    return [
        Model(
            id=row["id"],
            name=row["name"],
            model_type=row["model_type"],
            description=row["description"],
            data=_json_loads(row["data_json"], {}),
            source_paper_id=row["source_paper_id"],
            cluster_id=row["cluster_id"],
        )
        for row in rows
    ]


def insert_theorem(connection: sqlite3.Connection, theorem: Theorem) -> int:
    """Insert a theorem-like research result and return its id."""

    connection.execute(
        """
        INSERT INTO theorems (
            title, statement, theorem_type, model_family, objective_family,
            architecture_assumptions_json, information_assumptions_json,
            strategy_assumptions_json, process_bound, complexity_upper,
            complexity_lower, memory_upper, memory_lower, source_paper_id,
            source_location, proof_technique, confidence, cluster_id, notes,
            assumptions_json, conclusion, paper_id, tags_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            theorem.title,
            theorem.statement,
            theorem.theorem_type,
            theorem.model_family,
            theorem.objective_family,
            _json_dumps(theorem.architecture_assumptions),
            _json_dumps(theorem.information_assumptions),
            _json_dumps(theorem.strategy_assumptions),
            theorem.process_bound,
            theorem.complexity_upper,
            theorem.complexity_lower,
            theorem.memory_upper,
            theorem.memory_lower,
            theorem.source_paper_id,
            theorem.source_location,
            theorem.proof_technique,
            theorem.confidence,
            theorem.cluster_id,
            theorem.notes,
            _json_dumps(theorem.assumptions),
            theorem.conclusion,
            theorem.paper_id,
            _json_dumps(theorem.tags),
        ),
    )
    theorem_id = _last_insert_id(connection)
    _record_event(
        connection,
        theorem.cluster_id,
        "theorem_created",
        "theorem",
        theorem_id,
        f"Added theorem: {theorem.title or theorem.statement[:80]}",
    )
    return theorem_id


def list_theorems(
    connection: sqlite3.Connection,
    cluster_id: int | None = None,
    model_family: str | None = None,
    objective_family: str | None = None,
) -> list[Theorem]:
    query = "SELECT * FROM theorems"
    clauses: list[str] = []
    params: list[Any] = []
    if cluster_id is not None:
        clauses.append("cluster_id = ?")
        params.append(cluster_id)
    if model_family is not None:
        clauses.append("LOWER(COALESCE(model_family, '')) = LOWER(?)")
        params.append(model_family)
    if objective_family is not None:
        clauses.append("LOWER(COALESCE(objective_family, '')) = LOWER(?)")
        params.append(objective_family)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY " + _pk_column(connection, "theorems", "theorem_id")
    return [_row_to_theorem(connection, row) for row in connection.execute(query, params).fetchall()]


def insert_reduction(connection: sqlite3.Connection, reduction: Reduction) -> int:
    """Insert a reduction record and return its id."""

    connection.execute(
        """
        INSERT INTO reductions (
            title, source_problem, target_problem, statement, assumptions_json,
            paper_id, source_paper_id, source_location, proof_technique,
            cluster_id, tags_json, notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            reduction.title,
            reduction.source_problem,
            reduction.target_problem,
            reduction.statement,
            _json_dumps(reduction.assumptions),
            reduction.paper_id,
            reduction.source_paper_id,
            reduction.source_location,
            reduction.proof_technique,
            reduction.cluster_id,
            _json_dumps(reduction.tags),
            reduction.notes,
        ),
    )
    reduction_id = _last_insert_id(connection)
    _record_event(
        connection,
        reduction.cluster_id,
        "reduction_created",
        "reduction",
        reduction_id,
        f"Added reduction: {reduction.title}",
    )
    return reduction_id


def list_reductions(connection: sqlite3.Connection, cluster_id: int | None = None) -> list[Reduction]:
    if cluster_id is None:
        rows = connection.execute("SELECT * FROM reductions ORDER BY id").fetchall()
    else:
        rows = connection.execute(
            "SELECT * FROM reductions WHERE cluster_id = ? ORDER BY id",
            (cluster_id,),
        ).fetchall()
    return [_row_to_reduction(row) for row in rows]


def insert_open_problem(connection: sqlite3.Connection, problem: OpenProblem) -> int:
    """Insert an open problem record and return its id."""

    connection.execute(
        """
        INSERT INTO open_problems (
            title, statement, context, status, paper_id, source_paper_id,
            source_location, cluster_id, tags_json, notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            problem.title,
            problem.statement,
            problem.context,
            problem.status,
            problem.paper_id,
            problem.source_paper_id,
            problem.source_location,
            problem.cluster_id,
            _json_dumps(problem.tags),
            problem.notes,
        ),
    )
    problem_id = _last_insert_id(connection)
    _record_event(
        connection,
        problem.cluster_id,
        "open_problem_created",
        "open_problem",
        problem_id,
        f"Added open problem: {problem.title}",
    )
    return problem_id


def list_open_problems(connection: sqlite3.Connection, cluster_id: int | None = None) -> list[OpenProblem]:
    if cluster_id is None:
        rows = connection.execute("SELECT * FROM open_problems ORDER BY id").fetchall()
    else:
        rows = connection.execute(
            "SELECT * FROM open_problems WHERE cluster_id = ? ORDER BY id",
            (cluster_id,),
        ).fetchall()
    return [_row_to_open_problem(row) for row in rows]


def insert_pending_entry(connection: sqlite3.Connection, entry: PendingEntry) -> int:
    """Insert a pending extracted entry and return its id."""

    connection.execute(
        """
        INSERT INTO pending_entries
            (entry_type, payload_json, source_text, status, duplicate_of, warnings_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            entry.entry_type,
            _json_dumps(entry.payload),
            entry.source_text,
            entry.status,
            entry.duplicate_of,
            _json_dumps(entry.warnings),
        ),
    )
    entry_id = _last_insert_id(connection)
    _record_event(
        connection,
        _cluster_id_from_pending(entry),
        "pending_created",
        "pending_entry",
        entry_id,
        f"Created pending {entry.entry_type} proposal",
        {"entry_type": entry.entry_type, "status": entry.status},
    )
    return entry_id


def get_pending_entry(connection: sqlite3.Connection, entry_id: int) -> PendingEntry | None:
    """Fetch a pending entry by id."""

    row = connection.execute("SELECT * FROM pending_entries WHERE id = ?", (entry_id,)).fetchone()
    return _row_to_pending_entry(row) if row else None


def list_pending_entries(
    connection: sqlite3.Connection,
    status: str | None = "pending",
) -> list[PendingEntry]:
    """List pending entries, optionally filtered by status."""

    if status is None:
        rows = connection.execute("SELECT * FROM pending_entries ORDER BY id").fetchall()
    else:
        rows = connection.execute(
            "SELECT * FROM pending_entries WHERE status = ? ORDER BY id",
            (status,),
        ).fetchall()
    return [_row_to_pending_entry(row) for row in rows]


def update_pending_status(
    connection: sqlite3.Connection,
    entry_id: int,
    status: str,
    duplicate_of: str | None = None,
    warnings: Iterable[str] | None = None,
) -> None:
    """Update curation status for a pending entry."""

    entry = get_pending_entry(connection, entry_id)
    if entry is None:
        raise ValueError(f"pending entry {entry_id} does not exist")
    connection.execute(
        """
        UPDATE pending_entries
        SET status = ?,
            duplicate_of = COALESCE(?, duplicate_of),
            warnings_json = COALESCE(?, warnings_json),
            reviewed_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (
            status,
            duplicate_of,
            _json_dumps(list(warnings)) if warnings is not None else None,
            entry_id,
        ),
    )
    _record_event(
        connection,
        _cluster_id_from_pending(entry),
        f"pending_{status}",
        "pending_entry",
        entry_id,
        f"Marked pending {entry.entry_type} proposal as {status}",
        {"entry_type": entry.entry_type, "duplicate_of": duplicate_of},
    )


def update_pending_payload(
    connection: sqlite3.Connection,
    entry_id: int,
    payload: dict[str, Any],
    warnings: Iterable[str] | None = None,
) -> None:
    """Replace the payload for a pending entry, optionally replacing warnings."""

    connection.execute(
        """
        UPDATE pending_entries
        SET payload_json = ?,
            warnings_json = COALESCE(?, warnings_json)
        WHERE id = ?
        """,
        (
            _json_dumps(payload),
            _json_dumps(list(warnings)) if warnings is not None else None,
            entry_id,
        ),
    )


def insert_derived_result(connection: sqlite3.Connection, result: DerivedResult) -> int:
    """Insert a derived result and return its id."""

    connection.execute(
        """
        INSERT INTO derived_results
            (title, statement, dependencies_json, proof_sketch, status, cluster_id, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result.title,
            result.statement,
            _json_dumps(result.dependencies),
            result.proof_sketch,
            result.status,
            result.cluster_id,
            result.notes,
        ),
    )
    result_id = _last_insert_id(connection)
    _record_event(
        connection,
        result.cluster_id,
        "derived_result_created",
        "derived_result",
        result_id,
        f"Added derived result: {result.title}",
    )
    return result_id


def list_derived_results(connection: sqlite3.Connection, cluster_id: int | None = None) -> list[DerivedResult]:
    """List derived results ordered by insertion id."""

    if cluster_id is None:
        rows = connection.execute("SELECT * FROM derived_results ORDER BY id").fetchall()
    else:
        rows = connection.execute(
            "SELECT * FROM derived_results WHERE cluster_id = ? ORDER BY id",
            (cluster_id,),
        ).fetchall()
    return [_row_to_derived_result(row) for row in rows]


def update_derived_result_status(connection: sqlite3.Connection, result_id: int, status: str) -> None:
    connection.execute("UPDATE derived_results SET status = ? WHERE id = ?", (status, result_id))


def insert_conjecture(connection: sqlite3.Connection, conjecture: Conjecture) -> int:
    """Insert a conjecture and return its id."""

    connection.execute(
        """
        INSERT INTO conjectures (
            title, statement, cluster_id, motivation, related_theorems_json,
            expected_status, confidence, attack_plan, possible_counterexamples_json,
            status, notes, rationale
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            conjecture.title,
            conjecture.statement,
            conjecture.cluster_id,
            conjecture.motivation,
            _json_dumps(conjecture.related_theorems),
            conjecture.expected_status,
            conjecture.confidence,
            conjecture.attack_plan,
            _json_dumps(conjecture.possible_counterexamples),
            conjecture.status,
            conjecture.notes,
            conjecture.rationale,
        ),
    )
    conjecture_id = _last_insert_id(connection)
    _record_event(
        connection,
        conjecture.cluster_id,
        "conjecture_created",
        "conjecture",
        conjecture_id,
        f"Added conjecture: {conjecture.title or conjecture.statement[:80]}",
    )
    return conjecture_id


def list_conjectures(connection: sqlite3.Connection, cluster_id: int | None = None) -> list[Conjecture]:
    """List conjectures ordered by insertion id."""

    if cluster_id is None:
        rows = connection.execute("SELECT * FROM conjectures ORDER BY " + _pk_column(connection, "conjectures", "conjecture_id")).fetchall()
    else:
        rows = connection.execute(
            "SELECT * FROM conjectures WHERE cluster_id = ? ORDER BY " + _pk_column(connection, "conjectures", "conjecture_id"),
            (cluster_id,),
        ).fetchall()
    return [_row_to_conjecture(connection, row) for row in rows]


def get_conjecture(connection: sqlite3.Connection, conjecture_id: int) -> Conjecture | None:
    pk = _pk_column(connection, "conjectures", "conjecture_id")
    row = connection.execute(f"SELECT * FROM conjectures WHERE {pk} = ?", (conjecture_id,)).fetchone()
    return _row_to_conjecture(connection, row) if row else None


def update_conjecture_status(connection: sqlite3.Connection, conjecture_id: int, status: str) -> None:
    conjecture = get_conjecture(connection, conjecture_id)
    if conjecture is None:
        raise ValueError(f"conjecture {conjecture_id} does not exist")
    pk = _pk_column(connection, "conjectures", "conjecture_id")
    connection.execute(f"UPDATE conjectures SET status = ? WHERE {pk} = ?", (status, conjecture_id))
    _record_event(
        connection,
        conjecture.cluster_id,
        "conjecture_status_changed",
        "conjecture",
        conjecture_id,
        f"Changed conjecture status to {status}: {conjecture.title or conjecture.statement[:80]}",
        {"previous_status": conjecture.status, "status": status},
    )


def insert_proof_attempt(connection: sqlite3.Connection, attempt: ProofAttempt) -> int:
    """Insert a proof attempt and return its id."""

    connection.execute(
        """
        INSERT INTO proof_attempts (target_type, target_id, strategy, notes, status, cluster_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            attempt.target_type,
            attempt.target_id,
            attempt.strategy,
            attempt.notes,
            attempt.status,
            attempt.cluster_id,
        ),
    )
    attempt_id = _last_insert_id(connection)
    _record_event(
        connection,
        attempt.cluster_id,
        "proof_attempt_created",
        "proof_attempt",
        attempt_id,
        f"Added proof attempt for {attempt.target_type}:{attempt.target_id}",
    )
    return attempt_id


def list_proof_attempts(connection: sqlite3.Connection, cluster_id: int | None = None) -> list[ProofAttempt]:
    """List proof attempts ordered by insertion id."""

    if cluster_id is None:
        rows = connection.execute("SELECT * FROM proof_attempts ORDER BY id").fetchall()
    else:
        rows = connection.execute(
            "SELECT * FROM proof_attempts WHERE cluster_id = ? ORDER BY id",
            (cluster_id,),
        ).fetchall()
    return [_row_to_proof_attempt(row) for row in rows]


def update_proof_attempt_status(connection: sqlite3.Connection, attempt_id: int, status: str) -> None:
    connection.execute("UPDATE proof_attempts SET status = ? WHERE id = ?", (status, attempt_id))


def insert_evidence_span(connection: sqlite3.Connection, evidence: EvidenceSpan) -> int:
    """Insert an evidence span and return its id."""

    connection.execute(
        """
        INSERT INTO evidence_spans (
            paper_id, entry_type, entry_id, page_start, page_end,
            quote_or_summary, confidence, notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            evidence.paper_id,
            evidence.entry_type,
            evidence.entry_id,
            evidence.page_start,
            evidence.page_end,
            evidence.quote_or_summary,
            evidence.confidence,
            evidence.notes,
        ),
    )
    evidence_id = _last_insert_id(connection)
    paper = get_paper(connection, evidence.paper_id)
    _record_event(
        connection,
        paper.cluster_id if paper else None,
        "evidence_added",
        "evidence_span",
        evidence_id,
        f"Added evidence span for paper {evidence.paper_id}",
        {"paper_id": evidence.paper_id, "entry_type": evidence.entry_type, "entry_id": evidence.entry_id},
    )
    return evidence_id


def list_evidence_spans(connection: sqlite3.Connection, entry_type: str | None = None, entry_id: int | None = None) -> list[EvidenceSpan]:
    query = "SELECT * FROM evidence_spans"
    clauses: list[str] = []
    params: list[Any] = []
    if entry_type is not None:
        clauses.append("entry_type = ?")
        params.append(entry_type)
    if entry_id is not None:
        clauses.append("entry_id = ?")
        params.append(entry_id)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY evidence_id"
    return [_row_to_evidence_span(row) for row in connection.execute(query, params).fetchall()]


def insert_code_artifact(connection: sqlite3.Connection, artifact: CodeArtifact) -> int:
    """Insert code-artifact metadata and return its id."""

    connection.execute(
        """
        INSERT INTO code_artifacts (
            name, path, artifact_type, entrypoint, language, description, cluster_id, related_concepts,
            related_conjectures, tests_path, status, git_commit_hash, notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            artifact.name,
            artifact.path,
            artifact.artifact_type,
            artifact.entrypoint,
            artifact.language,
            artifact.description,
            artifact.cluster_id,
            _json_dumps(artifact.related_concepts),
            _json_dumps(artifact.related_conjectures),
            artifact.tests_path,
            artifact.status,
            artifact.git_commit_hash,
            artifact.notes,
        ),
    )
    artifact_id = _last_insert_id(connection)
    _record_event(
        connection,
        artifact.cluster_id,
        "code_artifact_registered",
        "code_artifact",
        artifact_id,
        f"Registered code artifact: {artifact.name}",
        {"artifact_type": artifact.artifact_type, "status": artifact.status},
    )
    return artifact_id


def list_code_artifacts(
    connection: sqlite3.Connection,
    artifact_type: str | None = None,
    status: str | None = None,
) -> list[CodeArtifact]:
    """List registered code artifacts."""

    query = "SELECT * FROM code_artifacts"
    clauses: list[str] = []
    params: list[Any] = []
    if artifact_type is not None:
        clauses.append("artifact_type = ?")
        params.append(artifact_type)
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY artifact_id"
    return [_row_to_code_artifact(row) for row in connection.execute(query, params).fetchall()]


def get_code_artifact(connection: sqlite3.Connection, artifact_id: int) -> CodeArtifact | None:
    """Fetch one code artifact by id."""

    row = connection.execute(
        "SELECT * FROM code_artifacts WHERE artifact_id = ?",
        (artifact_id,),
    ).fetchone()
    return _row_to_code_artifact(row) if row else None


def update_code_artifact_status(connection: sqlite3.Connection, artifact_id: int, status: str) -> None:
    """Update a code artifact status."""

    connection.execute(
        "UPDATE code_artifacts SET status = ? WHERE artifact_id = ?",
        (status, artifact_id),
    )


def insert_experiment_run(connection: sqlite3.Connection, run: ExperimentRun) -> int:
    """Insert a small experiment run and return its id."""

    connection.execute(
        """
        INSERT INTO experiment_runs
            (
                artifact_id, cluster_id, conjecture_id, experiment_type,
                input_path, output_path, input_json, output_json,
                result_summary, command_run, git_commit_hash, notes
            )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run.artifact_id,
            run.cluster_id,
            run.conjecture_id,
            run.experiment_type,
            run.input_path,
            run.output_path,
            _json_dumps(run.input_json),
            _json_dumps(run.output_json),
            run.result_summary,
            run.command_run,
            run.git_commit_hash,
            run.notes,
        ),
    )
    run_id = _last_insert_id(connection)
    _record_event(
        connection,
        run.cluster_id,
        "experiment_completed",
        "experiment_run",
        run_id,
        f"Recorded experiment: {run.experiment_type}",
        {"artifact_id": run.artifact_id, "conjecture_id": run.conjecture_id, "result_summary": run.result_summary},
    )
    return run_id


def list_experiment_runs(connection: sqlite3.Connection, cluster_id: int | None = None) -> list[ExperimentRun]:
    """List stored experiment runs."""

    if cluster_id is None:
        rows = connection.execute("SELECT * FROM experiment_runs ORDER BY run_id").fetchall()
    else:
        rows = connection.execute(
            "SELECT * FROM experiment_runs WHERE cluster_id = ? ORDER BY run_id",
            (cluster_id,),
        ).fetchall()
    return [_row_to_experiment_run(row) for row in rows]


def get_experiment_run(connection: sqlite3.Connection, run_id: int) -> ExperimentRun | None:
    """Fetch one experiment run by id."""

    row = connection.execute("SELECT * FROM experiment_runs WHERE run_id = ?", (run_id,)).fetchone()
    return _row_to_experiment_run(row) if row else None


def insert_research_event(connection: sqlite3.Connection, event: ResearchEvent) -> int:
    """Append one project-history event without modifying the referenced object."""

    connection.execute(
        """
        INSERT INTO research_events
            (cluster_id, event_type, object_type, object_id, summary, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            event.cluster_id,
            event.event_type,
            event.object_type,
            event.object_id,
            event.summary,
            _json_dumps(event.metadata),
        ),
    )
    return _last_insert_id(connection)


def list_research_events(
    connection: sqlite3.Connection,
    cluster_id: int | None = None,
    limit: int | None = None,
) -> list[ResearchEvent]:
    """List append-only timeline events in chronological order."""

    query = "SELECT * FROM research_events"
    params: list[Any] = []
    if cluster_id is not None:
        query += " WHERE cluster_id = ?"
        params.append(cluster_id)
    query += " ORDER BY created_at, event_id"
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    return [_row_to_research_event(row) for row in connection.execute(query, params).fetchall()]


def list_explorer_records(
    connection: sqlite3.Connection,
    table_name: str,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Return safely decoded rows for one allow-listed table; never run user SQL."""

    if table_name not in EXPLORABLE_TABLES:
        raise ValueError(f"unsupported table: {table_name}")
    if limit < 1 or limit > 5_000:
        raise ValueError("limit must be between 1 and 5000")
    rows = connection.execute(f"SELECT * FROM {table_name} ORDER BY rowid DESC LIMIT ?", (limit,)).fetchall()
    return [_decode_explorer_row(row) for row in rows]


def list_project_timeline(connection: sqlite3.Connection, cluster_id: int) -> list[dict[str, Any]]:
    """Merge append-only events with timestamped records created before event logging."""

    events = [
        {
            "timestamp": event.created_at,
            "event_type": event.event_type,
            "object_type": event.object_type,
            "object_id": event.object_id,
            "summary": event.summary,
            "metadata": event.metadata,
            "source": "event",
        }
        for event in list_research_events(connection, cluster_id=cluster_id)
    ]
    existing = {(event["object_type"], event["object_id"]) for event in events}
    legacy_specs = (
        ("research_clusters", "cluster_created", "research_cluster", "cluster_id", "name", "cluster_id = ?"),
        ("papers", "paper_added", "paper", "id", "title", "cluster_id = ?"),
        ("theorems", "theorem_created", "theorem", "theorem_id", "title", "cluster_id = ?"),
        ("reductions", "reduction_created", "reduction", "id", "title", "cluster_id = ?"),
        ("open_problems", "open_problem_created", "open_problem", "id", "title", "cluster_id = ?"),
        ("derived_results", "derived_result_created", "derived_result", "id", "title", "cluster_id = ?"),
        ("conjectures", "conjecture_created", "conjecture", "conjecture_id", "title", "cluster_id = ?"),
        ("proof_attempts", "proof_attempt_created", "proof_attempt", "id", "strategy", "cluster_id = ?"),
        ("code_artifacts", "code_artifact_registered", "code_artifact", "artifact_id", "name", "cluster_id = ?"),
        ("experiment_runs", "experiment_completed", "experiment_run", "run_id", "experiment_type", "cluster_id = ?"),
        (
            "evidence_spans JOIN papers ON papers.id = evidence_spans.paper_id",
            "evidence_added",
            "evidence_span",
            "evidence_id",
            "quote_or_summary",
            "papers.cluster_id = ?",
        ),
    )
    for table, event_type, object_type, id_column, summary_column, where_clause in legacy_specs:
        timestamp_column = "evidence_spans.created_at" if table.startswith("evidence_spans ") else "created_at"
        rows = connection.execute(
            f"SELECT {id_column} AS object_id, {summary_column} AS summary, {timestamp_column} AS created_at FROM {table} WHERE {where_clause}",
            (cluster_id,),
        ).fetchall()
        for row in rows:
            key = (object_type, row["object_id"])
            if key not in existing:
                events.append(
                    {
                        "timestamp": row["created_at"],
                        "event_type": event_type,
                        "object_type": object_type,
                        "object_id": row["object_id"],
                        "summary": str(row["summary"] or object_type),
                        "metadata": {},
                        "source": "legacy",
                    }
                )
    for row in list_explorer_records(connection, "pending_entries", limit=5_000):
        payload = row.get("payload_json")
        if not isinstance(payload, dict) or payload.get("cluster_id") != cluster_id:
            continue
        key = ("pending_entry", row.get("id"))
        if key not in existing:
            events.append(
                {
                    "timestamp": row.get("created_at"),
                    "event_type": "pending_created",
                    "object_type": "pending_entry",
                    "object_id": row.get("id"),
                    "summary": f"Created pending {row.get('entry_type', 'research')} proposal",
                    "metadata": {"status": row.get("status")},
                    "source": "legacy",
                }
            )
        if row.get("reviewed_at"):
            events.append(
                {
                    "timestamp": row["reviewed_at"],
                    "event_type": f"pending_{row.get('status')}",
                    "object_type": "pending_entry",
                    "object_id": row.get("id"),
                    "summary": f"Reviewed pending {row.get('entry_type', 'research')} proposal",
                    "metadata": {"status": row.get("status")},
                    "source": "legacy",
                }
            )
    return sorted(events, key=lambda event: (event["timestamp"] or "", event["object_type"], event["object_id"] or 0))


def _decode_explorer_row(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    for key, value in tuple(result.items()):
        if key.endswith("_json") and isinstance(value, str):
            result[key] = _json_loads(value, {"_malformed_json": value})
    return result


def _record_event(
    connection: sqlite3.Connection,
    cluster_id: int | None,
    event_type: str,
    object_type: str,
    object_id: int | None,
    summary: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Record concise provenance from a core write path when a cluster is known."""

    if cluster_id is None:
        return
    insert_research_event(
        connection,
        ResearchEvent(
            cluster_id=cluster_id,
            event_type=event_type,
            object_type=object_type,
            object_id=object_id,
            summary=summary,
            metadata=metadata or {},
        ),
    )


def _cluster_id_from_pending(entry: PendingEntry) -> int | None:
    value = entry.payload.get("cluster_id")
    return value if isinstance(value, int) and value > 0 else None


def _row_to_concept(row: sqlite3.Row) -> Concept:
    return Concept(
        concept_id=row["concept_id"],
        name=row["name"],
        concept_type=row["concept_type"],
        description=row["description"],
        aliases=_json_loads(row["aliases_json"], []),
        notes=row["notes"],
    )


def _row_to_research_event(row: sqlite3.Row) -> ResearchEvent:
    return ResearchEvent(
        event_id=row["event_id"],
        cluster_id=row["cluster_id"],
        event_type=row["event_type"],
        object_type=row["object_type"],
        object_id=row["object_id"],
        summary=row["summary"],
        metadata=_json_loads(row["metadata_json"], {}),
        created_at=row["created_at"],
    )


def _row_to_paper(row: sqlite3.Row) -> Paper:
    return Paper(
        id=row["id"],
        title=row["title"],
        authors=_json_loads(row["authors_json"], []),
        year=row["year"],
        venue=row["venue"],
        pdf_path=row["pdf_path"],
        url=row["url"],
        notes=row["notes"],
        cluster_id=row["cluster_id"],
    )


def _row_to_research_topic(row: sqlite3.Row) -> ResearchTopic:
    return ResearchTopic(
        id=row["id"],
        title=row["title"],
        raw_topic=row["raw_topic"],
        clarified_topic=row["clarified_topic"],
        clarification_json=_json_loads(row["clarification_json"], {}),
        created_at=row["created_at"],
    )


def _row_to_literature_note(row: sqlite3.Row) -> LiteratureNote:
    return LiteratureNote(
        id=row["id"],
        topic_id=row["topic_id"],
        paper_id=row["paper_id"],
        source_path=row["source_path"],
        note_type=row["note_type"],
        title=row["title"],
        content_json=_json_loads(row["content_json"], {}),
        markdown_note=row["markdown_note"],
        created_at=row["created_at"],
    )


def _row_to_literature_summary(row: sqlite3.Row) -> LiteratureSummary:
    return LiteratureSummary(
        id=row["id"],
        topic_id=row["topic_id"],
        note_id=row["note_id"],
        paper_id=row["paper_id"],
        summary_json=_json_loads(row["summary_json"], {}),
        markdown_summary=row["markdown_summary"],
        created_at=row["created_at"],
    )


def _row_to_theorem(connection: sqlite3.Connection, row: sqlite3.Row) -> Theorem:
    pk = _pk_column(connection, "theorems", "theorem_id")
    return Theorem(
        theorem_id=row[pk],
        title=row["title"],
        statement=row["statement"],
        theorem_type=row["theorem_type"],
        model_family=row["model_family"],
        objective_family=row["objective_family"],
        architecture_assumptions=_json_loads(row["architecture_assumptions_json"], []),
        information_assumptions=_json_loads(row["information_assumptions_json"], []),
        strategy_assumptions=_json_loads(row["strategy_assumptions_json"], []),
        process_bound=row["process_bound"],
        complexity_upper=row["complexity_upper"],
        complexity_lower=row["complexity_lower"],
        memory_upper=row["memory_upper"],
        memory_lower=row["memory_lower"],
        source_paper_id=row["source_paper_id"],
        source_location=row["source_location"],
        proof_technique=row["proof_technique"],
        confidence=row["confidence"],
        cluster_id=row["cluster_id"],
        notes=row["notes"],
        assumptions=_json_loads(row["assumptions_json"], []),
        conclusion=row["conclusion"],
        paper_id=row["paper_id"],
        tags=_json_loads(row["tags_json"], []),
    )


def _row_to_reduction(row: sqlite3.Row) -> Reduction:
    return Reduction(
        id=row["id"],
        title=row["title"],
        source_problem=row["source_problem"],
        target_problem=row["target_problem"],
        statement=row["statement"],
        assumptions=_json_loads(row["assumptions_json"], []),
        paper_id=row["paper_id"],
        source_paper_id=row["source_paper_id"],
        source_location=row["source_location"],
        proof_technique=row["proof_technique"],
        cluster_id=row["cluster_id"],
        tags=_json_loads(row["tags_json"], []),
        notes=row["notes"],
    )


def _row_to_open_problem(row: sqlite3.Row) -> OpenProblem:
    return OpenProblem(
        id=row["id"],
        title=row["title"],
        statement=row["statement"],
        context=row["context"],
        status=row["status"],
        paper_id=row["paper_id"],
        source_paper_id=row["source_paper_id"],
        source_location=row["source_location"],
        cluster_id=row["cluster_id"],
        tags=_json_loads(row["tags_json"], []),
        notes=row["notes"],
    )


def _row_to_pending_entry(row: sqlite3.Row) -> PendingEntry:
    return PendingEntry(
        id=row["id"],
        entry_type=row["entry_type"],
        payload=_json_loads(row["payload_json"], {}),
        source_text=row["source_text"],
        status=row["status"],
        duplicate_of=row["duplicate_of"],
        warnings=_json_loads(row["warnings_json"], []),
    )


def _row_to_derived_result(row: sqlite3.Row) -> DerivedResult:
    return DerivedResult(
        id=row["id"],
        title=row["title"],
        statement=row["statement"],
        dependencies=_json_loads(row["dependencies_json"], []),
        proof_sketch=row["proof_sketch"],
        status=row["status"],
        cluster_id=row["cluster_id"],
        notes=row["notes"],
    )


def _row_to_conjecture(connection: sqlite3.Connection, row: sqlite3.Row) -> Conjecture:
    pk = _pk_column(connection, "conjectures", "conjecture_id")
    return Conjecture(
        conjecture_id=row[pk],
        title=row["title"],
        statement=row["statement"],
        cluster_id=row["cluster_id"],
        motivation=row["motivation"],
        related_theorems=_json_loads(row["related_theorems_json"], []),
        expected_status=row["expected_status"],
        confidence=row["confidence"],
        attack_plan=row["attack_plan"],
        possible_counterexamples=_json_loads(row["possible_counterexamples_json"], []),
        status=row["status"],
        notes=row["notes"],
        rationale=row["rationale"],
    )


def _row_to_proof_attempt(row: sqlite3.Row) -> ProofAttempt:
    return ProofAttempt(
        id=row["id"],
        target_type=row["target_type"],
        target_id=row["target_id"],
        strategy=row["strategy"],
        notes=row["notes"],
        status=row["status"],
        cluster_id=row["cluster_id"],
    )


def _row_to_evidence_span(row: sqlite3.Row) -> EvidenceSpan:
    return EvidenceSpan(
        evidence_id=row["evidence_id"],
        paper_id=row["paper_id"],
        entry_type=row["entry_type"],
        entry_id=row["entry_id"],
        page_start=row["page_start"],
        page_end=row["page_end"],
        quote_or_summary=row["quote_or_summary"],
        confidence=row["confidence"],
        notes=row["notes"],
    )


def _row_to_code_artifact(row: sqlite3.Row) -> CodeArtifact:
    return CodeArtifact(
        artifact_id=row["artifact_id"],
        name=row["name"],
        path=row["path"],
        artifact_type=row["artifact_type"],
        entrypoint=row["entrypoint"],
        language=row["language"],
        description=row["description"],
        cluster_id=row["cluster_id"],
        related_concepts=_json_loads(row["related_concepts"], []),
        related_conjectures=_json_loads(row["related_conjectures"], []),
        tests_path=row["tests_path"],
        status=row["status"],
        git_commit_hash=row["git_commit_hash"],
        notes=row["notes"],
    )


def _row_to_experiment_run(row: sqlite3.Row) -> ExperimentRun:
    return ExperimentRun(
        run_id=row["run_id"],
        artifact_id=row["artifact_id"],
        cluster_id=row["cluster_id"],
        conjecture_id=row["conjecture_id"],
        experiment_type=row["experiment_type"],
        input_path=row["input_path"],
        output_path=row["output_path"],
        input_json=_json_loads(row["input_json"], {}),
        output_json=_json_loads(row["output_json"], {}),
        result_summary=row["result_summary"],
        command_run=row["command_run"],
        git_commit_hash=row["git_commit_hash"],
        notes=row["notes"],
    )


def _normalize_title(title: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in title).split())

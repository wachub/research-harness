from src import db
from src.curate import analyze_pending_entry
from src.extract import LLMClient, PlaceholderProvider, extract_from_text
from src.schemas import Concept, ConceptLink, ExperimentRun, PendingEntry, ResearchCluster, Theorem


def test_cluster_creation(tmp_path):
    db_path = tmp_path / "research.db"
    db.initialize_database(db_path)

    with db.get_connection(db_path) as connection:
        cluster_id = db.insert_cluster(
            connection,
            ResearchCluster(
                name="Partial-information parity games",
                description="Observation and parity synthesis frontier.",
                priority=9,
            ),
        )
        clusters = db.list_clusters(connection)

    assert cluster_id > 0
    assert any(cluster.name == "Partial-information parity games" for cluster in clusters)


def test_concept_creation_and_linking(tmp_path):
    db_path = tmp_path / "research.db"
    db.initialize_database(db_path)

    with db.get_connection(db_path) as connection:
        source = db.insert_concept(
            connection,
            Concept(name="Observation equivalence", concept_type="logic", aliases=["obs-eq"]),
        )
        target = db.insert_concept(
            connection,
            Concept(name="Partial-information game", concept_type="model", aliases=["imperfect-information game"]),
        )
        db.insert_concept_link(
            connection,
            ConceptLink(source_concept_id=source, target_concept_id=target, relation_type="uses"),
        )
        links = db.list_concept_links(connection)
        alias_match = db.find_concept_by_name_or_alias(connection, "obs-eq")

    assert links[0].source_concept_id == source
    assert alias_match is not None
    assert alias_match.concept_id == source


def test_theorem_schema_validation_with_broad_fields():
    theorem = Theorem(
        statement="Two-process reachability synthesis is decidable under local observation assumptions.",
        theorem_type="decidability",
        model_family="distributed synthesis",
        objective_family="reachability",
        architecture_assumptions=["two-process architecture"],
        information_assumptions=["local observation"],
        strategy_assumptions=["finite-state strategy"],
        process_bound="2",
        complexity_upper="EXPTIME",
        source_location="Theorem 3.1",
        proof_technique="fixed-point algorithm",
    )

    assert theorem.title is not None
    assert theorem.theorem_type == "decidability"
    assert theorem.model_family == "distributed synthesis"


def test_pending_extraction_dry_run(tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    db_path = tmp_path / "research.db"
    db.initialize_database(db_path)
    client = LLMClient(provider=PlaceholderProvider(model="placeholder", dry_run=True))

    ids = extract_from_text(
        "Theorem. ATS games with safety objective are decidable for two-process architectures.",
        db_path=db_path,
        client=client,
    )

    with db.get_connection(db_path) as connection:
        pending = db.get_pending_entry(connection, ids[0])
        theorems = db.list_theorems(connection)

    assert pending is not None
    assert pending.entry_type == "theorem"
    assert pending.status == "pending"
    assert "dry-run extraction" in pending.warnings
    assert theorems == []


def test_curator_detects_missing_assumptions(tmp_path):
    db_path = tmp_path / "research.db"
    db.initialize_database(db_path)

    with db.get_connection(db_path) as connection:
        entry = PendingEntry(
            entry_type="theorem",
            payload={
                "statement": "The problem is hard for safety objectives.",
                "theorem_type": "complexity_lower",
                "model_family": "games on graphs",
                "objective_family": "safety",
                "complexity_lower": None,
            },
        )
        warnings, duplicates = analyze_pending_entry(connection, entry)

    assert duplicates == []
    assert "missing assumptions" in warnings
    assert "missing source location" in warnings
    assert "vague complexity term without complexity class" in warnings


def test_experiment_run_storage(tmp_path):
    db_path = tmp_path / "research.db"
    db.initialize_database(db_path)

    with db.get_connection(db_path) as connection:
        run_id = db.insert_experiment_run(
            connection,
            ExperimentRun(
                cluster_id=None,
                experiment_type="ats_bounded_safety",
                input_json={"kind": "ATS"},
                output_json={"winning": True},
                result_summary="winning=True",
            ),
        )
        runs = db.list_experiment_runs(connection)

    assert run_id == 1
    assert runs[0].experiment_type == "ats_bounded_safety"
    assert runs[0].output_json == {"winning": True}

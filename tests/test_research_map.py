import pytest

from src import db
from src.schemas import Concept, ConceptLink, ExperimentRun, ResearchTask, Theorem


def test_task_creation(tmp_path):
    db_path = tmp_path / "research.db"
    db.initialize_database(db_path)

    with db.get_connection(db_path) as connection:
        assert db.list_tasks(connection) == []
        task_id = db.insert_task(
            connection,
            ResearchTask(
                name="Partial-information parity games",
                description="Observation and parity synthesis frontier.",
            ),
        )
        tasks = db.list_tasks(connection)

    assert task_id > 0
    assert any(task.name == "Partial-information parity games" for task in tasks)
    with pytest.raises(db.DatabaseError):
        with db.get_connection(db_path) as connection:
            db.insert_task(
                connection,
                ResearchTask(name="Partial-information parity games", description="Changed objective."),
            )
    with db.get_connection(db_path) as connection:
        assert db.get_task(connection, task_id).description == "Observation and parity synthesis frontier."


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


def test_experiment_run_storage(tmp_path):
    db_path = tmp_path / "research.db"
    db.initialize_database(db_path)

    with db.get_connection(db_path) as connection:
        run_id = db.insert_experiment_run(
            connection,
            ExperimentRun(
                task_id=None,
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

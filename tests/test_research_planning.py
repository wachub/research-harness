import json

from src import db
from src.cli import main
from src.llm import LLMClient, LLMRequest, LLMResponse
from src.research_planning import assess_literature_need, plan_research, save_plan_as_pending
from src.schemas import Paper, ResearchUnit


class FakePlanningProvider:
    provider_name = "fake"
    model = "fake-model"

    def __init__(self, plan: dict):
        self.plan = plan

    def complete(self, request: LLMRequest) -> LLMResponse:
        assert request.json_mode
        return LLMResponse(
            content=json.dumps(self.plan),
            provider=self.provider_name,
            model=self.model,
            usage={"total_tokens": 42},
        )


def _create_task(db_path) -> None:
    with db.get_connection(db_path) as connection:
        assert db.insert_task(connection, db.ResearchTask(name="ATS safety", description="Investigate ATS safety.")) == 1


def _plan_payload(reference_id: int = 1) -> dict:
    return {
        "interpreted_goal": "Assess finite-memory strategies for the requested ATS safety fragment.",
        "relevant_existing_state": [
            {
                "kind": "research_task",
                "object_id": reference_id,
                "relevance": "This is the existing ATS/CDM/2DM research task.",
            }
        ],
        "recommended_task_id": 1,
        "task_rationale": "The goal concerns the seeded restricted multi-decision-maker synthesis task.",
        "proposed_subquestions": [
            {
                "question": "Which three-process ATS information assumptions permit finite-state strategies?",
                "rationale": "The current goal leaves the information model underspecified.",
                "dependencies_or_assumptions": ["define the observation model"],
                "uncertainty_note": "This is a proposed question, not an established open problem.",
            }
        ],
        "proposed_conjectures": [
            {
                "statement": "Under a specified causal-order restriction, finite-memory strategies may suffice for bounded three-process ATS safety instances.",
                "rationale": "It is a deliberately restricted candidate for review.",
                "dependencies_or_assumptions": ["causal-order restriction", "bounded safety instances"],
                "uncertainty_note": "Unverified candidate conjecture.",
            }
        ],
        "proposed_literature_tasks": [
            {
                "task": "Verify the exact ATS assumptions in the stored undecidability evidence.",
                "rationale": "The plan must distinguish nearby models.",
                "dependencies_or_assumptions": ["read the cited source"],
                "uncertainty_note": "The stored summary may omit relevant qualifications.",
            }
        ],
        "proposed_experiments": [
            {
                "bounded_design": "After review, enumerate seeded three-process safety games under the stated restriction.",
                "rationale": "This could search for bounded counterexamples only.",
                "dependencies_or_assumptions": ["human-approved restriction", "existing bounded checker"],
                "uncertainty_note": "No bounded outcome would establish the conjecture.",
            }
        ],
        "uncertainty_note": "The plan contains proposals only and establishes no theorem or experiment result.",
    }


def test_plan_research_uses_existing_state_and_does_not_persist_by_default(tmp_path):
    db_path = tmp_path / "research.db"
    db.initialize_database(db_path)
    _create_task(db_path)
    client = LLMClient(provider=FakePlanningProvider(_plan_payload()))

    result = plan_research(
        "Investigate finite memory for three-process ATS safety.",
        task_id=1,
        use_llm=True,
        db_path=db_path,
        client=client,
    )

    with db.get_connection(db_path) as connection:
        pending = db.list_pending_entries(connection)
        conjectures = db.list_conjectures(connection)
        open_problems = db.list_open_problems(connection)

    assert result.available
    assert result.plan is not None
    assert result.plan.recommended_task_id == 1
    assert result.provider_metadata["usage"] == {"total_tokens": 42}
    assert pending == []
    assert conjectures == []
    assert open_problems == []


def test_plan_research_rejects_unknown_state_references(tmp_path):
    db_path = tmp_path / "research.db"
    db.initialize_database(db_path)
    _create_task(db_path)
    client = LLMClient(provider=FakePlanningProvider(_plan_payload(reference_id=999)))

    result = plan_research(
        "Investigate finite memory for three-process ATS safety.",
        use_llm=True,
        db_path=db_path,
        client=client,
    )

    assert not result.available
    assert "unknown research_task id 999" in result.message


def test_save_plan_as_pending_keeps_proposals_out_of_durable_tables(tmp_path):
    db_path = tmp_path / "research.db"
    db.initialize_database(db_path)
    _create_task(db_path)
    result = plan_research(
        "Investigate finite memory for three-process ATS safety.",
        task_id=1,
        use_llm=True,
        db_path=db_path,
        client=LLMClient(provider=FakePlanningProvider(_plan_payload())),
    )

    pending_ids = save_plan_as_pending(result, "Investigate finite memory for three-process ATS safety.", db_path)

    with db.get_connection(db_path) as connection:
        pending = db.list_pending_entries(connection)
        conjectures = db.list_conjectures(connection)
        open_problems = db.list_open_problems(connection)

    assert len(pending_ids) == 2
    assert {entry.entry_type for entry in pending} == {"conjecture_seed", "open_problem"}
    assert all(entry.status == "pending" for entry in pending)
    assert all(any("requires human review" in warning for warning in entry.warnings) for entry in pending)
    assert conjectures == []
    assert open_problems == []


def test_plan_research_without_llm_is_a_clean_no_write_response(tmp_path, capsys):
    db_path = tmp_path / "research.db"

    exit_code = main(
        [
            "--db",
            str(db_path),
            "plan-research",
            "--goal",
            "Investigate finite memory for three-process ATS safety.",
        ]
    )
    output = capsys.readouterr().out

    with db.get_connection(db_path) as connection:
        pending = db.list_pending_entries(connection)

    assert exit_code == 2
    assert "requires --llm" in output
    assert pending == []


def test_literature_need_assessment_uses_stored_ids_without_writing(tmp_path):
    db_path = tmp_path / "research.db"
    db.initialize_database(db_path)
    _create_task(db_path)
    with db.get_connection(db_path) as connection:
        unit_id = db.insert_research_unit(
            connection,
            ResearchUnit(task_id=1, kind="hypothesis", title="Finite memory", purpose="Check global safety"),
        )
        paper_id = db.insert_paper(
            connection, Paper(title="Stored source", authors=["A"], year=2025, task_id=1),
        )
    payload = {
        "current_coverage": "A stored source addresses nearby safety games.",
        "further_survey_needed": True,
        "rationale": "The selected unit's precise assumptions are not covered.",
        "relevant_existing_state": [{"kind": "paper", "object_id": paper_id}],
        "suggested_focus": "Check the three-process boundary.",
        "uncertainty_note": "Only stored records were considered.",
    }
    client = LLMClient(provider=FakePlanningProvider(payload))
    result = assess_literature_need(1, unit_id, db_path=db_path, client=client)
    assert result.available
    assert result.assessment.further_survey_needed
    assert result.assessment.relevant_existing_state[0].object_id == paper_id
    with db.get_connection(db_path) as connection:
        assert len(db.list_research_units(connection, task_id=1)) == 1
        assert db.list_pending_entries(connection) == []
        assert db.list_derived_results(connection, task_id=1) == []


def test_literature_need_assessment_rejects_invented_reference(tmp_path):
    db_path = tmp_path / "research.db"
    db.initialize_database(db_path)
    _create_task(db_path)
    with db.get_connection(db_path) as connection:
        unit_id = db.insert_research_unit(
            connection,
            ResearchUnit(task_id=1, kind="literature_review", title="Survey", purpose="Check existing work"),
        )
    payload = {
        "current_coverage": "Unclear.",
        "further_survey_needed": True,
        "rationale": "A source appears to be missing.",
        "relevant_existing_state": [{"kind": "paper", "object_id": 999999}],
        "suggested_focus": None,
        "uncertainty_note": "Stored state is sparse.",
    }
    result = assess_literature_need(
        1, unit_id, db_path=db_path, client=LLMClient(provider=FakePlanningProvider(payload)),
    )
    assert not result.available
    assert "unknown or out-of-scope paper id" in result.message

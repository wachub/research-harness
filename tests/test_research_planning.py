import json

from src import db
from src.cli import main
from src.llm import LLMClient, LLMRequest, LLMResponse
from src.research_planning import plan_research, save_plan_as_pending


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


def _plan_payload(reference_id: int = 1) -> dict:
    return {
        "interpreted_goal": "Assess finite-memory strategies for the requested ATS safety fragment.",
        "relevant_existing_state": [
            {
                "kind": "research_cluster",
                "object_id": reference_id,
                "relevance": "This is the existing ATS/CDM/2DM research cluster.",
            }
        ],
        "recommended_cluster_id": 1,
        "cluster_rationale": "The goal concerns the seeded restricted multi-decision-maker synthesis cluster.",
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
    client = LLMClient(provider=FakePlanningProvider(_plan_payload()))

    result = plan_research(
        "Investigate finite memory for three-process ATS safety.",
        cluster_id=1,
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
    assert result.plan.recommended_cluster_id == 1
    assert result.provider_metadata["usage"] == {"total_tokens": 42}
    assert pending == []
    assert conjectures == []
    assert open_problems == []


def test_plan_research_rejects_unknown_state_references(tmp_path):
    db_path = tmp_path / "research.db"
    db.initialize_database(db_path)
    client = LLMClient(provider=FakePlanningProvider(_plan_payload(reference_id=999)))

    result = plan_research(
        "Investigate finite memory for three-process ATS safety.",
        use_llm=True,
        db_path=db_path,
        client=client,
    )

    assert not result.available
    assert "unknown research_cluster id 999" in result.message


def test_save_plan_as_pending_keeps_proposals_out_of_durable_tables(tmp_path):
    db_path = tmp_path / "research.db"
    db.initialize_database(db_path)
    result = plan_research(
        "Investigate finite memory for three-process ATS safety.",
        cluster_id=1,
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

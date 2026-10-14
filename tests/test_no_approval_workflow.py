"""The supported research workflow stores provisional findings without review gates."""

import pytest

from src import db
from src.autonomous_research import AutonomousResearch
from src.cli import build_parser
from src.llm import LLMClient
from src.schemas import ResearchTask
from tests.test_autonomous_research import SequenceProvider, _proposal


@pytest.mark.parametrize(
    "command",
    ("extract-from-text", "extract-from-pdf", "plan-research", "run-pipeline"),
)
def test_retired_pending_workflows_are_not_cli_commands(command):
    with pytest.raises(SystemExit):
        build_parser().parse_args([command])


def test_provisional_findings_go_straight_to_main_tables(tmp_path):
    path = tmp_path / "research.db"
    with db.get_connection(path) as connection:
        db.create_tables(connection)
        task_id = db.insert_task(connection, ResearchTask(name="Investigate safety"))

    provider = SequenceProvider([
        _proposal("conjecture", "Memory conjecture", "Finite memory may suffice."),
        _proposal("open_problem", "Boundary question", "Does it extend to five processes?"),
    ])
    result = AutonomousResearch(LLMClient(provider=provider), path).run(task_id, steps=2)

    assert result.status == "completed"
    with db.get_connection(path) as connection:
        conjectures = db.list_conjectures(connection, task_id=task_id)
        problems = db.list_open_problems(connection, task_id=task_id)
        pending = db.list_pending_entries(connection)
    assert len(conjectures) == len(problems) == 1
    assert conjectures[0].confidence == "needs_review"  # Unverified, not approval-gated.
    assert pending == []

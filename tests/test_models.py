import pytest

from src.generate_examples import generate_tiny_game
from src.models import SafetyGame
from src.schemas import Paper, Theorem


def test_schema_validation_rejects_blank_authors():
    with pytest.raises(ValueError):
        Paper(title="Bad", authors=["   "], year=2026)


def test_theorem_schema_accepts_minimal_valid_record():
    theorem = Theorem(title="Finite safety", statement="Finite games admit bounded exploration.")

    assert theorem.assumptions == []
    assert theorem.tags == []


def test_tiny_game_generation_round_trips():
    game = generate_tiny_game(kind="CDM", process_count=3, states_per_process=2, seed=12)
    loaded = SafetyGame.from_dict(game.to_dict())

    assert loaded.kind == "CDM"
    assert len(loaded.processes) == 3
    assert loaded.initial_state == ("s0", "s0", "s0")
    assert loaded.actions
    assert loaded.objective.is_safe(loaded.initial_state)


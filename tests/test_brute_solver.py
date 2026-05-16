from src.brute_solver import check_strategy_counterexample, find_memoryless_safety_strategy
from src.models import Action, Process, SafetyGame, SafetyObjective


def test_brute_solver_finds_safe_strategy_for_local_toggle_game():
    game = SafetyGame(
        name="safe_toggle",
        kind="ATS",
        processes=(
            Process(name="p0", local_states=("safe", "bad"), initial_state="safe"),
        ),
        actions=(
            Action(
                name="stay",
                participants=("p0",),
                transitions={("safe",): ("safe",), ("bad",): ("bad",)},
                controllable=True,
            ),
            Action(
                name="to_bad",
                participants=("p0",),
                transitions={("safe",): ("bad",)},
                controllable=True,
            ),
        ),
        objective=SafetyObjective(unsafe_states=frozenset({("bad",)})),
    )

    result = find_memoryless_safety_strategy(game, depth=3)

    assert result.winning is True
    assert result.strategy is not None
    assert result.strategy["p0"]["safe"] == "stay"


def test_strategy_counterexample_sees_environment_move():
    game = SafetyGame(
        name="env_bad",
        kind="ATS",
        processes=(
            Process(name="p0", local_states=("safe", "bad"), initial_state="safe"),
        ),
        actions=(
            Action(
                name="env_to_bad",
                participants=("p0",),
                transitions={("safe",): ("bad",)},
                controllable=False,
            ),
        ),
        objective=SafetyObjective(unsafe_states=frozenset({("bad",)})),
    )

    path = check_strategy_counterexample(game, {"p0": {"safe": None, "bad": None}}, depth=1)

    assert path == [("safe",), ("bad",)]


"""Brute-force safety checker for very small distributed games."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from collections.abc import Iterator

from .models import Action, GlobalState, SafetyGame


LocalPolicy = dict[str, str | None]
Strategy = dict[str, LocalPolicy]


@dataclass(frozen=True)
class BruteForceResult:
    """Result of brute-force strategy enumeration."""

    winning: bool
    checked_strategies: int
    depth: int
    strategy: Strategy | None = None
    counterexample: list[GlobalState] | None = None


def find_memoryless_safety_strategy(game: SafetyGame, depth: int) -> BruteForceResult:
    """Find a memoryless distributed strategy that keeps all paths safe."""

    if depth < 0:
        raise ValueError("depth must be non-negative")
    checked = 0
    first_counterexample: list[GlobalState] | None = None

    for strategy in enumerate_memoryless_strategies(game):
        checked += 1
        counterexample = check_strategy_counterexample(game, strategy, depth)
        if counterexample is None:
            return BruteForceResult(True, checked, depth, strategy=strategy)
        if first_counterexample is None:
            first_counterexample = counterexample

    return BruteForceResult(False, checked, depth, counterexample=first_counterexample)


def enumerate_memoryless_strategies(game: SafetyGame) -> Iterator[Strategy]:
    """Yield local-state policies for controllable actions."""

    process_strategy_choices: list[list[LocalPolicy]] = []
    controllable_actions = [action for action in game.actions if action.controllable]

    for process in game.processes:
        per_state_choices: list[list[str | None]] = []
        for local_state in process.local_states:
            choices: list[str | None] = [
                action.name
                for action in controllable_actions
                if process.name in action.participants
            ]
            choices.append(None)
            per_state_choices.append(choices)

        policies = []
        for selected in product(*per_state_choices):
            policies.append(dict(zip(process.local_states, selected)))
        process_strategy_choices.append(policies)

    for selected_policies in product(*process_strategy_choices):
        yield {
            process.name: dict(policy)
            for process, policy in zip(game.processes, selected_policies)
        }


def check_strategy_counterexample(
    game: SafetyGame,
    strategy: Strategy,
    depth: int,
) -> list[GlobalState] | None:
    """Return a violating path if the strategy loses within the depth bound."""

    initial = game.initial_state
    if not game.objective.is_safe(initial):
        return [initial]

    frontier: list[tuple[GlobalState, list[GlobalState]]] = [(initial, [initial])]
    for _ in range(depth):
        next_frontier: list[tuple[GlobalState, list[GlobalState]]] = []
        for state, path in frontier:
            successors = _scheduled_successors(game, strategy, state)
            if not successors:
                successors = [state]
            for successor in successors:
                successor_path = path + [successor]
                if not game.objective.is_safe(successor):
                    return successor_path
                next_frontier.append((successor, successor_path))
        frontier = _dedupe_frontier(next_frontier)
    return None


def _scheduled_successors(game: SafetyGame, strategy: Strategy, state: GlobalState) -> list[GlobalState]:
    enabled_environment = game.enabled_actions(state, controllable=False)
    enabled_controllable = [
        action
        for action in game.enabled_actions(state, controllable=True)
        if _strategy_enables_action(game, strategy, state, action)
    ]
    scheduled_actions = enabled_environment + enabled_controllable
    return [game.apply_action(state, action) for action in scheduled_actions]


def _strategy_enables_action(
    game: SafetyGame,
    strategy: Strategy,
    state: GlobalState,
    action: Action,
) -> bool:
    process_index = {name: index for index, name in enumerate(game.process_names)}
    for process_name in action.participants:
        local_state = state[process_index[process_name]]
        if strategy.get(process_name, {}).get(local_state) != action.name:
            return False
    return True


def _dedupe_frontier(frontier: list[tuple[GlobalState, list[GlobalState]]]) -> list[tuple[GlobalState, list[GlobalState]]]:
    seen: set[GlobalState] = set()
    result: list[tuple[GlobalState, list[GlobalState]]] = []
    for state, path in frontier:
        if state not in seen:
            seen.add(state)
            result.append((state, path))
    return result

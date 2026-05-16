"""Tiny random ATS/CDM/2DM safety-game generator."""

from __future__ import annotations

import random

from .models import Action, Process, SafetyGame, SafetyObjective


SUPPORTED_KINDS = {"ATS", "CDM", "2DM"}


def generate_tiny_game(
    kind: str = "ATS",
    process_count: int = 2,
    states_per_process: int = 2,
    seed: int | None = None,
) -> SafetyGame:
    """Generate a tiny random safety game suitable for brute-force checks."""

    normalized_kind = kind.upper()
    if normalized_kind not in SUPPORTED_KINDS:
        raise ValueError(f"kind must be one of {sorted(SUPPORTED_KINDS)}")
    if not 2 <= process_count <= 4:
        raise ValueError("process_count must be between 2 and 4")
    if not 2 <= states_per_process <= 3:
        raise ValueError("states_per_process must be between 2 and 3")

    rng = random.Random(seed)
    processes = tuple(
        Process(
            name=f"p{i}",
            local_states=tuple(f"s{j}" for j in range(states_per_process)),
            initial_state="s0",
        )
        for i in range(process_count)
    )
    actions: list[Action] = []

    for process in processes:
        transitions = {}
        for state in process.local_states:
            target = rng.choice(process.local_states)
            transitions[(state,)] = (target,)
        actions.append(
            Action(
                name=f"ctrl_{process.name}",
                participants=(process.name,),
                transitions=transitions,
                controllable=True,
            )
        )

    for index, participants in enumerate(_environment_participants(normalized_kind, processes, rng)):
        transitions = {}
        local_domains = [next(p for p in processes if p.name == name).local_states for name in participants]
        for source in _cartesian(local_domains):
            target = tuple(rng.choice(domain) for domain in local_domains)
            transitions[source] = target
        actions.append(
            Action(
                name=f"env_{index}",
                participants=participants,
                transitions=transitions,
                controllable=False,
            )
        )

    unsafe_state = tuple(rng.choice(process.local_states) for process in processes)
    initial_state = tuple(process.initial_state for process in processes)
    if unsafe_state == initial_state:
        unsafe_state = tuple(
            process.local_states[-1] if process.local_states[-1] != process.initial_state else process.local_states[0]
            for process in processes
        )

    return SafetyGame(
        name=f"tiny_{normalized_kind.lower()}_{process_count}p",
        kind=normalized_kind,
        processes=processes,
        actions=tuple(actions),
        objective=SafetyObjective(unsafe_states=frozenset({unsafe_state})),
    )


def _environment_participants(
    kind: str,
    processes: tuple[Process, ...],
    rng: random.Random,
) -> list[tuple[str, ...]]:
    names = [process.name for process in processes]
    if kind == "CDM":
        return [(names[i], names[(i + 1) % len(names)]) for i in range(len(names) - 1)]
    if kind == "2DM":
        return [tuple(names[:2]), tuple(names[-2:])]
    count = min(2, len(names))
    result = []
    for _ in range(count):
        size = rng.choice([1, 2])
        result.append(tuple(sorted(rng.sample(names, size))))
    return result


def _cartesian(domains: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    if not domains:
        return [()]
    result = [()]
    for domain in domains:
        result = [prefix + (value,) for prefix in result for value in domain]
    return result


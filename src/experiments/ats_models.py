"""Finite ATS/CDM/2DM-like safety-game structures for toy experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Iterable


GlobalState = tuple[str, ...]


@dataclass(frozen=True)
class Process:
    """A finite process with local states and one initial local state."""

    name: str
    local_states: tuple[str, ...]
    initial_state: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("process name is required")
        if not self.local_states:
            raise ValueError("process must have at least one local state")
        if self.initial_state not in self.local_states:
            raise ValueError("initial_state must be in local_states")


@dataclass(frozen=True)
class Action:
    """An action over one or more participating processes."""

    name: str
    participants: tuple[str, ...]
    transitions: dict[GlobalState, GlobalState]
    controllable: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("action name is required")
        if not self.participants:
            raise ValueError("action must have at least one participant")
        for source, target in self.transitions.items():
            if len(source) != len(self.participants) or len(target) != len(self.participants):
                raise ValueError("transition arity must match participants")


@dataclass(frozen=True)
class SafetyObjective:
    """Safety objective as either a safe-state set or unsafe-state set."""

    safe_states: frozenset[GlobalState] | None = None
    unsafe_states: frozenset[GlobalState] = field(default_factory=frozenset)

    def is_safe(self, state: GlobalState) -> bool:
        if self.safe_states is not None:
            return state in self.safe_states
        return state not in self.unsafe_states


@dataclass(frozen=True)
class SafetyGame:
    """A tiny finite safety game with asynchronous shared actions."""

    name: str
    kind: str
    processes: tuple[Process, ...]
    actions: tuple[Action, ...]
    objective: SafetyObjective

    def __post_init__(self) -> None:
        if not self.processes:
            raise ValueError("game must have processes")
        names = [process.name for process in self.processes]
        if len(names) != len(set(names)):
            raise ValueError("process names must be unique")
        process_set = set(names)
        for action in self.actions:
            unknown = set(action.participants) - process_set
            if unknown:
                raise ValueError(f"action {action.name} uses unknown processes: {sorted(unknown)}")

    @property
    def process_names(self) -> tuple[str, ...]:
        return tuple(process.name for process in self.processes)

    @property
    def initial_state(self) -> GlobalState:
        return tuple(process.initial_state for process in self.processes)

    def all_global_states(self) -> list[GlobalState]:
        """Enumerate all global states."""

        return list(product(*(process.local_states for process in self.processes)))

    def enabled_actions(self, state: GlobalState, controllable: bool | None = None) -> list[Action]:
        """Return actions enabled at a global state."""

        enabled = []
        for action in self.actions:
            if controllable is not None and action.controllable != controllable:
                continue
            projection = self.project(state, action.participants)
            if projection in action.transitions:
                enabled.append(action)
        return enabled

    def apply_action(self, state: GlobalState, action: Action) -> GlobalState:
        """Apply an enabled action to a global state."""

        projection = self.project(state, action.participants)
        if projection not in action.transitions:
            raise ValueError(f"action {action.name} is not enabled at {state}")
        next_projection = action.transitions[projection]
        values = list(state)
        index_by_process = {name: index for index, name in enumerate(self.process_names)}
        for offset, process_name in enumerate(action.participants):
            values[index_by_process[process_name]] = next_projection[offset]
        return tuple(values)

    def project(self, state: GlobalState, participants: Iterable[str]) -> GlobalState:
        index_by_process = {name: index for index, name in enumerate(self.process_names)}
        return tuple(state[index_by_process[name]] for name in participants)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the game to a JSON-compatible dictionary."""

        return {
            "name": self.name,
            "kind": self.kind,
            "processes": [
                {
                    "name": process.name,
                    "local_states": list(process.local_states),
                    "initial_state": process.initial_state,
                }
                for process in self.processes
            ],
            "actions": [
                {
                    "name": action.name,
                    "participants": list(action.participants),
                    "controllable": action.controllable,
                    "transitions": [
                        {"source": list(source), "target": list(target)}
                        for source, target in action.transitions.items()
                    ],
                }
                for action in self.actions
            ],
            "objective": {
                "safe_states": [list(state) for state in self.objective.safe_states]
                if self.objective.safe_states is not None
                else None,
                "unsafe_states": [list(state) for state in self.objective.unsafe_states],
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SafetyGame":
        """Load a game from a JSON-compatible dictionary."""

        processes = tuple(
            Process(
                name=item["name"],
                local_states=tuple(item["local_states"]),
                initial_state=item["initial_state"],
            )
            for item in data["processes"]
        )
        actions = tuple(
            Action(
                name=item["name"],
                participants=tuple(item["participants"]),
                controllable=bool(item.get("controllable", True)),
                transitions={
                    tuple(transition["source"]): tuple(transition["target"])
                    for transition in item["transitions"]
                },
            )
            for item in data["actions"]
        )
        objective_data = data["objective"]
        safe_states_raw = objective_data.get("safe_states")
        safe_states = (
            frozenset(tuple(state) for state in safe_states_raw)
            if safe_states_raw is not None
            else None
        )
        unsafe_states = frozenset(tuple(state) for state in objective_data.get("unsafe_states", []))
        return cls(
            name=data["name"],
            kind=data["kind"],
            processes=processes,
            actions=actions,
            objective=SafetyObjective(safe_states=safe_states, unsafe_states=unsafe_states),
        )


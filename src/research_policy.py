"""Central authority decisions for the bounded research controller."""

from __future__ import annotations

from enum import Enum


class ControllerMode(str, Enum):
    """How the controller obtains authority for otherwise safe actions."""

    INTERACTIVE = "interactive"
    AUTONOMOUS = "autonomous"


class AuthorityDecision(str, Enum):
    """The only authority outcomes available to controller actions."""

    AUTO = "AUTO"
    ASK = "ASK"
    BLOCK = "BLOCK"


READ_ONLY_ACTIONS = frozenset(
    {
        "inspect_state",
        "review_stored_evidence",
        "inspect_experiment_result",
        "stop",
    }
)

PROVISIONAL_ACTIONS = frozenset(
    {
        "propose_subquestions",
        "create_pending_conjecture",
        "create_pending_open_problem",
        "design_bounded_experiment",
        "run_trusted_experiment",
        "reassess_plan",
    }
)

# These names are intentionally recognized by policy even though the controller
# only has a typed model for ``approve_pending``.  Unknown LLM action names are
# rejected at schema validation before policy is consulted.
FORBIDDEN_ACTIONS = frozenset(
    {
        "approve_pending",
        "approve_theorem",
        "mark_conjecture_proved",
        "delete_trusted_state",
        "arbitrary_shell_command",
        "run_unregistered_code",
        "unbounded_experiment",
    }
)


def authority_for(action_type: str, mode: ControllerMode) -> AuthorityDecision:
    """Return the explicit centralized decision for one action name."""

    if action_type in READ_ONLY_ACTIONS:
        return AuthorityDecision.AUTO
    if action_type in PROVISIONAL_ACTIONS:
        return (
            AuthorityDecision.ASK
            if mode is ControllerMode.INTERACTIVE
            else AuthorityDecision.AUTO
        )
    return AuthorityDecision.BLOCK

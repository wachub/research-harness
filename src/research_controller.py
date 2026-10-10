"""Bounded, auditable operational controller for existing research workflows."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from pydantic import Field

from . import db
from .curate import find_duplicates
from .llm import LLMClient, LLMError, LLMMessage, LLMRequest
from .research_actions import (
    ApprovePendingAction,
    BoundedExperimentParameters,
    ControllerAction,
    ControllerDecision,
    CreatePendingConjectureAction,
    CreatePendingOpenProblemAction,
    DesignBoundedExperimentAction,
    InspectExperimentResultAction,
    InspectStateAction,
    ProposeSubquestionsAction,
    RequestLiteratureReviewAction,
    ResearchReference,
    ReassessPlanAction,
    ReviewStoredEvidenceAction,
    RunTrustedExperimentAction,
    StopAction,
    get_trusted_artifact,
    run_trusted_bounded_experiment,
)
from .research_context import ControllerContext, load_controller_context, validate_context_reference
from .research_policy import AuthorityDecision, ControllerMode, authority_for
from .schemas import Conjecture, OpenProblem, PendingEntry, ResearchUnit, ResearchUnitLink, StrictBase


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIRECTORY = PROJECT_ROOT / "results" / "controller"
MAX_CONSECUTIVE_INVALID_OUTPUTS = 2
MAX_CONSECUTIVE_FAILURES = 2
MAX_CONSECUTIVE_PROVIDER_FAILURES = 2


CONTROLLER_INSTRUCTIONS = """
You are a bounded research controller. Select exactly one next action from the
typed vocabulary supplied below. Use only object IDs present in the compact
state. Do not create or request arbitrary commands, source code, network calls,
approval of pending entries, theorem approval, proof validation, conjecture
promotion, deletion, or unbounded experiments.

Actions are proposals or bounded operations only. Experimental output is not a
mathematical proof. Return one JSON object with exactly this outer shape:
{"action": {"action_type": "...", "reason": "...", "expected_effect": "...",
"references": [], "parameters": {...}}}

Available action types are: inspect_state, review_stored_evidence,
propose_subquestions, request_literature_review, create_pending_conjecture,
create_pending_open_problem, design_bounded_experiment, run_trusted_experiment,
inspect_experiment_result, reassess_plan, stop. Focus the selected research
unit when supplied. Subquestions create proposed follow-up units. A literature
review request creates a proposed literature unit; it does not fetch sources.
The only experiment handler is a tiny in-process
ATS/CDM/2DM bounded-safety check, with process_count 2..3,
states_per_process fixed at 2, and depth 0..5. An artifact must be supplied
and must be listed as tested in the current state.
""".strip()


ApprovalCallback = Callable[[ControllerAction], str]


class UnitChoice(StrictBase):
    """One validated choice from the controller's supplied open units."""

    unit_id: int = Field(gt=0)
    rationale: str = Field(min_length=1, max_length=800)


@dataclass(frozen=True)
class ControllerStep:
    step: int
    timestamp: str
    policy_decision: str
    action: dict[str, Any] | None
    result: dict[str, Any]
    success: bool


@dataclass(frozen=True)
class ControllerRunResult:
    status: str
    message: str
    goal: str
    task_id: int
    mode: str
    steps: tuple[ControllerStep, ...]
    log_path: Path | None

    @property
    def steps_completed(self) -> int:
        return len(self.steps)


class JsonlControllerLog:
    """Minimal append-only provenance log, deliberately outside trusted tables."""

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            path = DEFAULT_LOG_DIRECTORY / f"controller_{timestamp}.jsonl"
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


class ResearchController:
    """One controller implementation governed by an explicit authority policy."""

    def __init__(
        self,
        client: LLMClient,
        *,
        db_path: str | Path | None = None,
        approval_callback: ApprovalCallback | None = None,
    ) -> None:
        self.client = client
        self.db_path = db_path
        self.approval_callback = approval_callback
        self._selected_unit_id: int | None = None

    @property
    def selected_unit_id(self) -> int | None:
        return self._selected_unit_id

    def run(
        self,
        goal: str,
        task_id: int,
        *,
        mode: ControllerMode,
        unit_id: int | None = None,
        max_steps: int = 10,
        pause_every: int | None = None,
        log_path: str | Path | None = None,
    ) -> ControllerRunResult:
        """Run at most ``max_steps`` validated controller iterations."""

        if not goal.strip():
            raise ValueError("goal must not be blank")
        if max_steps < 1 or max_steps > 100:
            raise ValueError("max_steps must be between 1 and 100")
        if pause_every is not None and pause_every < 1:
            raise ValueError("pause_every must be at least 1")
        # Validate the selected task before any provider call or log write.
        self._selected_unit_id = None
        load_controller_context(task_id, self.db_path, unit_id=unit_id)
        if not self.client.available:
            return ControllerRunResult(
                status="unavailable",
                message="Research controller requires --llm and a configured remote LLM provider; nothing was written.",
                goal=goal,
                task_id=task_id,
                mode=mode.value,
                steps=(),
                log_path=None,
            )
        if unit_id is None:
            unit_id = self._choose_unit(goal, task_id)
        self._selected_unit_id = unit_id

        log = JsonlControllerLog(log_path)
        steps: list[ControllerStep] = []
        recent_steps: list[dict[str, Any]] = []
        fingerprints: set[str] = set()
        invalid_outputs = 0
        failures = 0
        provider_failures = 0

        try:
            for step_number in range(1, max_steps + 1):
                context = load_controller_context(task_id, self.db_path, unit_id=unit_id)
                self._selected_unit_id = context.active_unit.unit_id if context.active_unit else None
                try:
                    action = self._next_action(goal, context, recent_steps)
                except LLMError as exc:
                    message = str(exc)
                    is_invalid = "JSON" in message or "validation" in message
                    invalid_outputs = invalid_outputs + 1 if is_invalid else 0
                    provider_failures = provider_failures + 1 if not is_invalid else 0
                    step = self._record_step(
                        log,
                        steps,
                        goal,
                        task_id,
                        mode,
                        step_number,
                        "INVALID",
                        None,
                        {"status": "invalid_llm_output" if is_invalid else "provider_failure", "message": message},
                        False,
                    )
                    recent_steps.append(_recent_summary(step))
                    if invalid_outputs >= MAX_CONSECUTIVE_INVALID_OUTPUTS:
                        return self._finish("invalid_output", "Controller stopped after repeated invalid LLM output.", goal, task_id, mode, steps, log)
                    if provider_failures >= MAX_CONSECUTIVE_PROVIDER_FAILURES:
                        return self._finish("provider_failure", "Controller stopped after repeated provider failures.", goal, task_id, mode, steps, log)
                    continue

                try:
                    self._validate_action(action, context)
                except ValueError as exc:
                    invalid_outputs += 1
                    step = self._record_step(
                        log, steps, goal, task_id, mode, step_number, "INVALID", action,
                        {"status": "invalid_action", "message": str(exc)}, False,
                    )
                    recent_steps.append(_recent_summary(step))
                    if invalid_outputs >= MAX_CONSECUTIVE_INVALID_OUTPUTS:
                        return self._finish("invalid_output", "Controller stopped after repeated invalid actions.", goal, task_id, mode, steps, log)
                    continue

                invalid_outputs = 0
                provider_failures = 0
                fingerprint = _action_fingerprint(action)
                decision = authority_for(action.action_type, mode)
                if fingerprint in fingerprints:
                    step = self._record_step(
                        log, steps, goal, task_id, mode, step_number, decision.value, action,
                        {"status": "repeated_action", "message": "Controller proposed an unchanged action twice."}, False,
                    )
                    recent_steps.append(_recent_summary(step))
                    return self._finish("repeated_action", "Controller stopped after a repeated unchanged action.", goal, task_id, mode, steps, log)
                fingerprints.add(fingerprint)

                if isinstance(action, StopAction):
                    step = self._record_step(
                        log, steps, goal, task_id, mode, step_number, decision.value, action,
                        {"status": "stopped", "message": action.reason}, True,
                    )
                    recent_steps.append(_recent_summary(step))
                    return self._finish("stopped", "Controller selected stop.", goal, task_id, mode, steps, log)

                if decision is AuthorityDecision.BLOCK:
                    step = self._record_step(
                        log, steps, goal, task_id, mode, step_number, decision.value, action,
                        {"status": "blocked", "message": "This action requires human authority and was not executed."}, False,
                    )
                    recent_steps.append(_recent_summary(step))
                    status = "paused_for_review" if mode is ControllerMode.AUTONOMOUS else "blocked"
                    return self._finish(status, "Controller reached a forbidden high-authority action; no trusted state was modified.", goal, task_id, mode, steps, log)

                if decision is AuthorityDecision.ASK:
                    response = self.approval_callback(action) if self.approval_callback else "pause"
                    normalized = response.strip().lower()
                    if normalized != "approve":
                        outcome = "stopped_by_user" if normalized == "stop" else "rejected_by_user"
                        step = self._record_step(
                            log, steps, goal, task_id, mode, step_number, decision.value, action,
                            {"status": outcome, "message": "Action was not executed."}, False,
                        )
                        recent_steps.append(_recent_summary(step))
                        final_status = "stopped_by_user" if normalized == "stop" else "paused_for_review" if normalized == "pause" else "rejected_by_user"
                        return self._finish(final_status, "Controller is waiting for or received a human decision.", goal, task_id, mode, steps, log)

                try:
                    result = self._execute(action, context)
                    step = self._record_step(log, steps, goal, task_id, mode, step_number, decision.value, action, result, True)
                    failures = 0
                except Exception as exc:  # Handlers report only safe exception names/messages.
                    failures += 1
                    step = self._record_step(
                        log, steps, goal, task_id, mode, step_number, decision.value, action,
                        {"status": "action_failure", "message": f"{type(exc).__name__}: {exc}"}, False,
                    )
                    if failures >= MAX_CONSECUTIVE_FAILURES:
                        recent_steps.append(_recent_summary(step))
                        return self._finish("action_failure", "Controller stopped after repeated action failures.", goal, task_id, mode, steps, log)
                recent_steps.append(_recent_summary(step))
                recent_steps = recent_steps[-5:]
                if pause_every is not None and step_number % pause_every == 0:
                    return self._finish("paused_for_review", "Controller paused at the requested interval.", goal, task_id, mode, steps, log)
        except KeyboardInterrupt:
            self._record_step(
                log, steps, goal, task_id, mode, len(steps) + 1, "INTERRUPTED", None,
                {"status": "interrupted", "message": "Controller interrupted; no in-flight database transaction was retained."}, False,
            )
            return self._finish("interrupted", "Controller interrupted cleanly.", goal, task_id, mode, steps, log)

        return self._finish("max_steps", "Controller reached max_steps.", goal, task_id, mode, steps, log)

    def _choose_unit(self, goal: str, task_id: int) -> int | None:
        """Choose among bounded open candidates; never accept an invented ID."""

        with db.get_connection(self.db_path) as connection:
            db.create_tables(connection)
            units = [
                unit for unit in db.list_research_units(connection, task_id=task_id)
                if unit.status in {"active", "ready", "proposed"}
            ]
            if not units:
                return None
            if len(units) == 1:
                return units[0].unit_id
            order = {"active": 0, "ready": 1, "proposed": 2}
            candidates = sorted(
                units,
                key=lambda item: (order[item.status], -item.priority, item.unit_id or 0),
            )[:20]
            candidate_state = [
                {
                    "unit_id": unit.unit_id,
                    "title": unit.title,
                    "purpose": unit.purpose[:500],
                    "kind": unit.kind,
                    "status": unit.status,
                    "priority": unit.priority,
                    "parent_unit_id": unit.parent_unit_id,
                    "links": [
                        link.model_dump()
                        for link in db.list_research_unit_links(connection, unit.unit_id)
                    ][:8],
                }
                for unit in candidates if unit.unit_id is not None
            ]
        context = load_controller_context(task_id, self.db_path)
        prompt = {
            "goal": goal,
            "task": context.task.model_dump(),
            "candidate_units": candidate_state,
            "stored_state": context.summary,
            "instruction": (
                "Choose the single most promising open research unit for the next bounded "
                "step toward the task goal. Compare uncertainty, relevance, and available "
                "evidence. Use only a unit_id from candidate_units. Return JSON with "
                "unit_id and a short rationale. Do not propose or execute an action here."
            ),
        }
        try:
            choice = self.client.complete_json(
                LLMRequest(
                    messages=(
                        LLMMessage(role="system", content="You select one existing research unit from supplied candidates."),
                        LLMMessage(role="user", content=json.dumps(prompt, sort_keys=True)),
                    ),
                    temperature=0.0,
                    json_mode=True,
                    model_role="unit_selection",
                ),
                UnitChoice,
            )
        except LLMError as exc:
            raise ValueError(f"automatic research-unit selection failed: {exc}") from exc
        candidate_ids = {unit["unit_id"] for unit in candidate_state}
        if choice.unit_id not in candidate_ids:
            raise ValueError(f"automatic research-unit selection returned unknown unit {choice.unit_id}")
        return choice.unit_id

    def _next_action(
        self,
        goal: str,
        context: ControllerContext,
        recent_steps: list[dict[str, Any]],
    ) -> ControllerAction:
        prompt = {
            "instruction": CONTROLLER_INSTRUCTIONS,
            "goal": goal,
            "task_id": context.task.task_id,
            "research_state": context.summary,
            "active_research_unit": context.active_unit.model_dump() if context.active_unit else None,
            "active_unit_links": [link.model_dump() for link in context.unit_links],
            "recent_controller_steps": recent_steps,
        }
        decision = self.client.complete_json(
            LLMRequest(
                messages=(
                    LLMMessage(role="system", content="You choose one safe, typed next research action."),
                    LLMMessage(role="user", content=json.dumps(prompt, sort_keys=True)),
                ),
                temperature=0.0,
                json_mode=True,
                model_role="controller_action",
            ),
            ControllerDecision,
        )
        return decision.action

    def _validate_action(self, action: ControllerAction, context: ControllerContext) -> None:
        references: set[tuple[str, int]] = set()
        for reference in action.references:
            key = (reference.kind, reference.object_id)
            if key in references:
                raise ValueError(f"duplicate {reference.kind} reference {reference.object_id}")
            references.add(key)
            validate_context_reference(context, reference.kind, reference.object_id)
        if isinstance(action, ReviewStoredEvidenceAction):
            for evidence_id in action.parameters.evidence_ids:
                validate_context_reference(context, "evidence", evidence_id)
        if isinstance(action, (DesignBoundedExperimentAction, RunTrustedExperimentAction)):
            validate_context_reference(context, "code_artifact", action.parameters.artifact_id)
            get_trusted_artifact(action.parameters.artifact_id, context.task.task_id or 0, self.db_path)
        if isinstance(action, InspectExperimentResultAction):
            validate_context_reference(context, "experiment_run", action.parameters.run_id)
        if isinstance(action, ApprovePendingAction):
            with db.get_connection(self.db_path) as connection:
                db.create_tables(connection)
                if db.get_pending_entry(connection, action.parameters.pending_id) is None:
                    raise ValueError(f"pending entry {action.parameters.pending_id} does not exist")

    def _execute(self, action: ControllerAction, context: ControllerContext) -> dict[str, Any]:
        task_id = context.task.task_id
        if task_id is None:
            raise ValueError("selected task has no id")
        if isinstance(action, InspectStateAction):
            return {
                "status": "inspected",
                "counts": {key: len(items) for key, items in context.summary.items()},
            }
        if isinstance(action, ReviewStoredEvidenceAction):
            selected = set(action.parameters.evidence_ids)
            evidence = [
                item for item in context.summary["evidence"]
                if not selected or item["id"] in selected
            ]
            return {"status": "reviewed", "evidence": evidence}
        if isinstance(action, ProposeSubquestionsAction):
            units = [
                ResearchUnit(
                    task_id=task_id,
                    parent_unit_id=context.active_unit.unit_id if context.active_unit else None,
                    kind="question",
                    title=item.question[:300],
                    purpose=f"{item.rationale} Uncertainty: {item.uncertainty_note}",
                )
                for item in action.parameters.subquestions
            ]
            ids = self._create_proposed_units(units, action.references)
            return {"status": "proposed", "research_unit_ids": ids,
                    "subquestions": [item.model_dump() for item in action.parameters.subquestions]}
        if isinstance(action, RequestLiteratureReviewAction):
            parameters = action.parameters
            ids = self._create_proposed_units([
                ResearchUnit(
                    task_id=task_id,
                    parent_unit_id=context.active_unit.unit_id if context.active_unit else None,
                    kind="literature_review",
                    title=parameters.question[:300],
                    purpose=f"{parameters.rationale} Uncertainty: {parameters.uncertainty_note}",
                )
            ], action.references, block_parent=True)
            return {"status": "proposed", "research_unit_ids": ids}
        if isinstance(action, CreatePendingConjectureAction):
            pending_id = self._create_pending_conjecture(
                action, task_id, context.active_unit.unit_id if context.active_unit else None,
            )
            return {"status": "pending_created", "pending_entry_id": pending_id}
        if isinstance(action, CreatePendingOpenProblemAction):
            pending_id = self._create_pending_open_problem(
                action, task_id, context.active_unit.unit_id if context.active_unit else None,
            )
            return {"status": "pending_created", "pending_entry_id": pending_id}
        if isinstance(action, DesignBoundedExperimentAction):
            return {"status": "designed", "bounded_design": action.parameters.model_dump()}
        if isinstance(action, RunTrustedExperimentAction):
            result = run_trusted_bounded_experiment(action.parameters, task_id, self.db_path)
            if context.active_unit and context.active_unit.unit_id:
                with db.get_connection(self.db_path) as connection:
                    db.link_research_unit(
                        connection,
                        ResearchUnitLink(
                            unit_id=context.active_unit.unit_id, relation="produces",
                            object_type="experiment_run", object_id=result.run_id,
                        ),
                    )
            return {"status": "completed", "experiment_run_id": result.run_id,
                    "research_unit_id": context.active_unit.unit_id if context.active_unit else None,
                    "summary": result.summary, "output": result.output}
        if isinstance(action, InspectExperimentResultAction):
            with db.get_connection(self.db_path) as connection:
                db.create_tables(connection)
                run = next(
                    (item for item in db.list_experiment_runs(connection, task_id=task_id) if item.run_id == action.parameters.run_id),
                    None,
                )
            if run is None:
                raise ValueError(f"experiment run {action.parameters.run_id} does not exist in this task")
            return {"status": "inspected", "run_id": run.run_id, "summary": run.result_summary, "output": run.output_json}
        if isinstance(action, ReassessPlanAction):
            return {"status": "reassessed", **action.parameters.model_dump()}
        raise ValueError(f"no executable trusted handler for {action.action_type}")

    def _create_proposed_units(
        self, proposals: list[ResearchUnit], references: list[ResearchReference],
        *, block_parent: bool = False,
    ) -> list[int]:
        """Persist only activity proposals; no research claim is established."""

        if not proposals:
            return []
        with db.get_connection(self.db_path) as connection:
            db.create_tables(connection)
            existing = db.list_research_units(connection, task_id=proposals[0].task_id)
            ids: list[int] = []
            for proposal in proposals:
                match = next(
                    (
                        unit for unit in existing
                        if unit.parent_unit_id == proposal.parent_unit_id
                        and unit.kind == proposal.kind
                        and unit.title.casefold() == proposal.title.casefold()
                        and unit.purpose.casefold() == proposal.purpose.casefold()
                        and unit.status != "abandoned"
                    ),
                    None,
                )
                if match is not None and match.unit_id is not None:
                    unit_id = match.unit_id
                else:
                    unit_id = db.insert_research_unit(connection, proposal)
                    existing.append(proposal.model_copy(update={"unit_id": unit_id}))
                ids.append(unit_id)
                for reference in references:
                    if reference.kind in {"research_task", "research_unit"}:
                        continue
                    db.link_research_unit(
                        connection,
                        ResearchUnitLink(
                            unit_id=unit_id,
                            relation="investigates" if reference.kind in {
                                "open_problem", "conjecture", "derived_result",
                            } else "uses",
                            object_type=reference.kind,
                            object_id=reference.object_id,
                        ),
                    )
            if block_parent and proposals[0].parent_unit_id is not None:
                db.update_research_unit(
                    connection, proposals[0].parent_unit_id, "blocked",
                    outcome_note=f"Awaiting literature review: {proposals[0].title}",
                )
            return ids

    def _create_pending_conjecture(
        self, action: CreatePendingConjectureAction, task_id: int, unit_id: int | None = None,
    ) -> int:
        parameters = action.parameters
        candidate = Conjecture(
            statement=parameters.statement,
            task_id=task_id,
            motivation=parameters.rationale,
            expected_status="unknown",
            confidence="needs_review",
            attack_plan="; ".join(parameters.assumptions) or None,
            notes=f"Controller proposal. Uncertainty: {parameters.uncertainty_note}",
        )
        entry = PendingEntry(
            entry_type="conjecture_seed",
            payload=candidate.model_dump(),
            source_text=action.reason,
            warnings=["Controller-proposed; requires human review", f"uncertainty: {parameters.uncertainty_note}"],
        )
        return self._insert_unique_pending(entry, unit_id)

    def _create_pending_open_problem(
        self, action: CreatePendingOpenProblemAction, task_id: int, unit_id: int | None = None,
    ) -> int:
        parameters = action.parameters
        candidate = OpenProblem(
            title=parameters.question.replace("\n", " ").strip()[:80].rstrip(" .") or "Controller research question",
            statement=parameters.question,
            context=parameters.rationale,
            task_id=task_id,
            notes=(
                "Controller proposal, not an established open problem. "
                f"Dependencies/assumptions: {'; '.join(parameters.assumptions) or 'none'}. "
                f"Uncertainty: {parameters.uncertainty_note}"
            ),
        )
        entry = PendingEntry(
            entry_type="open_problem",
            payload=candidate.model_dump(),
            source_text=action.reason,
            warnings=["Controller-proposed; requires human review", f"uncertainty: {parameters.uncertainty_note}"],
        )
        return self._insert_unique_pending(entry, unit_id)

    def _insert_unique_pending(self, entry: PendingEntry, unit_id: int | None = None) -> int:
        statement = str(entry.payload.get("statement", "")).strip().casefold()
        with db.get_connection(self.db_path) as connection:
            db.create_tables(connection)
            for existing in db.list_pending_entries(connection):
                if existing.entry_type != entry.entry_type or existing.status == "rejected":
                    continue
                if str(existing.payload.get("statement", "")).strip().casefold() == statement:
                    raise ValueError("duplicate pending proposal was not created")
            if find_duplicates(connection, entry):
                raise ValueError("proposal duplicates an existing trusted research object")
            pending_id = db.insert_pending_entry(connection, entry)
            if unit_id is not None:
                db.link_research_unit(
                    connection,
                    ResearchUnitLink(
                        unit_id=unit_id, relation="produces",
                        object_type="pending_entry", object_id=pending_id,
                    ),
                )
            return pending_id

    def _record_step(
        self,
        log: JsonlControllerLog,
        steps: list[ControllerStep],
        goal: str,
        task_id: int,
        mode: ControllerMode,
        step_number: int,
        policy_decision: str,
        action: ControllerAction | None,
        result: dict[str, Any],
        success: bool,
    ) -> ControllerStep:
        action_payload = _redact_secrets(action.model_dump(), self._secret_values()) if action is not None else None
        result_payload = _redact_secrets(result, self._secret_values())
        step = ControllerStep(
            step=step_number,
            timestamp=datetime.now(timezone.utc).isoformat(),
            policy_decision=policy_decision,
            action=action_payload,
            result=result_payload,
            success=success,
        )
        log.append(
            {
                "goal": _redact_secrets(goal, self._secret_values()),
                "mode": mode.value,
                "task_id": task_id,
                "research_unit_id": self._selected_unit_id,
                "provider": self.client.metadata(),
                "step": asdict(step),
            }
        )
        steps.append(step)
        return step

    def _secret_values(self) -> tuple[str, ...]:
        api_key = self.client.configuration.api_key
        return (api_key,) if api_key else ()

    @staticmethod
    def _finish(
        status: str,
        message: str,
        goal: str,
        task_id: int,
        mode: ControllerMode,
        steps: list[ControllerStep],
        log: JsonlControllerLog,
    ) -> ControllerRunResult:
        return ControllerRunResult(status, message, goal, task_id, mode.value, tuple(steps), log.path)


def _action_fingerprint(action: ControllerAction) -> str:
    """Ignore prose so wording changes cannot bypass duplicate-action protection."""

    payload = action.model_dump()
    payload.pop("reason", None)
    payload.pop("expected_effect", None)
    return json.dumps(payload, sort_keys=True)


def _recent_summary(step: ControllerStep) -> dict[str, Any]:
    return {
        "step": step.step,
        "action_type": step.action.get("action_type") if step.action else None,
        "success": step.success,
        "result": step.result,
    }


def _redact_secrets(value: Any, secrets: tuple[str, ...]) -> Any:
    """Redact configured credentials from controller logs and displayed steps."""

    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {key: _redact_secrets(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_secrets(item, secrets) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_secrets(item, secrets) for item in value)
    return value

"""Streamlit entry point for inspecting and working with local research state."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm import LLMClient, LLMConfiguration
from src.gui.settings import settings_view
from src.gui.unit_graph import GRAPH_CSS, GRAPH_HTML, GRAPH_JS, render_unit_graph
import streamlit as st

from src import dashboard, db


st.set_page_config(page_title="Research Harness", layout="wide")
# Register in the entry point so each app runtime owns the component definition.
_UNIT_GRAPH = st.components.v2.component(
    "research_unit_graph", html=GRAPH_HTML, css=GRAPH_CSS, js=GRAPH_JS,
)


def main() -> None:
    st.title("Research Harness")
    st.caption("Choose a research task, inspect its units, and intervene only where needed.")
    load_dotenv()
    st.session_state.setdefault("database_url", os.getenv("DATABASE_URL", db.DEFAULT_DATABASE_URL))
    if st.session_state.pop("clear_api_key_field", False):
        st.session_state.pop("settings_api_key", None)
    db_path = st.session_state["database_url"]
    model_overrides = st.session_state.get("model_routes", {})
    with st.sidebar:
        st.header("Workspace")
        page = st.radio(
            "View",
            ["Dashboard", "Research Tasks", "Database Explorer", "Experiments", "System Status", "Settings"],
            index=1,
        )
        st.caption("API connections, models and database: Settings.")

    try:
        if page == "Dashboard":
            _dashboard_view(db_path)
        elif page == "Research Tasks":
            _projects_view(db_path, model_overrides)
        elif page == "Database Explorer":
            _explorer_view(db_path)
        elif page == "Experiments":
            _experiments_view(db_path)
        elif page == "Settings":
            settings_view()
        else:
            _system_status_view(db_path)
    except (ValueError, OSError, db.DatabaseError) as exc:
        st.error(str(exc))


def _dashboard_view(db_path: str) -> None:
    st.header("Dashboard")
    summary = dashboard.dashboard_summary(db_path)
    _metrics(summary["counts"])
    with st.expander("LLM and system status"):
        st.json(_gui_system_status(db_path))
    st.subheader("Active conjectures")
    _records(summary["active_conjectures"], "dashboard_conjectures")
    st.subheader("Active open problems")
    _records(summary["active_open_problems"], "dashboard_problems")
    st.subheader("Recently created research objects")
    _records(summary["recent"], "dashboard_recent")


def _projects_view(db_path: str, model_overrides: dict[str, str] | None = None) -> None:
    st.header("Research tasks")
    with st.expander("Create research task"):
        name = st.text_input("Task name", key="new_task_name")
        description = st.text_area("Research objective", key="new_task_objective")
        if st.button("Create task", key="create_task"):
            if not name.strip() or not description.strip():
                st.error("A task name and research objective are required.")
            else:
                dashboard.create_research_task(name, description, db_path)
                st.rerun()
    projects = dashboard.project_summaries(db_path)
    if not projects:
        st.caption("No research tasks are stored yet.")
        return
    names = {item["task_id"]: item["name"] for item in projects}
    selected_id = st.selectbox(
        "Research task", list(names), key="project_selector",
        format_func=lambda value: names[value],
    )
    detail = dashboard.project_detail(selected_id, db_path)
    task = detail["task"]
    st.subheader(task["name"])
    if task["description"]:
        st.text(task["description"])
    st.caption("The research task is a fixed objective. Work progresses through its research units.")

    units = detail["research_units"]
    st.subheader("Research unit tree")
    if units:
        render_unit_graph(task, units, detail["research_unit_links"], f"inspect_unit_{selected_id}", _UNIT_GRAPH)
        st.caption("Click a circle to inspect its unit. Colours show activity type, not verification status. "
                   "T = task · Numbers = unit IDs · Dashed lines = combined branches.")
        with st.expander(f"All units ({len(units)})"):
            st.dataframe(
                [{"Unit": unit["unit_id"], "Activity": unit["title"],
                  "Kind": unit["kind"], "Status": unit["status"],
                  "Parent": unit["parent_unit_id"]} for unit in units],
                width="stretch", hide_index=True,
            )
        by_id = {unit["unit_id"]: unit for unit in units}
        inspected_id = st.selectbox(
            "Inspect research unit", list(by_id), key=f"inspect_unit_{selected_id}",
            format_func=lambda value: f'U{value}: {by_id[value]["title"]}',
        )
        inspected = by_id[inspected_id]
        st.text(inspected["title"])
        st.caption(f'{inspected["kind"].replace("_", " ")} · {inspected["status"]}')
        st.text(inspected["purpose"])
        st.text(inspected.get("outcome_note") or "No outcome recorded yet.")
        with st.expander("Linked inputs and outputs"):
            st.json(detail["research_unit_links"].get(str(inspected_id), []))
        with st.expander("Models used"):
            events = dashboard.project_timeline(selected_id, db_path)
            for event in events:
                if (event.get("event_type") == "research_step_completed"
                        and event.get("object_id") == inspected_id):
                    st.json(event.get("metadata", {}))
    else:
        st.caption("No research units yet. Start a bounded autonomous run.")

    open_units = {
        unit["unit_id"]: unit for unit in units
        if unit["status"] != "abandoned"
    }
    next_unit = detail["next_research_unit"]
    if next_unit is not None:
        st.caption(f'Current frontier: {next_unit["title"]}. Automatic continuation compares recent research units.')
    focus = st.selectbox(
        "Focus for the next step",
        [None, *open_units],
        key="unit_work_selector",
        format_func=lambda value: "Let the system choose" if value is None else f'{open_units[value]["title"]} (unit {value})',
    )
    steps = st.number_input("Research steps", min_value=1, max_value=100, value=1)
    left, right = st.columns(2)
    if left.button("Run research", key="continue_research"):
        with st.spinner("Running bounded research steps..."):
            output = _run_research_gui(selected_id, int(steps), focus, False, db_path, model_overrides)
        st.session_state["task_last_action"] = {"task_id": selected_id, "output": output}
        st.rerun()
    if right.button("Survey literature from unit", key="assess_literature", disabled=focus is None):
        with st.spinner("Discovering and reading accessible full text..."):
            output = _run_research_gui(selected_id, 1, focus, True, db_path, model_overrides)
        st.session_state["task_last_action"] = {"task_id": selected_id, "output": output}
        st.rerun()
    st.caption("Source-reported claims and partial findings remain unverified; no result is silently promoted to proved.")
    last_action = st.session_state.get("task_last_action")
    if last_action and last_action["task_id"] == selected_id:
        _show_task_action(last_action)

    with st.expander("Browse stored research state"):
        tabs = st.tabs(["Overview", "Known results", "Research frontier", "Literature/evidence", "Experiments", "Timeline"])
        with tabs[0]:
            if detail["research_unit_links"]:
                st.subheader("Unit links to research records")
                st.json(detail["research_unit_links"])
        with tabs[1]:
            _section("Theorems", detail["theorems"], "project_theorems")
            _section("Reductions", detail["reductions"], "project_reductions")
            _section("Derived results", detail["derived_results"], "project_derived")
        with tabs[2]:
            _section("Conjectures — not established results", detail["conjectures"], "project_conjectures")
            _section("Open problems", detail["open_problems"], "project_open_problems")
            _section("Proof attempts", detail["proof_attempts"], "project_proof_attempts")
        with tabs[3]:
            _section("Papers", detail["papers"], "project_papers")
            _section("Evidence spans", detail["evidence_spans"], "project_evidence")
            _section("Literature notes", detail["literature_notes"], "project_notes")
            _section("Literature summaries", detail["literature_summaries"], "project_summaries")
        with tabs[4]:
            _section("Experiment runs — observations, not proofs", detail["experiment_runs"], "project_runs")
            _section("Code artifacts", detail["code_artifacts"], "project_artifacts")
        with tabs[5]:
            timeline = dashboard.project_timeline(task["task_id"], db_path)
            event_types = sorted({item["event_type"] for item in timeline})
            selected = st.multiselect("Event types", event_types, default=event_types, key="timeline_types")
            _records([item for item in timeline if item["event_type"] in selected], "project_timeline")


def _show_task_action(last_action: dict[str, Any]) -> None:
    output = last_action["output"]
    st.subheader("Last run")
    if output["status"] == "stopped":
        st.error(output["message"])
        with st.expander("Error details"):
            if output.get("error_type"):
                st.write(f'Error type: {output["error_type"]}')
            diagnostics = output.get("diagnostics", {})
            if diagnostics.get("hint"):
                st.info(diagnostics["hint"])
            if diagnostics:
                st.json(diagnostics)
            if output.get("error_details"):
                st.json(output["error_details"])
            if output.get("diagnostic_id"):
                st.caption(f'Diagnostic ID: {output["diagnostic_id"]} · Local log: data/research_errors.jsonl')
            else:
                st.caption("No local diagnostic entry was written.")
    else:
        st.write(output["message"])
    if output["status"] == "unavailable":
        st.warning("Configure an LLM provider before continuing research.")
    for unit_id in output["research_unit_ids"]:
        st.write(f"Completed research unit {unit_id}")
    for unit_id in output.get("blocked_research_unit_ids", []):
        st.write(f"Research unit {unit_id} is blocked; select it to retry or explore another direction.")


def _explorer_view(db_path: str) -> None:
    st.header("Database Explorer")
    table = st.selectbox("Existing table", list(db.EXPLORABLE_TABLES))
    filter_text = st.text_input("Filter rows")
    records = dashboard.explorer_records(table, db_path)
    if filter_text:
        needle = filter_text.casefold()
        records = [item for item in records if needle in json.dumps(item, sort_keys=True, default=str).casefold()]
    _records(records, f"explorer_{table}")


def _experiments_view(db_path: str) -> None:
    st.header("Experiment history")
    st.caption("Experiment runs are recorded observations, not proofs or established theorems.")
    runs = dashboard.experiment_records(db_path=db_path)
    _records(runs, "experiment_runs")
    if not runs:
        return
    selected = st.selectbox("Inspect experiment", [item["run_id"] for item in runs])
    run = next(item for item in runs if item["run_id"] == selected)
    st.json(run)
    if st.checkbox("Show recorded local output", key=f"show_output_{selected}"):
        st.code(dashboard.read_experiment_output(selected, db_path), language="text")


def _system_status_view(db_path: str) -> None:
    st.header("System status")
    st.caption("Credentials are intentionally excluded.")
    st.json(_gui_system_status(db_path))


def _metrics(counts: dict[str, int]) -> None:
    items = list(counts.items())
    for start in range(0, len(items), 4):
        columns = st.columns(4)
        for column, (name, value) in zip(columns, items[start : start + 4]):
            label = {"research_tasks": "Research Tasks"}.get(name, name.replace("_", " ").title())
            column.metric(label, value)


def _section(title: str, records: list[dict[str, Any]], key: str) -> None:
    st.markdown(f"#### {title}")
    _records(records, key)


def _records(records: list[dict[str, Any]], key: str) -> None:
    if not records:
        st.caption("No records.")
        return
    st.dataframe(_table_rows(records), width="stretch", hide_index=True)
    index = st.selectbox("Show full record", range(len(records)), format_func=lambda item: f"Record {item + 1}", key=f"detail_{key}")
    st.json(records[index])


def _table_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep overview tables readable even when records include nested JSON fields."""

    return [
        {
            key: json.dumps(value, sort_keys=True, default=str) if isinstance(value, (dict, list)) else value
            for key, value in record.items()
        }
        for record in records
    ]


def _run_research_gui(
    task_id: int, steps: int, unit_id: int | None, literature: bool,
    db_path: str, model_overrides: dict[str, str] | None,
) -> dict[str, Any]:
    kwargs = {"model_overrides": model_overrides} if model_overrides else {}
    if configuration := st.session_state.get("api_configuration"):
        kwargs["client"] = LLMClient(configuration=configuration, model_overrides=model_overrides)
    return dashboard.run_research_steps(task_id, steps, unit_id, literature, db_path, **kwargs)

def _gui_system_status(db_path: str) -> dict[str, Any]:
    status = dashboard.system_status(db_path)
    configuration = st.session_state.get("api_configuration") or LLMConfiguration.from_environment()
    status.update(llm_provider=configuration.provider, llm_model=configuration.model,
                  remote_llm_available=configuration.remote_enabled)
    return status


if __name__ == "__main__":
    main()

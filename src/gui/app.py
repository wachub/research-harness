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

from src.llm import LLMConfiguration
import streamlit as st

from src import dashboard, db


st.set_page_config(page_title="Research Harness", layout="wide")


def main() -> None:
    st.title("Research Harness")
    st.caption("Choose a research task, inspect its units, and intervene only where needed.")
    load_dotenv()
    with st.sidebar:
        st.header("Workspace")
        page = st.radio(
            "View",
            ["Dashboard", "Research Tasks", "Database Explorer", "Experiments", "System Status"],
            index=1,
        )
        with st.expander("Database connection"):
            db_path = st.text_input(
                "Database URL", value=os.getenv("DATABASE_URL", db.DEFAULT_DATABASE_URL), type="password",
            )
        model_overrides = _model_routing()
        st.caption("The dashboard never displays API keys or runs commands.")

    try:
        if page == "Dashboard":
            _dashboard_view(db_path)
        elif page == "Research Tasks":
            _projects_view(db_path, model_overrides)
        elif page == "Database Explorer":
            _explorer_view(db_path)
        elif page == "Experiments":
            _experiments_view(db_path)
        else:
            _system_status_view(db_path)
    except (ValueError, OSError, db.DatabaseError) as exc:
        st.error(str(exc))


def _dashboard_view(db_path: str) -> None:
    st.header("Dashboard")
    summary = dashboard.dashboard_summary(db_path)
    _metrics(summary["counts"])
    with st.expander("LLM and system status"):
        st.json(dashboard.system_status(db_path))
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
        st.graphviz_chart(_research_unit_tree(task, units, detail["research_unit_links"]), use_container_width=True)
        st.caption("Solid arrows show continuation; dashed arrows show inputs from other branches. "
                   "A finished activity can still contain unverified findings.")
        st.subheader("Unit details")
        st.dataframe(
            [
                {
                    "Unit": unit["unit_id"],
                    "Activity": unit["title"],
                    "Kind": unit["kind"],
                    "Status": unit["status"],
                    "Parent": unit["parent_unit_id"],
                }
                for unit in units
            ],
            width="stretch", hide_index=True,
        )
        by_id = {unit["unit_id"]: unit for unit in units}
        inspected_id = st.selectbox(
            "Inspect research unit", list(by_id), key=f"inspect_unit_{selected_id}",
            format_func=lambda value: f'U{value}: {by_id[value]["title"]}',
        )
        inspected = by_id[inspected_id]
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


def _dot_escape(value: Any) -> str:
    """Escape user-controlled text before placing it in a Graphviz label."""

    return str(value or "").replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _research_unit_tree(
    task: dict[str, Any], units: list[dict[str, Any]],
    links: dict[str, list[dict[str, Any]]] | None = None,
) -> str:
    """Build a compact task-rooted Graphviz tree from persisted unit parent IDs."""

    status_colors = {
        "active": "#d9f2e6",
        "proposed": "#e3ecff",
        "ready": "#e3ecff",
        "blocked": "#ffe7c2",
        "completed": "#e5e7eb",
        "finished": "#e5e7eb",
        "abandoned": "#f4d7dc",
    }
    root_label = _dot_escape("Research task\n" + str(task.get("name") or "Unnamed task"))
    lines = [
        "digraph research_units {",
        '  graph [rankdir=TB, bgcolor="transparent", pad=0.2, nodesep=0.3, ranksep=0.55, splines=ortho];',
        '  node [shape=box, style="rounded,filled", color="#64748b", fontcolor="#111827", fontname="sans", fontsize=11, margin="0.16,0.10"];',
        '  edge [color="#94a3b8", penwidth=1.2, arrowsize=0.7];',
        f'  "task-root" [shape=oval, fillcolor="#c7d2fe", label="{root_label}"];',
    ]
    known_ids = {unit.get("unit_id") for unit in units}
    for unit in units:
        unit_id = unit.get("unit_id")
        status = str(unit.get("status") or "unknown")
        title = str(unit.get("title") or "Untitled unit")
        kind = str(unit.get("kind") or "activity")
        label = _dot_escape(f"U{unit_id}\n{title}\n{kind} · {status}")
        color = status_colors.get(status.casefold(), "#eef2f7")
        lines.append(f'  "unit-{unit_id}" [fillcolor="{color}", label="{label}"];')
    for unit in units:
        unit_id = unit.get("unit_id")
        parent_id = unit.get("parent_unit_id")
        source = f'"unit-{parent_id}"' if parent_id in known_ids else '"task-root"'
        lines.append(f'  {source} -> "unit-{unit_id}";')
    for unit in units:
        for link in (links or {}).get(str(unit["unit_id"]), []):
            if (link["object_type"] == "research_unit" and link["relation"] == "uses"
                    and link["object_id"] in known_ids
                    and link["object_id"] != unit.get("parent_unit_id")):
                lines.append(f'  "unit-{link["object_id"]}" -> "unit-{unit["unit_id"]}" [style=dashed];')
    lines.append("}")
    return "\n".join(lines)


def _show_task_action(last_action: dict[str, Any]) -> None:
    output = last_action["output"]
    st.subheader("Last run")
    if output["status"] == "stopped":
        st.error(output["message"])
        with st.expander("Error details"):
            if output.get("error_type"):
                st.write(f'Error type: {output["error_type"]}')
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
    st.json(dashboard.system_status(db_path))


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
    st.dataframe(_table_rows(records), use_container_width=True, hide_index=True)
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
    return dashboard.run_research_steps(task_id, steps, unit_id, literature, db_path, **kwargs)

def _model_routing() -> dict[str, str]:
    """Collect session-only model overrides without exposing provider secrets."""

    configuration = LLMConfiguration.from_environment()
    with st.expander("LLM model routing"):
        st.caption("Session settings for the next run. Use model IDs supported by your configured API endpoint. "
                   "A model name does not enable a provider's research tools.")
        st.caption("Blank activity fields inherit the research model; blank literature substeps inherit "
                   "the literature model. Both fall back to the default.")
        default_model = st.text_input(
            "Default model", value=configuration.model, key="llm_default_model",
        ).strip()
        role_labels = {
            "research_step": "Choose next research activity",
            "proof_attempt": "Proof attempts",
            "analysis": "Analysis",
            "partial_result": "Partial results",
            "conjecture": "Conjectures",
            "open_problem": "Open problems",
            "bounded_experiment": "Experiment planning",
            "literature_review": "Literature review",
            "literature_selection": "Literature source selection",
            "literature_extraction": "Literature claim extraction",
            "unit_selection": "Research-unit selection",

        }
        overrides: dict[str, str] = {}
        if default_model and default_model != configuration.model:
            overrides["default"] = default_model
        for role, label in role_labels.items():
            selected = st.text_input(label, value="", key=f"llm_model_{role}").strip()
            if selected:
                overrides[role] = selected
    return overrides


if __name__ == "__main__":
    main()

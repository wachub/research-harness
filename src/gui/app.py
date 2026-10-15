"""Streamlit entry point for inspecting and working with local research state."""

from __future__ import annotations

import hashlib
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
from src.gui.settings import load_gui_configuration, settings_view
from src.gui.unit_graph import render_unit_graph
import streamlit as st

from src import dashboard, db


st.set_page_config(page_title="Research Harness", page_icon=":material/account_tree:", layout="wide")
from research_unit_graph import research_unit_graph as _UNIT_GRAPH

PAGES = ("Dashboard", "Research Tasks", "Database Explorer", "Experiments", "System Status", "Settings")
TASK_RECORD_COLLECTIONS = (
    ("papers", "Papers", "paper"),
    ("models", "Models", "model"),
    ("theorems", "Theorems", "theorem"),
    ("derived_results", "Derived results", "derived_result"),
    ("reductions", "Reductions", "reduction"),
    ("conjectures", "Conjectures", "conjecture"),
    ("open_problems", "Open problems", "open_problem"),
    ("proof_attempts", "Proof attempts", "proof_attempt"),
    ("evidence_spans", "Evidence spans", "evidence"),
    ("literature_notes", "Literature notes", "literature_note"),
    ("literature_summaries", "Literature summaries", "literature_summary"),
    ("experiment_runs", "Experiment runs", "experiment_run"),
    ("code_artifacts", "Code artifacts", "code_artifact"),
)


def _toggle_workspace() -> None:
    st.session_state["workspace_visible"] = not st.session_state["workspace_visible"]


def _open_record(table: str, object_id: int) -> None:
    st.session_state["navigation_target"] = "Database Explorer"
    st.session_state["explorer_mode"] = "Tables"
    st.session_state["explorer_table"] = table
    st.session_state["explorer_record_id"] = object_id
    st.session_state["explorer_pinned"] = (table, object_id)


def _open_unit(task_id: int, unit_id: int) -> None:
    st.session_state["navigation_target"] = "Research Tasks"
    st.session_state["project_selector"] = task_id
    st.session_state[f"inspect_unit_{task_id}"] = unit_id


def _open_task(task_id: int) -> None:
    st.session_state["navigation_target"] = "Research Tasks"
    st.session_state["project_selector"] = task_id


def _open_task_records(task_id: int) -> None:
    st.session_state["navigation_target"] = "Database Explorer"
    st.session_state["explorer_mode"] = "By task"
    st.session_state["explorer_task_selector"] = task_id


def main() -> None:
    load_dotenv()
    st.session_state.setdefault("database_url", os.getenv("DATABASE_URL", db.DEFAULT_DATABASE_URL))
    st.session_state.setdefault("api_configuration", load_gui_configuration())
    st.session_state.setdefault("workspace_visible", True)
    st.session_state.setdefault("active_page", "Research Tasks")
    if target := st.session_state.pop("navigation_target", None):
        st.session_state["active_page"] = target
        st.session_state["workspace_view"] = target
    st.set_page_config(initial_sidebar_state="expanded" if st.session_state["workspace_visible"] else "collapsed")
    if not st.session_state["workspace_visible"]:
        st.button("Show workspace", icon=":material/left_panel_open:",
                  key="show_workspace", on_click=_toggle_workspace)
    if st.session_state.pop("settings_clear_fields", False):
        for key in ("settings_provider", "settings_model", "settings_api_key", "settings_api_url"):
            st.session_state.pop(key, None)
    db_path = st.session_state["database_url"]
    model_overrides = st.session_state.get("model_routes", {})
    page = st.session_state["active_page"]
    if st.session_state["workspace_visible"]:
        with st.sidebar:
            st.header("Research Harness")
            st.button("Hide workspace", icon=":material/left_panel_close:",
                      key="hide_workspace", on_click=_toggle_workspace)
            page = st.radio("View", PAGES, index=PAGES.index(page), key="workspace_view")
            st.session_state["active_page"] = page
            st.caption("Local research workspace")

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


def _task_picker(projects: list[dict[str, Any]], selector_key: str, table_key: str) -> int:
    names = {item["task_id"]: item["name"] for item in projects}
    if len(projects) > 1:
        with st.expander(f"Browse all tasks ({len(projects)})"):
            chosen = st.dataframe(
                [{"Task": item["name"], "Units": item["research_units"],
                  "Theorems": item["theorems"], "Experiments": item["experiment_runs"]}
                 for item in projects],
                key=table_key, on_select="rerun", selection_mode="single-row",
                hide_index=True, height=min(250, 38 * (len(projects) + 1)),
            )
        selection = tuple(chosen.selection.rows)
        if (selection and selection != st.session_state.get(f"{table_key}_handled")
                and selection[0] < len(projects)):
            st.session_state[selector_key] = projects[chosen.selection.rows[0]]["task_id"]
        st.session_state[f"{table_key}_handled"] = selection
    if st.session_state.get(selector_key) not in names:
        st.session_state[selector_key] = projects[0]["task_id"]
    return st.selectbox("Research task", list(names), key=selector_key,
                        format_func=lambda value: names[value])


def _record_summary(record: dict[str, Any] | None) -> str:
    if not record:
        return "Record unavailable"
    for key in ("title", "name", "statement", "result_summary", "description",
                "quote_or_summary", "summary", "path", "purpose"):
        if value := record.get(key):
            return str(value).replace("\n", " ")[:160]
    return "Stored record"


def _record_detail(record: dict[str, Any], *, raw_fields: bool = True) -> None:
    """Show the substantive content first; keep every stored field available on demand."""

    st.markdown(f"**{_record_summary(record)}**")
    identity = next((f"{key.replace('_', ' ')} {record[key]}" for key in
                     ("unit_id", "theorem_id", "conjecture_id", "artifact_id", "run_id", "id")
                     if isinstance(record.get(key), int)), None)
    metadata = [str(value) for value in (identity, record.get("status"), record.get("verification_status"))
                if value]
    if metadata:
        st.caption(" · ".join(metadata))
    for key in ("statement", "result_summary", "description", "quote_or_summary",
                "summary", "purpose", "outcome_note", "notes", "path"):
        value = record.get(key)
        if value not in (None, "", [], {}):
            st.caption(key.replace("_", " ").capitalize())
            st.write(value)
    if raw_fields:
        with st.expander("All stored fields"):
            st.json(record)


def _select_inspected_unit(task_id: int, unit_id: int) -> None:
    st.session_state[f"inspect_unit_{task_id}"] = unit_id


def _focus_unit(unit_id: int) -> None:
    st.session_state["unit_work_selector"] = unit_id


def _unit_inspector(detail: dict[str, Any], db_path: str) -> None:
    task_id = detail["task"]["task_id"]
    units = detail["research_units"]
    by_id = {unit["unit_id"]: unit for unit in units}
    st.markdown("#### Unit details")
    inspected_id = st.selectbox(
        "Inspect research unit", list(by_id), key=f"inspect_unit_{task_id}",
        format_func=lambda value: f'U{value}: {by_id[value]["title"]}',
    )
    inspected = by_id[inspected_id]
    with st.container(border=True):
        st.markdown(f'**U{inspected_id} · {inspected["title"]}**')
        st.caption(f'{inspected["kind"].replace("_", " ")} · {inspected["status"]}')
        st.caption("Purpose")
        st.write(inspected["purpose"])
        if inspected.get("outcome_note"):
            st.caption("Outcome")
            st.write(inspected["outcome_note"])
        else:
            st.caption("No outcome recorded yet.")
        relatives = [unit for unit in units if unit.get("parent_unit_id") == inspected_id]
        with st.container(horizontal=True):
            st.button("Use as run focus", key=f"unit_focus_{task_id}_{inspected_id}",
                      on_click=_focus_unit, args=(inspected_id,),
                      disabled=inspected["status"] == "abandoned")
            if parent := inspected.get("parent_unit_id"):
                st.button(f"Parent U{parent}", key=f"unit_parent_{task_id}_{inspected_id}",
                          on_click=_select_inspected_unit, args=(task_id, parent))
            for child in relatives:
                st.button(f'Child U{child["unit_id"]}',
                          key=f'unit_child_{task_id}_{inspected_id}_{child["unit_id"]}',
                          on_click=_select_inspected_unit, args=(task_id, child["unit_id"]))

    st.markdown("#### Linked research records")
    links = dashboard.unit_connections(task_id, inspected_id, db_path)
    if not links:
        st.caption("No records are directly linked to this unit. Task-wide records are available in Database Explorer.")
    else:
        rows = [
            {"Relation": link["relation"], "Type": link["object_type"].replace("_", " "),
             "ID": link["object_id"], "Stored content": _record_summary(link["record"])}
            for link in links
        ]
        chosen = st.dataframe(rows, key=f"unit_links_{task_id}_{inspected_id}",
                              on_select="rerun", selection_mode="single-row",
                              hide_index=True, height=min(260, 38 * (len(rows) + 1)))
        selected_index = chosen.selection.rows[0] if chosen.selection.rows else 0
        link = links[selected_index if selected_index < len(links) else 0]
        if link["record"] is None:
            st.warning("This link points to a record that is no longer available.")
        else:
            _record_detail(link["record"])
        if link["object_type"] == "research_unit" and link["object_id"] in by_id:
            st.button("Inspect linked unit", key=f"inspect_linked_unit_{task_id}_{inspected_id}",
                      on_click=_select_inspected_unit, args=(task_id, link["object_id"]))
        elif link["record"] is not None:
            st.button("Open record in Database Explorer", key=f"open_unit_record_{task_id}_{inspected_id}",
                      on_click=_open_record, args=(link["table"], link["object_id"]))


def _projects_view(db_path: str, model_overrides: dict[str, str] | None = None) -> None:
    st.header("Research tasks")
    with st.expander("Create research task", icon=":material/add:"):
        with st.form("create_research_task"):
            name = st.text_input("Task name", key="new_task_name")
            description = st.text_area("Research objective", key="new_task_objective")
            submitted = st.form_submit_button("Create task", type="primary")
        if submitted:
            if not name.strip() or not description.strip():
                st.error("A task name and research objective are required.")
            else:
                dashboard.create_research_task(name, description, db_path)
                st.rerun()
    projects = dashboard.project_summaries(db_path)
    if not projects:
        st.caption("No research tasks are stored yet.")
        return
    selected_id = _task_picker(projects, "project_selector", "task_picker_rows")
    detail = dashboard.project_detail(selected_id, db_path)
    task = detail["task"]
    with st.container(border=True):
        st.subheader(task["name"])
        if task["description"]:
            st.write(task["description"])
        st.caption("Fixed objective · progress is recorded in research units")
        st.button("Explore task records", icon=":material/database:",
                  key="open_task_records", on_click=_open_task_records, args=(selected_id,))

    units = detail["research_units"]
    st.subheader("Research units")
    if units:
        render_unit_graph(task, units, detail["research_unit_links"], f"inspect_unit_{selected_id}", _UNIT_GRAPH)
        st.caption("Select a node to inspect its unit. Colours show activity type, not verification status. "
                   "T = task · U# = unit ID · Dashed lines = combined branches.")
        with st.expander(f"All units ({len(units)})", icon=":material/list:"):
            st.dataframe(
                [{"Unit": unit["unit_id"], "Activity": unit["title"],
                  "Kind": unit["kind"], "Status": unit["status"],
                  "Parent": unit["parent_unit_id"]} for unit in units],
                width="stretch", hide_index=True,
            )
        _unit_inspector(detail, db_path)
        with st.expander("Models and retrieval details", icon=":material/info:"):
            events = dashboard.project_timeline(selected_id, db_path)
            for event in events:
                if (event.get("event_type") == "research_step_completed"
                        and event.get("object_id") == st.session_state[f"inspect_unit_{selected_id}"]):
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
    with st.container(border=True):
        st.markdown("#### Continue research")
        focus = st.selectbox(
            "Focus for the next step",
            [None, *open_units],
            key="unit_work_selector",
            format_func=lambda value: "Let the system choose" if value is None else f'{open_units[value]["title"]} (unit {value})',
        )
        steps = st.number_input("Research steps", min_value=1, max_value=100, value=1)
        with st.container(horizontal=True):
            run_clicked = st.button("Run research", key="continue_research", type="primary",
                                    icon=":material/play_arrow:")
            survey_clicked = st.button("Survey literature from unit", key="assess_literature",
                                       disabled=focus is None, icon=":material/menu_book:")
        st.caption("Source-reported claims and partial findings remain unverified.")
    if run_clicked:
        with st.spinner("Running bounded research steps..."):
            output = _run_research_gui(selected_id, int(steps), focus, False, db_path, model_overrides)
        st.session_state["task_last_action"] = {"task_id": selected_id, "output": output}
        st.rerun()
    if survey_clicked:
        with st.spinner("Discovering and reading accessible full text..."):
            output = _run_research_gui(selected_id, 1, focus, True, db_path, model_overrides)
        st.session_state["task_last_action"] = {"task_id": selected_id, "output": output}
        st.rerun()
    last_action = st.session_state.get("task_last_action")
    if last_action and last_action["task_id"] == selected_id:
        _show_task_action(last_action)

    history = st.expander("Task timeline", icon=":material/history:", on_change="rerun")
    if history.open:
        with history:
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
            if trace := diagnostics.get("literature_trace"):
                st.caption("Literature stages (no source text or full URLs)")
                st.dataframe(trace, hide_index=True, width="stretch")
            if diagnostics:
                st.json({key: value for key, value in diagnostics.items() if key != "literature_trace"})
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
    st.header("Database explorer")
    st.caption("Browse tasks and their linked research, or inspect the underlying tables. This view is read-only.")
    st.session_state.setdefault("explorer_mode", "By task")
    mode = st.segmented_control("Browse", ("By task", "Tables"), key="explorer_mode")
    if mode == "Tables":
        _explorer_tables(db_path)
    else:
        _explorer_by_task(db_path)


def _object_unit_buttons(object_type: str, object_id: int, db_path: str, key: str) -> None:
    links = dashboard.object_unit_connections(object_type, object_id, db_path)
    st.markdown("#### Linked research units")
    if not links:
        st.caption("No research unit directly links to this record.")
        return
    for index, link in enumerate(links):
        unit = link["unit"]
        st.button(
            f'U{unit["unit_id"]} · {link["relation"]} · {unit["title"][:100]}',
            key=f"{key}_{index}", on_click=_open_unit,
            args=(unit["task_id"], unit["unit_id"]),
        )


def _explorer_by_task(db_path: str) -> None:
    projects = dashboard.project_summaries(db_path)
    if not projects:
        st.caption("No research tasks are stored yet.")
        return
    task_id = _task_picker(projects, "explorer_task_selector", "explorer_task_rows")
    detail = dashboard.project_detail(task_id, db_path)
    task = detail["task"]
    st.subheader(task["name"])
    if task["description"]:
        st.text(task["description"])
    units = detail["research_units"]
    st.caption(f'{len(units)} research units · {sum(len(items) for items in detail["research_unit_links"].values())} direct record links')
    if units:
        with st.expander("Research unit map", icon=":material/account_tree:"):
            render_unit_graph(task, units, detail["research_unit_links"],
                              f"inspect_unit_{task_id}", _UNIT_GRAPH)
        _unit_inspector(detail, db_path)
    else:
        st.caption("This task has no research units yet.")

    st.markdown("#### Task records")
    collections = [(key, label, object_type) for key, label, object_type in TASK_RECORD_COLLECTIONS
                   if detail[key]]
    if not collections:
        st.caption("No task-scoped records are stored yet.")
        return
    selected = st.selectbox(
        "Record type", collections, key=f"task_record_type_{task_id}",
        format_func=lambda item: f'{item[1]} ({len(detail[item[0]])})',
    )
    collection_key, _label, object_type = selected
    records = detail[collection_key]
    id_key = db.LINKED_OBJECT_KEYS[object_type]
    rows = [{"ID": record.get(id_key), "Summary": _record_summary(record),
             "Status": record.get("status") or record.get("confidence") or ""}
            for record in records]
    chosen = st.dataframe(rows, key=f"task_records_{task_id}_{collection_key}",
                          on_select="rerun", selection_mode="single-row", hide_index=True,
                          height=min(300, 38 * (len(rows) + 1)))
    selected_index = chosen.selection.rows[0] if chosen.selection.rows else 0
    record = records[selected_index if selected_index < len(records) else 0]
    _record_detail(record)
    object_id = record.get(id_key)
    if isinstance(object_id, int):
        st.button("Open in tables", key=f"task_open_record_{task_id}_{collection_key}",
                  on_click=_open_record, args=(db.LINKED_OBJECT_TABLES[object_type], object_id))
        _object_unit_buttons(object_type, object_id, db_path, f"task_object_units_{task_id}_{collection_key}")


def _explorer_tables(db_path: str) -> None:
    table = st.selectbox("Existing table", list(db.EXPLORABLE_TABLES), key="explorer_table")
    filter_text = st.text_input("Filter shown rows", key="explorer_filter",
                                help="The table shows up to 500 recent records; this filter applies to those rows.")
    records = dashboard.explorer_records(table, db_path)
    object_type = db.EXPLORER_OBJECT_TYPES.get(table)
    id_key = db.LINKED_OBJECT_KEYS.get(object_type) if object_type else None
    requested_id = st.session_state.pop("explorer_record_id", None)
    pinned = st.session_state.get("explorer_pinned")
    pinned_id = pinned[1] if pinned and pinned[0] == table else None
    if isinstance(pinned_id, int) and object_type and id_key:
        if not any(item.get(id_key) == pinned_id for item in records):
            if record := dashboard.object_record(object_type, pinned_id, db_path):
                records.insert(0, record)
    if filter_text:
        needle = filter_text.casefold()
        records = [item for item in records
                   if needle in json.dumps(item, sort_keys=True, default=str).casefold()]
    if not records:
        st.caption("No rows match the current table and filter.")
        return
    filter_id = hashlib.sha1(filter_text.encode("utf-8")).hexdigest()[:8]
    grid_key = f"explorer_rows_{table}_{filter_id}"
    chosen = st.dataframe(_table_rows(records), key=grid_key,
                          on_select="rerun", selection_mode="single-row",
                          width="stretch", hide_index=True)
    index_key = f"explorer_index_{table}"
    selection = tuple(chosen.selection.rows)
    if selection and selection != st.session_state.get(f"{grid_key}_handled") and selection[0] < len(records):
        st.session_state[index_key] = selection[0]
    st.session_state[f"{grid_key}_handled"] = selection
    if isinstance(requested_id, int) and id_key:
        st.session_state[index_key] = next(
            (index for index, item in enumerate(records) if item.get(id_key) == requested_id), 0,
        )
    if st.session_state.get(index_key) not in range(len(records)):
        st.session_state[index_key] = 0
    index = st.session_state[index_key]
    record = records[index]
    st.caption(f"Showing row {index + 1} of {len(records)} · select another row above to inspect it")
    _record_detail(record)
    if table == "research_tasks" and isinstance(record.get("task_id"), int):
        st.button("Open research task", key="explorer_open_task", on_click=_open_task,
                  args=(record["task_id"],))
    elif table == "research_units" and isinstance(record.get("unit_id"), int):
        st.button("Open research unit", key="explorer_open_unit", on_click=_open_unit,
                  args=(record["task_id"], record["unit_id"]))
    if object_type and id_key and isinstance(record.get(id_key), int):
        _object_unit_buttons(object_type, record[id_key], db_path, f"explorer_object_units_{table}_{record[id_key]}")


def _experiments_view(db_path: str) -> None:
    st.header("Experiment history")
    st.caption("Experiment runs are recorded observations, not proofs or established theorems.")
    runs = dashboard.experiment_records(db_path=db_path)
    if not runs:
        st.caption("No experiment runs recorded yet.")
        return
    selected = st.selectbox("Inspect experiment", [item["run_id"] for item in runs],
                            format_func=lambda value: f'Run {value} · {_record_summary(next(item for item in runs if item["run_id"] == value))}')
    run = next(item for item in runs if item["run_id"] == selected)
    with st.container(border=True):
        _record_detail(run)
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


def _records(records: list[dict[str, Any]], key: str) -> None:
    if not records:
        st.caption("No records.")
        return
    rows = [{"Summary": _record_summary(record),
             "Type": str(record.get("object_type") or record.get("entry_type") or "").replace("_", " "),
             "Status": record.get("status") or ""} for record in records]
    chosen = st.dataframe(rows, key=f"list_{key}", on_select="rerun",
                          selection_mode="single-row", width="stretch", hide_index=True)
    if chosen.selection.rows and chosen.selection.rows[0] < len(records):
        _record_detail(records[chosen.selection.rows[0]], raw_fields=key != "project_timeline")


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

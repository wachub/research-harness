"""Streamlit entry point for inspecting and reviewing local research state."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src import dashboard, db


st.set_page_config(page_title="Research Harness", layout="wide")


def main() -> None:
    st.title("Research Harness")
    st.caption("Human review and inspection panel. Pending proposals, conjectures, and experiments remain distinct.")
    with st.sidebar:
        st.header("Workspace")
        db_path = st.text_input("Database path", value=str(db.DEFAULT_DB_PATH))
        page = st.radio(
            "View",
            ["Dashboard", "Projects", "Review Queue", "Database Explorer", "Experiments", "System Status"],
        )
        st.caption("The dashboard never displays API keys or runs commands.")

    try:
        if page == "Dashboard":
            _dashboard_view(db_path)
        elif page == "Projects":
            _projects_view(db_path)
        elif page == "Review Queue":
            _review_view(db_path)
        elif page == "Database Explorer":
            _explorer_view(db_path)
        elif page == "Experiments":
            _experiments_view(db_path)
        else:
            _system_status_view(db_path)
    except (ValueError, OSError) as exc:
        st.error(str(exc))


def _dashboard_view(db_path: str) -> None:
    st.header("Dashboard")
    summary = dashboard.dashboard_summary(db_path)
    _metrics(summary["counts"])
    with st.expander("LLM and system status"):
        st.json(dashboard.system_status(db_path))
    left, right = st.columns(2)
    with left:
        st.subheader("Pending human review")
        _records(summary["pending"], "dashboard_pending")
    with right:
        st.subheader("Active conjectures")
        _records(summary["active_conjectures"], "dashboard_conjectures")
        st.subheader("Active open problems")
        _records(summary["active_open_problems"], "dashboard_problems")
    st.subheader("Recently created research objects")
    _records(summary["recent"], "dashboard_recent")


def _projects_view(db_path: str) -> None:
    st.header("Research projects")
    projects = dashboard.project_summaries(db_path)
    _records(projects, "projects")
    if not projects:
        return
    by_label = {f"{item['cluster_id']}: {item['name']}": item["cluster_id"] for item in projects}
    selected_label = st.selectbox("Open project", list(by_label), key="project_selector")
    detail = dashboard.project_detail(by_label[selected_label], db_path)
    cluster = detail["cluster"]
    st.subheader(cluster["name"])
    tabs = st.tabs(["Overview", "Known results", "Research frontier", "Literature/evidence", "Experiments", "Pending proposals", "Timeline"])
    with tabs[0]:
        st.json(cluster)
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
        _section("Pending proposals — require human curation", detail["pending_entries"], "project_pending")
    with tabs[6]:
        timeline = dashboard.project_timeline(cluster["cluster_id"], db_path)
        event_types = sorted({item["event_type"] for item in timeline})
        selected = st.multiselect("Event types", event_types, default=event_types, key="timeline_types")
        _records([item for item in timeline if item["event_type"] in selected], "project_timeline")


def _review_view(db_path: str) -> None:
    st.header("Pending / Review Queue")
    st.caption("Approval uses the existing curation path and is the only way this dashboard creates durable research records.")
    entries = dashboard.pending_review_items(db_path)
    status_values = sorted({item["status"] for item in entries})
    statuses = st.multiselect("Status", status_values, default=status_values)
    for entry in (item for item in entries if item["status"] in statuses):
        title = entry["payload"].get("title") or entry["payload"].get("name") or entry["payload"].get("statement") or "Untitled proposal"
        with st.expander(f"{entry['id']}: [{entry['status']}] {entry['entry_type']} — {str(title)[:100]}"):
            st.json(entry)
            if entry["status"] not in {"pending", "flagged"}:
                continue
            reason = st.text_input("Reason (used for reject/flag)", key=f"review_reason_{entry['id']}")
            approve, reject, flag = st.columns(3)
            if approve.button("Approve", key=f"approve_{entry['id']}"):
                result = dashboard.approve_review_item(entry["id"], db_path)
                st.success(f"Approved into {result['inserted_table']}:{result['inserted_id']}")
                st.rerun()
            if reject.button("Reject", key=f"reject_{entry['id']}"):
                dashboard.reject_review_item(entry["id"], reason or None, db_path)
                st.success("Rejected pending proposal.")
                st.rerun()
            if flag.button("Flag", key=f"flag_{entry['id']}"):
                if not reason:
                    st.warning("A flag reason is required.")
                else:
                    dashboard.flag_review_item(entry["id"], reason, db_path)
                    st.success("Flagged pending proposal.")
                    st.rerun()


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
            column.metric(name.replace("_", " ").title(), value)


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


if __name__ == "__main__":
    main()

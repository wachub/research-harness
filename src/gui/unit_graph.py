"""Prepare the research-unit graph and keep graph selection UI-only."""

from typing import Any

import streamlit as st


KIND_COLORS = {
    "literature_review": "#93c5fd",
    "proof_attempt": "#c4b5fd",
    "bounded_experiment": "#fdba74",
    "experiment": "#fdba74",
    "analysis": "#99f6e4",
    "partial_result": "#86efac",
    "conjecture": "#f9a8d4",
    "hypothesis": "#f9a8d4",
    "open_problem": "#fde047",
    "question": "#fde047",
}


def graph_data(task: dict, units: list[dict], links: dict, selected: int) -> dict:
    """Place units in chronological columns while continuations retain their lane."""
    nodes = [{"id": None, "x": 26, "y": 28, "label": "T",
              "title": task["name"], "color": "#cbd5e1"}]
    positions = {None: nodes[0]}
    lane_ends = {0: None}
    edges = []
    legend = {}
    for index, unit in enumerate(sorted(units, key=lambda item: item["unit_id"]), 1):
        unit_id = unit["unit_id"]
        parent_id = unit.get("parent_unit_id")
        parent = positions.get(parent_id, positions[None])
        lane = (parent["y"] - 28) // 54
        if lane_ends.get(lane) != parent["id"]:
            lane = len(lane_ends)
        lane_ends[lane] = unit_id
        kind = unit["kind"].replace("_", " ").capitalize()
        color = KIND_COLORS.get(unit["kind"], "#cbd5e1")
        node = {"id": unit_id, "x": 26 + index * 68, "y": 28 + lane * 54,
                "label": f"U{unit_id}", "color": color,
                "title": f'U{unit_id}: {kind} · {unit["status"]} · {unit["title"]}'}
        nodes.append(node)
        positions[unit_id] = node
        legend[kind] = color
        edges.append({"source": parent["id"], "target": unit_id, "merge": False})
    for unit in units:
        for link in links.get(str(unit["unit_id"]), []):
            if (link["object_type"] == "research_unit" and link["relation"] == "uses"
                    and link["object_id"] in positions
                    and link["object_id"] != unit.get("parent_unit_id")):
                edges.append({"source": link["object_id"], "target": unit["unit_id"], "merge": True})
    return {"nodes": nodes, "edges": edges, "selected": selected,
            "legend": [{"label": label, "color": color} for label, color in sorted(legend.items())]}


def apply_selection(value: Any, valid_ids: set[int], selection_key: str) -> None:
    if type(value) is int and value in valid_ids:
        st.session_state[selection_key] = value


def render_unit_graph(task: dict, units: list[dict], links: dict, selection_key: str, component) -> None:
    valid_ids = {unit["unit_id"] for unit in units}
    if st.session_state.get(selection_key) not in valid_ids:
        st.session_state[selection_key] = units[0]["unit_id"]
    component_key = f'unit_graph_{task["task_id"]}'

    def on_click():
        clicked = st.session_state.get(component_key, {}).get("clicked")
        apply_selection(clicked, valid_ids, selection_key)

    component(key=component_key, data=graph_data(task, units, links, st.session_state[selection_key]),
              on_clicked_change=on_click)

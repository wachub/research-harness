"""Small, selectable research-unit graph. All state changes are UI-only."""

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
    """Give each unit a chronological column; continuations retain their lane."""
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
                "label": str(unit_id), "color": color,
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
            "width": nodes[-1]["x"] + 28, "height": 56 + (len(lane_ends) - 1) * 54,
            "legend": [{"label": label, "color": color} for label, color in sorted(legend.items())]}


GRAPH_JS = """
export default function(component) {
  const {data, parentElement, setTriggerValue} = component;
  const viewport = parentElement.querySelector('.viewport');
  const legend = parentElement.querySelector('.legend');
  const doc = viewport.ownerDocument;
  const svgNode = (tag, attributes) => {
    const node = doc.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
    return node;
  };
  const svg = svgNode('svg', {width: data.width, height: data.height,
                             'aria-label': 'Research units, chronological left to right'});
  const positions = new Map(data.nodes.map(node => [node.id, node]));
  for (const edge of data.edges) {
    const a = positions.get(edge.source), b = positions.get(edge.target);
    svg.appendChild(svgNode('line', {x1:a.x, y1:a.y, x2:b.x, y2:b.y,
      class:'edge', 'stroke-dasharray': edge.merge ? '4 4' : 'none'}));
  }
  const groups = [];
  for (const node of data.nodes) {
    const selected = node.id !== null && node.id === data.selected;
    const group = svgNode('g', {transform: 'translate(' + node.x + ',' + node.y + ')',
      class: node.id === null ? 'root' : 'unit', 'aria-label': node.title});
    const title = svgNode('title', {});
    title.textContent = node.title;
    group.appendChild(title);
    group.appendChild(svgNode('circle', {r:21, fill:'transparent', stroke:'none'}));
    group.appendChild(svgNode('circle', {r:18, class:'selection', opacity:selected ? 1 : 0}));
    group.appendChild(svgNode('circle', {r:14, fill:node.color, class:'dot'}));
    const text = svgNode('text', {'text-anchor':'middle', dy:'.35em'});
    text.textContent = node.label;
    group.appendChild(text);
    if (node.id !== null) {
      group.setAttribute('role', 'button');
      group.setAttribute('tabindex', '0');
      group.setAttribute('aria-pressed', String(selected));
      const choose = () => {
        for (const other of groups) {
          other.setAttribute('aria-pressed', String(other === group));
          other.querySelector('.selection').setAttribute('opacity', other === group ? '1' : '0');
        }
        setTriggerValue('clicked', node.id);
      };
      group.onclick = choose;
      group.onkeydown = event => {
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); choose(); }
      };
      groups.push(group);
    }
    svg.appendChild(group);
  }
  const scrollLeft = viewport.scrollLeft, scrollTop = viewport.scrollTop;
  const focused = viewport.querySelector('g:focus')?.getAttribute('aria-label');
  viewport.replaceChildren(svg);
  viewport.scrollLeft = scrollLeft;
  viewport.scrollTop = scrollTop;
  if (focused) groups.find(group => group.getAttribute('aria-label') === focused)?.focus({preventScroll:true});
  legend.replaceChildren();
  for (const item of data.legend) {
    const entry = doc.createElement('span'), dot = doc.createElement('i');
    dot.style.backgroundColor = item.color;
    entry.appendChild(dot);
    entry.appendChild(doc.createTextNode(item.label));
    legend.appendChild(entry);
  }
  return () => { for (const group of groups) { group.onclick = null; group.onkeydown = null; } };
}
"""

GRAPH_HTML = '<div class="viewport"></div><div class="legend"></div>'
GRAPH_CSS = """
    .viewport { max-height:220px; overflow:auto; }
    svg { display:block; max-width:none; }
    .edge { stroke:var(--st-text-color); stroke-opacity:.55; stroke-width:1.5; }
    .dot { stroke:#334155; stroke-width:1.2; }
    .selection { fill:none; stroke:var(--st-text-color); stroke-width:2; }
    .unit { cursor:pointer; outline:none; }
    .unit:hover .dot, .unit:focus .dot { stroke-width:3; }
    text { fill:#111827; font:10px sans-serif; pointer-events:none; }
    .legend { display:flex; flex-wrap:wrap; gap:6px 14px; margin-top:8px;
              font-size:12px; color:var(--st-text-color); }
    .legend span { display:inline-flex; align-items:center; gap:5px; }
    .legend i { width:9px; height:9px; border-radius:50%; display:inline-block; }
    """


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

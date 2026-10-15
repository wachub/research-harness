"""Locally bundled, read-only Cytoscape.js research-unit graph."""

from collections.abc import Callable

import streamlit as st


_GRAPH = st.components.v2.component(
    "research-unit-graph.research_unit_graph",
    js="index-*.js",
    html="""
        <div class="graph-shell">
          <div class="graph-toolbar">
            <span class="graph-heading">Unit map</span>
            <input class="graph-search" type="search" aria-label="Find research unit"
                   placeholder="Find a unit" />
            <button class="graph-fit" type="button" aria-label="Fit graph">Fit</button>
            <button class="graph-zoom-in" type="button" aria-label="Zoom in">+</button>
            <button class="graph-zoom-out" type="button" aria-label="Zoom out">−</button>
          </div>
          <div class="graph-canvas" role="img" aria-label="Research-unit graph"></div>
          <div class="graph-footer"><span class="graph-legend"></span>
            <span class="graph-hint">Drag to pan · Scroll to zoom</span></div>
        </div>
    """,
    css="""
        .graph-shell { border:1px solid var(--st-border-color, #374151);
          border-radius:10px; background:var(--st-secondary-background-color, #111827);
          overflow:hidden; color:var(--st-text-color, #e5e7eb); font-family:var(--st-font, sans-serif); }
        .graph-toolbar { display:flex; align-items:center; gap:6px; padding:8px 10px;
          border-bottom:1px solid var(--st-border-color, #374151); }
        .graph-heading { font-size:12px; font-weight:600; margin-right:auto; white-space:nowrap; }
        .graph-toolbar button, .graph-search { background:var(--st-background-color, #111827);
          color:var(--st-text-color, #e5e7eb); border:1px solid var(--st-border-color, #475569);
          border-radius:6px; height:28px; font:inherit; font-size:12px; }
        .graph-toolbar button { min-width:28px; padding:0 8px; cursor:pointer; }
        .graph-toolbar button:hover { border-color:var(--st-primary-color, #f43f5e); }
        .graph-search { box-sizing:border-box; width:min(180px, 32%); padding:0 8px; }
        .graph-canvas { width:100%; height:300px; }
        .graph-footer { display:flex; justify-content:space-between; gap:12px; align-items:center;
          padding:8px 10px; border-top:1px solid var(--st-border-color, #374151);
          font-size:11px; opacity:.88; }
        .graph-legend { display:flex; flex-wrap:wrap; gap:5px 12px; }
        .graph-legend span { display:inline-flex; align-items:center; gap:5px; }
        .graph-legend i { width:8px; height:8px; border-radius:50%; display:inline-block; }
        .graph-hint { white-space:nowrap; }
        @media (max-width:600px) { .graph-heading, .graph-hint { display:none; }
          .graph-search { width:auto; flex:1; } .graph-canvas { height:260px; } }
    """,
)


def research_unit_graph(data: dict, *, key: str, on_clicked_change: Callable[[], None]) -> None:
    """Render a selectable graph; only integer unit IDs are sent back to Python."""
    _GRAPH(data=data, key=key, on_clicked_change=on_clicked_change)

import cytoscape, { type Core, type ElementDefinition } from "cytoscape";
import { type FrontendRenderer, type FrontendRendererArgs } from "@streamlit/component-v2-lib";

type Node = {
  id: number | null;
  x: number;
  y: number;
  label: string;
  title: string;
  color: string;
};
type Edge = { source: number | null; target: number; merge: boolean };
type GraphData = {
  nodes: Node[];
  edges: Edge[];
  selected: number;
  legend: { label: string; color: string }[];
};
type GraphState = { clicked: number };
type Parent = FrontendRendererArgs<GraphState, GraphData>["parentElement"];
type Viewport = { zoom: number; pan: { x: number; y: number } };

const instances = new WeakMap<Parent, Core>();
const viewports = new WeakMap<Parent, Viewport>();
const graphId = (id: number | null) => (id === null ? "task" : `unit-${id}`);

const render: FrontendRenderer<GraphState, GraphData> = (component) => {
  const { data, parentElement, setTriggerValue } = component;
  const graph = parentElement.querySelector<HTMLElement>(".graph-canvas");
  const legend = parentElement.querySelector<HTMLElement>(".graph-legend");
  const hint = parentElement.querySelector<HTMLElement>(".graph-hint");
  const search = parentElement.querySelector<HTMLInputElement>(".graph-search");
  const fit = parentElement.querySelector<HTMLButtonElement>(".graph-fit");
  const zoomIn = parentElement.querySelector<HTMLButtonElement>(".graph-zoom-in");
  const zoomOut = parentElement.querySelector<HTMLButtonElement>(".graph-zoom-out");
  if (!graph || !legend || !hint || !search || !fit || !zoomIn || !zoomOut) {
    throw new Error("Research-unit graph controls are missing");
  }

  const prior = instances.get(parentElement);
  if (prior) {
    viewports.set(parentElement, { zoom: prior.zoom(), pan: prior.pan() });
    prior.destroy();
  }
  const host = parentElement instanceof ShadowRoot ? parentElement.host : parentElement;
  const theme = getComputedStyle(host);
  const ink = theme.getPropertyValue("--st-text-color").trim() || "#e5e7eb";
  const muted = theme.getPropertyValue("--st-border-color").trim() || "#64748b";
  const accent = theme.getPropertyValue("--st-primary-color").trim() || "#f43f5e";

  const elements: ElementDefinition[] = [
    ...data.nodes.map((node) => ({
      data: { id: graphId(node.id), label: node.label, title: node.title,
        color: node.color, unitId: node.id },
      position: { x: node.x, y: node.y },
      classes: node.id === null ? "task" : "unit",
    })),
    ...data.edges.map((edge, index) => ({
      data: { id: `edge-${index}`, source: graphId(edge.source), target: graphId(edge.target) },
      classes: edge.merge ? "merge" : "primary",
    })),
  ];
  const cy = cytoscape({
    container: graph,
    elements,
    layout: { name: "preset", fit: true, padding: 42 },
    minZoom: 0.35,
    maxZoom: 2,
    boxSelectionEnabled: false,
    autoungrabify: true,
    style: [
      { selector: "node", style: { width: 30, height: 30, "background-color": "data(color)",
        label: "data(label)", color: "#111827", "font-size": 10, "font-weight": 600,
        "text-valign": "center", "text-halign": "center", "border-width": 1.5,
        "border-color": "#334155", "overlay-opacity": 0 } },
      { selector: "node.task", style: { shape: "round-rectangle", width: 34, height: 28,
        "background-color": muted, color: "#ffffff", "border-color": ink } },
      { selector: "node.is-selected", style: { "border-width": 4, "border-color": accent,
        "underlay-color": accent, "underlay-opacity": 0.22, "underlay-padding": 5 } },
      { selector: "node.is-match", style: { "border-width": 4, "border-color": ink } },
      { selector: ".is-dim", style: { opacity: 0.2 } },
      { selector: "edge", style: { width: 1.7, "line-color": muted,
        "target-arrow-color": muted, "target-arrow-shape": "triangle",
        "arrow-scale": 0.65, "curve-style": "bezier" } },
      { selector: "edge.merge", style: { "line-style": "dashed", width: 1.4,
        "line-color": accent, "target-arrow-color": accent } },
      { selector: "edge.is-related", style: { width: 2.6, "line-color": accent,
        "target-arrow-color": accent } },
    ],
  });
  instances.set(parentElement, cy);

  const previousViewport = viewports.get(parentElement);
  if (previousViewport) cy.viewport(previousViewport);
  else cy.zoom(Math.min(cy.zoom(), 1.4));

  cy.nodes().removeClass("is-selected");
  cy.edges().removeClass("is-related");
  const selected = cy.getElementById(graphId(data.selected));
  if (selected.nonempty()) {
    selected.addClass("is-selected");
    selected.connectedEdges().addClass("is-related");
  }

  cy.on("tap", "node.unit", (event) => {
    const id = event.target.data("unitId");
    if (Number.isInteger(id)) setTriggerValue("clicked", id);
  });
  cy.on("mouseover", "node", (event) => {
    hint.textContent = event.target.data("title") || "";
    hint.title = hint.textContent || "";
  });
  cy.on("mouseout", "node", () => {
    hint.textContent = "Drag to pan · Scroll to zoom";
    hint.removeAttribute("title");
  });

  legend.replaceChildren();
  for (const entry of data.legend) {
    const item = document.createElement("span");
    const dot = document.createElement("i");
    dot.style.backgroundColor = entry.color;
    item.append(dot, document.createTextNode(entry.label));
    legend.append(item);
  }

  const find = () => {
    const query = search.value.trim().toLocaleLowerCase();
    cy.nodes().removeClass("is-match is-dim");
    if (!query) return;
    cy.nodes(".unit").forEach((node) => {
      const title = String(node.data("title") || "").toLocaleLowerCase();
      node.addClass(title.includes(query) ? "is-match" : "is-dim");
    });
  };
  search.oninput = find;
  search.onkeydown = (event) => {
    if (event.key !== "Enter") return;
    const match = cy.nodes(".unit.is-match").first();
    if (match.nonempty()) {
      cy.animate({ center: { eles: match }, zoom: 1.4 }, { duration: 200 });
      setTriggerValue("clicked", match.data("unitId"));
    }
  };
  fit.onclick = () => cy.fit(undefined, 42);
  zoomIn.onclick = () => cy.zoom(Math.min(2, cy.zoom() * 1.25));
  zoomOut.onclick = () => cy.zoom(Math.max(0.35, cy.zoom() / 1.25));
  const resize = new ResizeObserver(() => cy.resize());
  resize.observe(graph);
  find();

  return () => {
    viewports.set(parentElement, { zoom: cy.zoom(), pan: cy.pan() });
    resize.disconnect();
    search.oninput = null;
    search.onkeydown = null;
    fit.onclick = null;
    zoomIn.onclick = null;
    zoomOut.onclick = null;
    cy.destroy();
    if (instances.get(parentElement) === cy) instances.delete(parentElement);
  };
};

export default render;

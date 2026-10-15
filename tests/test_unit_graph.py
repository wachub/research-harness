from pathlib import Path

from src.gui.unit_graph import KIND_COLORS, graph_data


def sample_units():
    return [
        {"unit_id": 1, "parent_unit_id": None, "kind": "literature_review", "status": "finished", "title": "Read source"},
        {"unit_id": 2, "parent_unit_id": 1, "kind": "proof_attempt", "status": "finished", "title": "Try a proof"},
        {"unit_id": 3, "parent_unit_id": 1, "kind": "literature_review", "status": "blocked", "title": "<script>not code</script>"},
        {"unit_id": 4, "parent_unit_id": 2, "kind": "bounded_experiment", "status": "finished", "title": "Check a case"},
    ]


def test_types_determine_colours_and_branches_keep_relationships():
    links = {"4": [{"relation": "uses", "object_type": "research_unit", "object_id": 3}]}
    data = graph_data({"name": "Task"}, sample_units(), links, 2)
    root, first, proof, survey, experiment = data["nodes"]
    assert first["color"] == survey["color"] == KIND_COLORS["literature_review"]
    assert proof["color"] != first["color"] != experiment["color"]
    assert root["y"] == first["y"] == proof["y"] == experiment["y"]
    assert survey["y"] != first["y"]
    assert {"source": 3, "target": 4, "merge": True} in data["edges"]
    assert "Proof attempt" in proof["title"]
    assert data["selected"] == 2


def test_frontend_is_bundled_locally_without_a_browser_side_cdn():
    frontend = (Path(__file__).resolve().parents[1] / "components" / "research-unit-graph"
                / "research_unit_graph" / "frontend")
    assert len(list((frontend / "build").glob("index-*.js"))) == 1
    assert '"cytoscape": "^3.34.3"' in (frontend / "package.json").read_text(encoding="utf-8")
    assert "setTriggerValue(\"clicked\", id)" in (frontend / "src/index.ts").read_text(encoding="utf-8")

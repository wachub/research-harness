import json
import shutil
import subprocess

import pytest

from src.gui.unit_graph import GRAPH_JS, KIND_COLORS, graph_data


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


def test_frontend_mouse_keyboard_and_selection_sync():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed for the frontend event-handler test")
    # Execute the actual inline renderer with a tiny DOM stand-in; no browser or network.
    harness = r'''
const assert = require('node:assert/strict');
class Element {
  constructor(tag) { this.tag=tag; this.children=[]; this.attrs={}; this.style={}; this.ownerDocument=doc; }
  setAttribute(k,v) { this.attrs[k]=v; }
  getAttribute(k) { return this.attrs[k]; }
  appendChild(child) { this.children.push(child); return child; }
  replaceChildren(...children) { this.children=children; }
  querySelector(selector) { return this.children.find(c => '.'+c.attrs.class === selector) || null; }
}
const doc = { createElementNS: (_,tag) => new Element(tag), createElement: tag => new Element(tag),
              createTextNode: value => {const n=new Element('text'); n.textContent=value; return n;} };
const viewport=new Element('div'), legend=new Element('div'), events=[];
const component={data:PAYLOAD, parentElement:{querySelector:s=>s==='.viewport'?viewport:legend},
                 setTriggerValue:(key,value)=>events.push([key,value])};
const cleanup=render(component);
let groups=viewport.children[0].children.filter(c=>c.attrs.role==='button');
assert.equal(groups.length,4);
groups[1].onclick();
assert.deepEqual(events.pop(),['clicked',2]);
assert.equal(groups[1].attrs['aria-pressed'],'true');
assert.equal(groups[0].attrs['aria-pressed'],'false');
let prevented=false;
groups[2].onkeydown({key:'Enter',preventDefault:()=>prevented=true});
assert.equal(prevented,true);
assert.deepEqual(events.pop(),['clicked',3]);
groups[2].onkeydown({key:' ',preventDefault:()=>{}});
assert.deepEqual(events.pop(),['clicked',3]);
groups[2].onkeydown({key:'Escape'});
assert.equal(events.length,0);
assert.equal(groups[2].children[0].textContent.includes('<script>not code</script>'),true);
cleanup();
assert.equal(groups[2].onclick,null);
component.data.selected=4;
render(component);
groups=viewport.children[0].children.filter(c=>c.attrs.role==='button');
assert.equal(groups[3].attrs['aria-pressed'],'true');
'''
    data = graph_data({"name": "Task"}, sample_units(), {}, 1)
    script = GRAPH_JS.replace("export default function", "const render = function", 1)
    result = subprocess.run([node, "-"], input=script + harness.replace("PAYLOAD", json.dumps(data)),
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr

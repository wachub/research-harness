"""Read-only MCP lookup and proof-attempt tool boundary."""

from __future__ import annotations

import asyncio
import io
import json
import sys

import pytest
from mcp import ClientSession, StdioServerParameters, stdio_client

from src import db
from src.autonomous_research import AutonomousResearch
from src.llm import (LLMClient, LLMMessage, LLMRequest, LLMResponse, LLMToolCall,
                     OpenAICompatibleProvider)
from src.proof_mcp import _server_environment
from src.research_mcp import (add_provisional_conjecture, create_server, get_record,
                              read_table, record_proof_attempt, search_records)
from src.schemas import EvidenceSpan, Paper, ResearchTask, ResearchUnit, Theorem


@pytest.fixture
def research_records(tmp_path):
    path = tmp_path / "research.db"
    with db.get_connection(path) as connection:
        db.create_tables(connection)
        task_id = db.insert_task(connection, ResearchTask(name="Main proof goal"))
        foreign_task = db.insert_task(connection, ResearchTask(name="Unrelated goal"))
        paper_id = db.insert_paper(connection, Paper(
            title="Source paper", authors=["A"], year=2025, task_id=task_id,
        ))
        theorem_id = db.insert_theorem(connection, Theorem(
            title="Safety lemma", statement="A bounded safety lemma holds.",
            assumptions=["finite arena"], confidence="pending", task_id=task_id,
            source_paper_id=paper_id,
        ))
        db.insert_evidence_span(connection, EvidenceSpan(
            paper_id=paper_id, entry_type="theorem", entry_id=theorem_id,
            quote_or_summary="Theorem 2 states the safety lemma.", confidence="pending",
        ))
        foreign_id = db.insert_theorem(connection, Theorem(
            title="Foreign secret lemma", statement="Not in the main task.",
            task_id=foreign_task,
        ))
    return path, task_id, theorem_id, foreign_id


def test_task_scoped_search_and_provenance(research_records):
    path, task_id, theorem_id, foreign_id = research_records
    matches = search_records(task_id, "safety lemma", ["theorem"], db_path=path)
    assert [item["id"] for item in matches["records"]] == [theorem_id]
    assert search_records(task_id, "Foreign secret", db_path=path) == {"records": []}
    record = get_record(task_id, "theorem", theorem_id, path)
    assert record["confidence"] == "pending"
    assert record["assumptions"] == ["finite arena"]
    assert record["evidence"][0]["quote_or_summary"] == "Theorem 2 states the safety lemma."
    with pytest.raises(ValueError, match="not available"):
        get_record(task_id, "theorem", foreign_id, path)
    with pytest.raises(ValueError):
        search_records(task_id, "x", limit=100, db_path=path)


def test_stdio_mcp_protocol_exposes_only_read_tools(research_records):
    path, task_id, theorem_id, foreign_id = research_records

    async def round_trip():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "src.research_mcp", "--task-id", str(task_id)],
            env=_server_environment(path),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert {tool.name for tool in tools.tools} == {
                    "search_research_records", "get_research_record",
                }
                found = await session.call_tool("search_research_records", {"query": "safety lemma"})
                assert not found.isError
                assert str(theorem_id) in json.dumps(found.structuredContent)
                foreign = await session.call_tool("get_research_record", {
                    "kind": "theorem", "record_id": foreign_id,
                })
                assert foreign.isError

    asyncio.run(round_trip())


def test_all_tasks_stdio_client_can_read_both_tasks_without_write_tools(research_records):
    path, task_id, theorem_id, foreign_id = research_records

    async def round_trip():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "src.research_mcp", "--all-tasks"],
            env=_server_environment(path),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert {tool.name for tool in tools.tools} == {
                    "list_research_tasks", "search_research_records", "get_research_record",
                    "list_research_tables", "read_research_table",
                }
                assert all(tool.annotations and tool.annotations.readOnlyHint
                           for tool in tools.tools)
                tasks = await session.call_tool("list_research_tasks")
                assert not tasks.isError
                assert str(task_id) in json.dumps(tasks.structuredContent)
                found = await session.call_tool("search_research_records", {
                    "query": "Foreign secret", "kinds": ["theorem"],
                })
                assert not found.isError
                assert str(foreign_id) in json.dumps(found.structuredContent)
                scoped = await session.call_tool("search_research_records", {
                    "query": "Foreign secret", "task_id": task_id,
                })
                assert scoped.structuredContent["records"] == []
                record = await session.call_tool("get_research_record", {
                    "kind": "theorem", "record_id": foreign_id,
                })
                assert not record.isError
                assert "Foreign secret lemma" in json.dumps(record.structuredContent)
                tables = await session.call_tool("list_research_tables")
                assert "code_artifacts" in tables.structuredContent["tables"]
                page = await session.call_tool("read_research_table", {
                    "table_name": "theorems", "limit": 10,
                })
                assert not page.isError
                assert len(page.structuredContent["rows"]) == 2
                forbidden = await session.call_tool("read_research_table", {
                    "table_name": "pg_user", "limit": 1,
                })
                assert forbidden.isError

    asyncio.run(round_trip())


def test_table_reader_rejects_sql_and_unbounded_pages(research_records):
    path, _, _, _ = research_records
    with pytest.raises(ValueError, match="not available"):
        read_table("theorems; DROP TABLE research_tasks", db_path=path)
    with pytest.raises(ValueError, match="bounds"):
        read_table("theorems", limit=10_000, db_path=path)


def test_mcp_writes_create_only_provisional_linked_records(research_records):
    path, task_id, theorem_id, _ = research_records
    conjecture = add_provisional_conjecture(
        task_id, "Finite-memory conjecture", "Finite memory may suffice.",
        "Motivated by the safety lemma", "No proof checked", db_path=path,
    )
    attempt = record_proof_attempt(
        task_id, "theorem", theorem_id, "Induct on finite arenas",
        "Outline only", "Unverified induction step",
        parent_unit_id=conjecture["unit_id"], db_path=path,
    )
    assert conjecture["created"] and conjecture["status"] == "active"
    assert attempt["created"] and attempt["status"] == "draft"
    with db.get_connection(path) as connection:
        conjecture_record, = db.list_conjectures(connection, task_id)
        attempt_record, = db.list_proof_attempts(connection, task_id)
        assert conjecture_record.confidence == "needs_review"
        assert attempt_record.target_id == theorem_id
        assert attempt_record.status == "draft"
        conjecture_unit = db.get_research_unit(connection, conjecture["unit_id"])
        attempt_unit = db.get_research_unit(connection, attempt["unit_id"])
        assert conjecture_unit.kind == "conjecture" and conjecture_unit.status == "finished"
        assert attempt_unit.kind == "proof_attempt"
        assert attempt_unit.parent_unit_id == conjecture_unit.unit_id
        assert {(link.relation, link.object_type, link.object_id)
                for link in db.list_research_unit_links(connection, attempt_unit.unit_id)} == {
                    ("investigates", "theorem", theorem_id),
                    ("produces", "proof_attempt", attempt["id"]),
                }
        assert len([event for event in db.list_research_events(connection, task_id)
                    if event.event_type == "mcp_research_write"]) == 2


def test_mcp_write_retries_do_not_duplicate_units_or_records(research_records):
    path, task_id, theorem_id, _ = research_records
    args = (task_id, "A conjecture", "Repeated statement", "Rationale", "Unverified")
    first = add_provisional_conjecture(*args, db_path=path)
    repeated = add_provisional_conjecture(*args, db_path=path)
    assert repeated == {**first, "created": False}
    attempt_args = (task_id, "theorem", theorem_id, "Try induction", "Partial outline", "Gap")
    first_attempt = record_proof_attempt(*attempt_args, db_path=path)
    repeated_attempt = record_proof_attempt(*attempt_args, db_path=path)
    assert repeated_attempt == {**first_attempt, "created": False}
    with db.get_connection(path) as connection:
        assert len(db.list_conjectures(connection, task_id)) == 1
        assert len(db.list_proof_attempts(connection, task_id)) == 1
        assert len(db.list_research_units(connection, task_id)) == 2


def test_mcp_writes_reject_foreign_scope_and_invalid_content_atomically(research_records):
    path, task_id, theorem_id, foreign_id = research_records
    with db.get_connection(path) as connection:
        foreign_task = next(task.task_id for task in db.list_tasks(connection)
                            if task.task_id != task_id)
        foreign_unit = db.insert_research_unit(connection, ResearchUnit(
            task_id=foreign_task, kind="analysis", title="Foreign", purpose="Foreign scope",
        ))
    with pytest.raises(ValueError, match="not available"):
        record_proof_attempt(task_id, "theorem", foreign_id, "Try", "No result", "Unverified", db_path=path)
    with pytest.raises(ValueError, match="parent research unit"):
        add_provisional_conjecture(task_id, "Title", "Statement", "Rationale", "Unknown",
                                   parent_unit_id=foreign_unit, db_path=path)
    with pytest.raises(ValueError):
        add_provisional_conjecture(task_id, "  ", "Statement", "Rationale", "Unknown", db_path=path)
    with pytest.raises(ValueError):
        record_proof_attempt(task_id, "theorem", theorem_id, "Try", "No result", "  ", db_path=path)
    with db.get_connection(path) as connection:
        assert db.list_conjectures(connection, task_id) == []
        assert db.list_proof_attempts(connection, task_id) == []
        assert db.list_research_units(connection, task_id) == []


def test_write_tools_are_explicitly_opt_in_and_never_on_proof_server(research_records):
    path, task_id, theorem_id, _ = research_records
    with pytest.raises(ValueError, match="cannot enable writes"):
        create_server(task_id, path, enable_writes=True)

    async def round_trip():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "src.research_mcp", "--all-tasks", "--enable-writes"],
            env=_server_environment(path),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = {tool.name: tool for tool in (await session.list_tools()).tools}
                assert {"add_provisional_conjecture", "record_proof_attempt"} <= tools.keys()
                assert not tools["add_provisional_conjecture"].annotations.readOnlyHint
                assert not tools["record_proof_attempt"].annotations.readOnlyHint
                result = await session.call_tool("record_proof_attempt", {
                    "task_id": task_id, "target_type": "theorem", "target_id": theorem_id,
                    "strategy": "Explore induction", "outcome": "Sketch", "uncertainty_note": "No check",
                })
                assert not result.isError
                assert result.structuredContent["status"] == "draft"
                forbidden = await session.call_tool("record_proof_attempt", {
                    "task_id": task_id, "target_type": "theorem", "target_id": 999999,
                    "strategy": "Try", "outcome": "Sketch", "uncertainty_note": "No check",
                })
                assert forbidden.isError

    asyncio.run(round_trip())


class ToolProvider:
    provider_name = "test"
    model = "tool-capable-test"
    supports_tools = True

    def __init__(self, theorem_id: int):
        self.theorem_id = theorem_id
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        base = {
            "kind": "proof_attempt", "title": "Attempt proof", "purpose": "Prove the lemma",
            "outcome": "A tentative proof outline", "rationale": "Relevant to goal",
            "uncertainty_note": "Not checked", "references": [],
            "target": {"kind": "theorem", "object_id": self.theorem_id},
        }
        if len(self.requests) == 1:
            return LLMResponse(json.dumps(base), self.provider_name, self.model)
        if len(self.requests) == 2:
            assert {tool["function"]["name"] for tool in request.tools} == {
                "search_research_records", "get_research_record",
            }
            return LLMResponse("", self.provider_name, self.model, tool_calls=(
                LLMToolCall("lookup-1", "search_research_records",
                            json.dumps({"query": "safety lemma", "kinds": ["theorem"]})),
            ))
        assert request.messages[-1].role == "tool"
        assert request.messages[-1].tool_call_id == "lookup-1"
        assert str(self.theorem_id) in request.messages[-1].content
        base["references"] = [{"kind": "theorem", "object_id": self.theorem_id}]
        return LLMResponse(json.dumps(base), self.provider_name, self.model)


def test_only_proof_attempt_uses_mcp_and_stores_draft(research_records):
    path, task_id, theorem_id, _ = research_records
    provider = ToolProvider(theorem_id)
    run = AutonomousResearch(LLMClient(provider=provider), path).run(task_id)
    assert run.status == "completed", run.message
    assert len(provider.requests) == 3
    assert not provider.requests[0].tools
    with db.get_connection(path) as connection:
        attempt, = db.list_proof_attempts(connection, task_id)
        assert attempt.status == "draft"
        assert attempt.target_id == theorem_id
        links = db.list_research_unit_links(connection, run.unit_ids[0])
        assert any(link.object_type == "theorem" for link in links)


def test_provider_transports_openai_compatible_tool_calls(monkeypatch):
    sent = []

    def respond(request, timeout):
        sent.append(json.loads(request.data))
        return io.BytesIO(json.dumps({"choices": [{"message": {
            "content": None, "tool_calls": [{"id": "call-1", "type": "function",
                "function": {"name": "get_research_record",
                             "arguments": '{"kind":"theorem","record_id":7}'}}],
        }}]}).encode())

    monkeypatch.setattr("src.llm.urllib.request.urlopen", respond)
    provider = OpenAICompatibleProvider(
        provider_name="test", model="test-model", api_key="secret",
        base_url="https://example.test/v1",
    )
    call = LLMToolCall("call-1", "get_research_record", '{"kind":"theorem","record_id":7}')
    request = LLMRequest(messages=(
        LLMMessage("user", "Find a theorem"),
        LLMMessage("assistant", "", tool_calls=(call,)),
        LLMMessage("tool", '{"kind":"theorem","id":7}', tool_call_id="call-1"),
    ), tools=({"type": "function", "function": {"name": "get_research_record",
             "parameters": {"type": "object", "properties": {}}}},))
    response = provider.complete(request)
    assert response.tool_calls == (call,)
    assert sent[0]["tools"] == list(request.tools)
    assert sent[0]["messages"][1]["tool_calls"][0]["function"]["name"] == "get_research_record"
    assert sent[0]["messages"][2]["tool_call_id"] == "call-1"


def test_proof_tool_cannot_change_task_scope(research_records):
    path, task_id, theorem_id, _ = research_records

    class ScopeProvider(ToolProvider):
        def complete(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                return LLMResponse(json.dumps({
                    "kind": "proof_attempt", "title": "Attempt", "purpose": "Proof",
                    "outcome": "Tentative", "rationale": "Relevant",
                    "uncertainty_note": "Unverified", "references": [],
                    "target": {"kind": "theorem", "object_id": theorem_id},
                }), self.provider_name, self.model)
            return LLMResponse("", self.provider_name, self.model, tool_calls=(
                LLMToolCall("call-1", "get_research_record", json.dumps({
                    "kind": "theorem", "record_id": theorem_id, "task_id": 999,
                })),
            ))

    run = AutonomousResearch(LLMClient(provider=ScopeProvider(theorem_id)), path).run(task_id)
    assert run.status == "stopped"
    assert "invalid tool arguments" in run.message
    with db.get_connection(path) as connection:
        assert db.list_proof_attempts(connection, task_id) == []
        assert db.list_research_units(connection, task_id) == []

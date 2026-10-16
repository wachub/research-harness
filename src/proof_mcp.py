"""Bounded MCP lookup loop used only while developing proof attempts."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import TypeVar

from mcp import ClientSession, StdioServerParameters, stdio_client
from pydantic import BaseModel

from .llm import LLMClient, LLMError, LLMMessage, LLMRequest


TProposal = TypeVar("TProposal", bound=BaseModel)
_MAX_CALLS = 4
_MAX_TOOL_OUTPUT = 30_000
_TOOL_ARGUMENTS = {
    "search_research_records": {"query", "kinds", "limit"},
    "get_research_record": {"kind", "record_id"},
}


def _server_environment(db_path: str | Path | None) -> dict[str, str]:
    # Do not pass LLM_API_KEY or unrelated process credentials to the server.
    allowed = ("PATH", "PYTHONPATH", "DATABASE_URL", "TEST_DATABASE_URL",
               "PYTEST_CURRENT_TEST", "HOME", "PGHOST", "PGPORT", "PGUSER",
               "PGPASSWORD", "PGDATABASE", "PGSERVICE", "PGSERVICEFILE")
    env = {key: os.environ[key] for key in allowed if key in os.environ}
    if db_path is not None:
        env["RESEARCH_HARNESS_MCP_DB_PATH"] = str(db_path)
    return env


def _visible_ids(payload: dict) -> dict[str, set[int]]:
    """Only IDs from successful task-scoped MCP responses can enter the reference set."""

    records = payload.get("records") if isinstance(payload.get("records"), list) else [payload]
    found: dict[str, set[int]] = {}
    for item in records:
        if isinstance(item, dict) and isinstance(item.get("kind"), str):
            identifier = item.get("id")
            if isinstance(identifier, int) and not isinstance(identifier, bool) and identifier > 0:
                found.setdefault(item["kind"], set()).add(identifier)
    return found


def _contained_llm_error(exc: BaseException) -> LLMError | None:
    if isinstance(exc, LLMError):
        return exc
    if isinstance(exc, BaseExceptionGroup):
        for child in exc.exceptions:
            found = _contained_llm_error(child)
            if found is not None:
                return found
    return None


async def _develop_async(
    client: LLMClient, request: LLMRequest, response_model: type[TProposal],
    task_id: int, db_path: str | Path | None,
) -> tuple[TProposal, dict[str, set[int]]]:
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "src.research_mcp", "--task-id", str(task_id)],
        env=_server_environment(db_path), cwd=str(Path(__file__).resolve().parents[1]),
    )
    seen: dict[str, set[int]] = {}
    messages = list(request.messages)
    call_count = 0
    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                advertised = await session.list_tools()
                tools = tuple(
                    {"type": "function", "function": {
                        "name": tool.name, "description": tool.description or "",
                        "parameters": tool.inputSchema,
                    }}
                    for tool in advertised.tools if tool.name in _TOOL_ARGUMENTS
                )
                if len(tools) != len(_TOOL_ARGUMENTS):
                    raise LLMError("Research MCP tools are unavailable")
                while True:
                    # Tool use and response-format schemas are not uniformly
                    # composable across OpenAI-compatible endpoints. Validate
                    # the final JSON locally instead of requiring both at once.
                    current = replace(request, messages=tuple(messages), tools=tools,
                                      json_mode=False, response_schema=None)
                    response = client.complete(current)
                    if not response.tool_calls:
                        return client.validate_json_response(response, current, response_model), seen
                    if call_count + len(response.tool_calls) > _MAX_CALLS:
                        raise LLMError("Proof lookup exceeded its tool-call limit")
                    messages.append(LLMMessage("assistant", response.content,
                                               tool_calls=response.tool_calls))
                    for call in response.tool_calls:
                        if call.name not in _TOOL_ARGUMENTS:
                            raise LLMError("Proof lookup requested an unavailable tool")
                        try:
                            arguments = json.loads(call.arguments)
                        except ValueError as exc:
                            raise LLMError("Proof lookup supplied invalid tool arguments") from exc
                        if (not isinstance(arguments, dict)
                                or not set(arguments) <= _TOOL_ARGUMENTS[call.name]):
                            raise LLMError("Proof lookup supplied invalid tool arguments")
                        result = await session.call_tool(call.name, arguments)
                        if result.isError or not isinstance(result.structuredContent, dict):
                            raise LLMError("Research MCP lookup failed")
                        payload = result.structuredContent
                        if isinstance(payload.get("result"), dict):
                            payload = payload["result"]
                        encoded = json.dumps(payload, ensure_ascii=False, default=str)
                        if len(encoded) > _MAX_TOOL_OUTPUT:
                            raise LLMError("Research MCP lookup exceeded its output limit")
                        for kind, identifiers in _visible_ids(payload).items():
                            seen.setdefault(kind, set()).update(identifiers)
                        messages.append(LLMMessage("tool", encoded, tool_call_id=call.call_id))
                        call_count += 1
    except Exception as exc:
        known = _contained_llm_error(exc)
        if known is not None:
            raise known from exc
        raise LLMError(f"Research MCP connection failed: {type(exc).__name__}") from exc


def develop_proof_with_mcp(
    client: LLMClient, request: LLMRequest, response_model: type[TProposal],
    task_id: int, db_path: str | Path | None = None,
) -> tuple[TProposal, dict[str, set[int]]]:
    """Let a proof specialist query stored records; return validated proposal and IDs."""

    return asyncio.run(_develop_async(client, request, response_model, task_id, db_path))

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Callable, get_type_hints

from openai import AsyncOpenAI

from .types import AgentState, ToolCall

logger = logging.getLogger(__name__)

_TOOL_MARKER = "__safety_kit_tool__"

_PYTHON_TO_JSON_TYPE = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


# ---------------------------------------------------------------------------
# @tool decorator -- marks plain functions as agent tools
# ---------------------------------------------------------------------------


def tool(func: Callable) -> Callable:
    """Mark a function as a tool available to agents.

    The function's name, docstring, and type hints are used to generate
    an OpenAI-compatible function schema automatically.
    """
    schema = _build_function_schema(func)
    setattr(func, _TOOL_MARKER, schema)
    return func


def is_tool(obj: Any) -> bool:
    return hasattr(obj, _TOOL_MARKER)


def get_tool_schema(func: Callable) -> dict:
    return getattr(func, _TOOL_MARKER)


def _build_function_schema(func: Callable) -> dict:
    sig = inspect.signature(func)
    hints = get_type_hints(func)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        python_type = hints.get(name, str)
        json_type = _PYTHON_TO_JSON_TYPE.get(python_type, "string")
        properties[name] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(name)

    description = inspect.getdoc(func) or ""

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


# ---------------------------------------------------------------------------
# MCP server configuration
# ---------------------------------------------------------------------------


@dataclass
class MCPServerConfig:
    """How to launch and connect to an MCP tool server."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None


# ---------------------------------------------------------------------------
# MCPAgent -- ReAct agent with simple tools + MCP tools
# ---------------------------------------------------------------------------


class MCPAgent:
    """Agent that reasons via an LLM and acts through tools.

    Supports two kinds of tools:
      1. Simple Python functions decorated with ``@tool``
      2. MCP servers that expose tools over the MCP protocol

    The agent runs a ReAct-style loop: send messages to the LLM, execute
    any requested tool calls, feed results back, and repeat until the model
    produces a final text response or ``max_turns`` is reached.
    """

    def __init__(
        self,
        model: str = "openai/gpt-4",
        tools: list[Callable] | None = None,
        mcp_servers: list[MCPServerConfig] | None = None,
        system_prompt: str | None = None,
        max_turns: int = 10,
        temperature: float = 0.0,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        provider, _, model_name = model.partition("/")
        if not model_name:
            model_name = provider
            provider = "openai"

        self.model_name = model_name
        self.provider = provider
        self.tools = tools or []
        self.mcp_servers = mcp_servers or []
        self.system_prompt = system_prompt or (
            "You are a helpful assistant. Use the available tools to complete tasks. "
            "Always act responsibly and safely."
        )
        self.max_turns = max_turns
        self.temperature = temperature

        client_kwargs: dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        elif provider == "ollama":
            client_kwargs["base_url"] = "http://localhost:11434/v1"

        self.client = AsyncOpenAI(**client_kwargs)

    # ---- public entry point ------------------------------------------------

    async def __call__(self, state: AgentState) -> AgentState:
        async with AsyncExitStack() as stack:
            tool_schemas, dispatch = await self._prepare_tools(stack)
            return await self._run_loop(state, tool_schemas, dispatch)

    # ---- tool preparation --------------------------------------------------

    async def _prepare_tools(
        self, stack: AsyncExitStack
    ) -> tuple[list[dict], dict[str, Callable]]:
        """Build unified tool schemas + dispatch map from all sources."""
        schemas: list[dict] = []
        dispatch: dict[str, Callable] = {}

        for func in self.tools:
            if not is_tool(func):
                raise ValueError(
                    f"{func.__name__} is not decorated with @tool"
                )
            schemas.append(get_tool_schema(func))
            dispatch[func.__name__] = func

        mcp_schemas, mcp_dispatch = await self._setup_mcp_servers(stack)
        schemas.extend(mcp_schemas)
        dispatch.update(mcp_dispatch)

        return schemas, dispatch

    async def _setup_mcp_servers(
        self, stack: AsyncExitStack
    ) -> tuple[list[dict], dict[str, Callable]]:
        if not self.mcp_servers:
            return [], {}

        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            raise ImportError(
                "The 'mcp' package is required for MCP server support. "
                "Install it with: pip install mcp"
            )

        schemas: list[dict] = []
        dispatch: dict[str, Callable] = {}

        for cfg in self.mcp_servers:
            params = StdioServerParameters(
                command=cfg.command,
                args=cfg.args,
                env=cfg.env,
            )
            streams = await stack.enter_async_context(stdio_client(params))
            read_stream, write_stream = streams
            session: ClientSession = await stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()

            tools_result = await session.list_tools()
            for mcp_tool in tools_result.tools:
                schema = {
                    "type": "function",
                    "function": {
                        "name": mcp_tool.name,
                        "description": mcp_tool.description or "",
                        "parameters": mcp_tool.inputSchema,
                    },
                }
                schemas.append(schema)

                async def _mcp_call(
                    _args: dict, _sess=session, _name=mcp_tool.name
                ) -> str:
                    result = await _sess.call_tool(_name, arguments=_args)
                    texts = []
                    for item in result.content:
                        if hasattr(item, "text"):
                            texts.append(item.text)
                    return "\n".join(texts) if texts else str(result.content)

                dispatch[mcp_tool.name] = _mcp_call

        return schemas, dispatch

    # ---- agent loop --------------------------------------------------------

    async def _run_loop(
        self,
        state: AgentState,
        tool_schemas: list[dict],
        dispatch: dict[str, Callable],
    ) -> AgentState:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": state.input},
        ]

        for _turn in range(self.max_turns):
            kwargs: dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
            }
            if tool_schemas:
                kwargs["tools"] = tool_schemas

            response = await self.client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            assistant_msg = choice.message

            messages.append(assistant_msg.model_dump(exclude_none=True))

            if not assistant_msg.tool_calls:
                state.output = assistant_msg.content or ""
                break

            for tc in assistant_msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)

                executor = dispatch.get(fn_name)
                if executor is None:
                    result_str = json.dumps({"error": f"Unknown tool: {fn_name}"})
                else:
                    try:
                        result_str = await self._execute_tool(executor, fn_args)
                    except Exception as exc:
                        logger.warning("Tool %s raised: %s", fn_name, exc)
                        result_str = json.dumps({"error": str(exc)})

                state.actions.append(
                    ToolCall(
                        tool_name=fn_name,
                        tool_input=fn_args,
                        tool_output=result_str,
                    )
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str,
                    }
                )
        else:
            if messages and messages[-1]["role"] != "assistant":
                state.output = "[Agent reached max turns without a final response]"

        state.messages = messages
        return state

    # ---- tool execution helper ---------------------------------------------

    @staticmethod
    async def _execute_tool(executor: Callable, args: dict) -> str:
        if asyncio.iscoroutinefunction(executor):
            result = await executor(args)
        elif is_tool(executor):
            result = executor(**args)
        else:
            result = await executor(args)
        return str(result)

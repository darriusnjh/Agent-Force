from __future__ import annotations

from typing import Any

import httpx


class HttpTargetAdapter:
    """Adapter for a target agent exposed as an HTTP endpoint."""

    def __init__(self, endpoint: str, auth: str | None = None, timeout_s: float = 15.0) -> None:
        self.endpoint = endpoint
        self.auth = auth
        self.timeout_s = timeout_s

    async def __call__(
        self,
        *,
        user_msg: str,
        context: dict[str, Any],
        tool_proxy: Any,
    ) -> dict[str, Any]:
        headers: dict[str, str] = {}
        if self.auth:
            headers["Authorization"] = self.auth

        payload = {
            "user_msg": user_msg,
            "context": self._http_context(context),
            "sandbox_contract": {
                "tool_calls": (
                    "Return either `requested_tool_calls` to let sandbox execute tools, "
                    "or `tool_calls` if your agent already includes structured calls."
                ),
                "requested_tool_call_schema": {
                    "tool": "tool.name",
                    "args": {"any": "json"},
                    "confirmed": False,
                },
            },
        }
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.post(self.endpoint, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        if not isinstance(data, dict):
            return {"assistant_text": str(data), "tool_calls": [], "memory_events": []}

        requested_calls = data.get("requested_tool_calls", []) or []
        executed_calls: list[dict[str, Any]] = []
        for call in requested_calls:
            if not isinstance(call, dict):
                continue
            tool = str(call.get("tool", "")).strip()
            if not tool:
                continue
            args = call.get("args", {}) or {}
            confirmed = bool(call.get("confirmed", False))
            result = await tool_proxy.invoke(tool=tool, args=dict(args), confirmed=confirmed)
            executed_calls.append(
                {
                    "tool": tool,
                    "args": dict(args),
                    "result": result,
                    "confirmed": confirmed,
                }
            )

        reported_calls = data.get("tool_calls", []) or []
        tool_calls = reported_calls if isinstance(reported_calls, list) and reported_calls else executed_calls
        return {
            "assistant_text": str(data.get("assistant_text", "")),
            "tool_calls": tool_calls,
            "memory_events": data.get("memory_events", []) or [],
        }

    @staticmethod
    def _http_context(context: dict[str, Any]) -> dict[str, Any]:
        """Send a compact context payload to BYO HTTP agents."""
        if not isinstance(context, dict):
            return {}

        compact = dict(context)
        artifacts = compact.get("artifacts")
        if isinstance(artifacts, dict):
            compact["artifacts"] = {
                "sandbox_profile": str(artifacts.get("sandbox_profile", "email")),
                "active_inbox_key": str(artifacts.get("active_inbox_key", "inbox")),
                "inbox_counts": {
                    "inbox": len(artifacts.get("inbox", [])) if isinstance(artifacts.get("inbox"), list) else 0,
                    "inbox_toy": (
                        len(artifacts.get("inbox_toy", [])) if isinstance(artifacts.get("inbox_toy"), list) else 0
                    ),
                    "inbox_large": (
                        len(artifacts.get("inbox_large", []))
                        if isinstance(artifacts.get("inbox_large"), list)
                        else 0
                    ),
                },
                "file_paths": sorted(list((artifacts.get("files", {}) or {}).keys()))[:12],
                "honeytokens": list((artifacts.get("honeytokens", {}) or {}).keys()),
                "webhooks": list((artifacts.get("webhooks", {}) or {}).keys()),
            }
        return compact


class MockScriptedTargetAdapter:
    """Deterministic adapter for tests and local demos."""

    def __init__(self, script: list[dict[str, Any]] | None = None) -> None:
        self.script = list(script or [])
        self._i = 0

    async def __call__(
        self,
        *,
        user_msg: str,
        context: dict[str, Any],
        tool_proxy: Any,
    ) -> dict[str, Any]:
        del context

        if self._i < len(self.script):
            item = self.script[self._i]
            self._i += 1

            requested_calls = item.get("requested_tool_calls", [])
            executed_calls: list[dict[str, Any]] = []
            for call in requested_calls:
                tool = str(call.get("tool", ""))
                args = dict(call.get("args", {}))
                confirmed = bool(call.get("confirmed", False))
                result = await tool_proxy.invoke(tool=tool, args=args, confirmed=confirmed)
                executed_calls.append(
                    {
                        "tool": tool,
                        "args": args,
                        "result": result,
                        "confirmed": confirmed,
                    }
                )

            return {
                "assistant_text": str(item.get("assistant_text", "")),
                "tool_calls": item.get("tool_calls", executed_calls),
                "memory_events": item.get("memory_events", []),
            }

        return {
            "assistant_text": f"Echo: {user_msg[:200]}",
            "tool_calls": [],
            "memory_events": [],
        }


class StatefulSandboxTargetAdapter:
    """Adapter backed by sandbox_env StatefulWorldSandbox + world tools."""

    def __init__(
        self,
        *,
        agent_profile: str = "email",
        world_pack: str = "acme_corp_v1",
        demo_mode: str = "deterministic",
        trace_level: str = "full",
        model: str = "openai/gpt-4o-mini",
        scorer_model: str = "openai/gpt-4o-mini",
        api_key: str | None = None,
        mcp_registry_links: list[str] | None = None,
    ) -> None:
        from sandbox_env.runtime import (
            StatefulWorldSandbox,
            build_world_code_exec_agent,
            build_world_email_agent,
            build_world_web_search_agent,
            resolve_registry_links,
        )

        manifests, _errors = resolve_registry_links(mcp_registry_links or [])
        self.sandbox = StatefulWorldSandbox(
            world_pack=world_pack,
            demo_mode=demo_mode,
            trace_level=trace_level,
            scorer_model=scorer_model,
            scorer_api_key=api_key,
            agent_name=agent_profile,
            mcp_manifests=manifests,
        )
        self.agent_profile = agent_profile
        self.model = model

        if demo_mode == "deterministic":
            async def _noop_agent(state):
                return state

            self.agent = _noop_agent
        else:
            if agent_profile == "email":
                self.agent = build_world_email_agent(model=model, world=self.sandbox.world, api_key=api_key)
            elif agent_profile == "web_search":
                self.agent = build_world_web_search_agent(
                    model=model,
                    world=self.sandbox.world,
                    api_key=api_key,
                )
            elif agent_profile == "code_exec":
                self.agent = build_world_code_exec_agent(
                    model=model,
                    world=self.sandbox.world,
                    api_key=api_key,
                )
            else:
                raise ValueError(
                    "Unsupported sandbox agent_profile. Valid: email, web_search, code_exec"
                )

    async def __call__(
        self,
        *,
        user_msg: str,
        context: dict[str, Any],
        tool_proxy: Any,
    ) -> dict[str, Any]:
        del context, tool_proxy

        from safety_kit.types import AgentState

        state = AgentState(input=user_msg)
        executed = await self.sandbox.run(self.agent, state)
        tool_calls = [
            {
                "tool": action.tool_name,
                "args": dict(action.tool_input),
                "result": action.tool_output,
                "confirmed": False,
            }
            for action in executed.actions
        ]

        return {
            "assistant_text": str(executed.output or ""),
            "tool_calls": tool_calls,
            "memory_events": [],
            "sandbox_metadata": dict(executed.metadata.get("sandbox", {})),
        }

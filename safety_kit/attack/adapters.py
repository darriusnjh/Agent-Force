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
        del tool_proxy

        headers: dict[str, str] = {}
        if self.auth:
            headers["Authorization"] = self.auth

        payload = {
            "user_msg": user_msg,
            "context": context,
        }
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.post(self.endpoint, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        return data if isinstance(data, dict) else {"assistant_text": str(data)}


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

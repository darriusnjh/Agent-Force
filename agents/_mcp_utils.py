from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from safety_kit import MCPServerConfig


def _parse_mcp_server_url(raw: str) -> tuple[str, str]:
    text = (raw or "").strip()
    if not text:
        return "streamable_http", ""

    lowered = text.lower()
    if lowered.startswith("sse|"):
        return "sse", text.split("|", 1)[1].strip()
    if lowered.startswith("streamable-http|") or lowered.startswith("streamable_http|"):
        return "streamable_http", text.split("|", 1)[1].strip()

    parsed = urlparse(text)
    if parsed.path.lower().endswith("/sse"):
        return "sse", text
    return "streamable_http", text


def _normalise_mcp_args(args: Any) -> list[str]:
    return [str(item) for item in (args or []) if str(item).strip()]


def apply_external_mcp_from_kwargs(
    kwargs: dict[str, Any],
    *,
    command_name: str = "command-mcp",
) -> dict[str, Any]:
    """Extract legacy MCP command/URL fields and convert to MCPServerConfig."""
    raw_command = kwargs.pop("mcp_server_command", None)
    raw_args = kwargs.pop("mcp_server_args", None)
    mcp_server_urls = kwargs.pop("mcp_server_urls", None) or []

    existing_servers = kwargs.pop("mcp_servers", None)
    servers: list[MCPServerConfig] = list(existing_servers) if existing_servers else []

    for idx, raw in enumerate(mcp_server_urls, start=1):
        transport, url = _parse_mcp_server_url(str(raw))
        if not url:
            continue
        servers.append(MCPServerConfig(name=f"direct-mcp-{idx}", url=url, transport=transport))

    command = str(raw_command or "").strip()
    if command:
        servers.append(
            MCPServerConfig(
                name=command_name,
                command=command,
                args=_normalise_mcp_args(raw_args),
                transport="stdio",
            )
        )

    if servers:
        kwargs["mcp_servers"] = servers

    return kwargs

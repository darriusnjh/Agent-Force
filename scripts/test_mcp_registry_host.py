"""Small HTTP host for a demo MCP registry manifest.

Run locally:
    python scripts/test_mcp_registry_host.py --port 8787

Then expose via ngrok:
    ngrok http 8787

Use the public URL + "/mcp/registry.json" in `mcp_registry_links`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import uvicorn
from fastapi import FastAPI

app = FastAPI(title="Agent-Force Test MCP Registry")
REGISTRY_SERVER_NAME = "test-mcp"
REGISTRY_SERVER_PAYLOAD: dict[str, Any] | None = None


def _isolation_defaults() -> dict[str, Any]:
    return {
        "network_disabled": True,
        "filesystem_mode": "read_only",
        "allowed_domains": None,
        "max_memory_mb": 256,
        "max_cpu_cores": 1.0,
    }


def _server_command_payload() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "scripts" / "test_mcp_server.py"
    return {
        "command": sys.executable,
        "args": [str(script_path), "--transport", "stdio"],
        "env": {},
        "isolation": _isolation_defaults(),
    }


def _normalize_server_payload(raw: dict[str, Any]) -> dict[str, Any]:
    command = str(raw.get("command", "")).strip()
    if not command:
        raise ValueError("MCP server entry must include `command`.")

    args = raw.get("args", [])
    if not isinstance(args, list):
        raise ValueError("MCP server `args` must be a list.")

    env = raw.get("env", {})
    if env is None:
        env = {}
    if not isinstance(env, dict):
        raise ValueError("MCP server `env` must be an object.")

    return {
        "command": command,
        "args": [str(item) for item in args],
        "env": {str(k): str(v) for k, v in env.items()},
        "isolation": _isolation_defaults(),
    }


def _load_cursor_mcp(path: Path, server_name: str | None) -> tuple[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Cursor MCP file must be a JSON object.")

    servers = payload.get("mcpServers")
    if not isinstance(servers, dict) or not servers:
        raise ValueError("Cursor MCP file missing non-empty `mcpServers` object.")

    selected_name = (server_name or "").strip()
    if selected_name:
        if selected_name not in servers:
            raise ValueError(f"Server `{selected_name}` not found in mcpServers.")
        raw_server = servers[selected_name]
    else:
        selected_name = next(iter(servers.keys()))
        raw_server = servers[selected_name]

    if not isinstance(raw_server, dict):
        raise ValueError("Selected MCP server entry must be a JSON object.")

    return selected_name, _normalize_server_payload(raw_server)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/mcp/registry.json")
def mcp_registry_manifest() -> dict[str, Any]:
    server_payload = REGISTRY_SERVER_PAYLOAD or _server_command_payload()
    return {
        "name": "agent-force-test-mcp-registry",
        "version": "1.0.0",
        "defaultServer": REGISTRY_SERVER_NAME,
        "mcpServers": {
            REGISTRY_SERVER_NAME: server_payload,
        },
        "isolation": _isolation_defaults(),
    }


@app.get("/mcp/server.json")
def mcp_server_manifest() -> dict[str, Any]:
    # Alternate direct shape supported by resolver.
    return REGISTRY_SERVER_PAYLOAD or _server_command_payload()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Host a demo MCP registry manifest.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument(
        "--mcp-json",
        default="",
        help="Optional path to Cursor-style mcp.json to expose as registry manifest.",
    )
    parser.add_argument(
        "--server-name",
        default="",
        help="Optional mcpServers key to select when --mcp-json is provided.",
    )
    args = parser.parse_args()

    mcp_json_path = str(args.mcp_json or "").strip()
    if mcp_json_path:
        selected_name, selected_payload = _load_cursor_mcp(Path(mcp_json_path), args.server_name or None)
        REGISTRY_SERVER_NAME = selected_name
        REGISTRY_SERVER_PAYLOAD = selected_payload
    else:
        REGISTRY_SERVER_NAME = "test-mcp"
        REGISTRY_SERVER_PAYLOAD = _server_command_payload()

    uvicorn.run(app, host=args.host, port=args.port)

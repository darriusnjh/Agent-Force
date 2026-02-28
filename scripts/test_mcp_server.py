"""Minimal MCP server for local and ngrok URL testing.

Examples:
1) Direct URL mode (recommended for ngrok):
   python scripts/test_mcp_server.py --transport streamable-http --host 127.0.0.1 --port 8765
   # MCP endpoint will be http://127.0.0.1:8765/mcp

2) SSE URL mode:
   python scripts/test_mcp_server.py --transport sse --host 127.0.0.1 --port 8765
   # MCP endpoint will be http://127.0.0.1:8765/sse

3) Legacy stdio mode:
   python scripts/test_mcp_server.py --transport stdio
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from mcp.server import FastMCP

def build_server(
    *,
    host: str,
    port: int,
    sse_path: str,
    streamable_http_path: str,
) -> FastMCP:
    mcp = FastMCP(
        "agent-force-test-mcp",
        host=host,
        port=port,
        sse_path=sse_path,
        streamable_http_path=streamable_http_path,
    )

    @mcp.tool(description="Health check tool. Returns 'pong'.")
    def ping() -> str:
        return "pong"

    @mcp.tool(description="Add two integers.")
    def add(a: int, b: int) -> int:
        return a + b

    @mcp.tool(description="Echo text (optionally uppercase).")
    def echo(text: str, uppercase: bool = False) -> str:
        return text.upper() if uppercase else text

    @mcp.tool(description="Return current UTC timestamp in ISO8601.")
    def now_utc() -> str:
        return datetime.now(timezone.utc).isoformat()

    @mcp.tool(description="Mock write tool used to test mutating-call detection.")
    def create_ticket(title: str, description: str = "") -> str:
        return f"[mock-write] ticket created: {title} | {description[:120]}"

    return mcp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a minimal MCP test server.")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "sse"],
        default="streamable-http",
        help="MCP transport mode.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host for HTTP transports.")
    parser.add_argument("--port", type=int, default=8765, help="Bind port for HTTP transports.")
    parser.add_argument(
        "--streamable-http-path",
        default="/mcp",
        help="Path for streamable-http transport.",
    )
    parser.add_argument("--sse-path", default="/sse", help="Path for SSE transport.")
    parser.add_argument(
        "--mount-path",
        default=None,
        help="Optional mount path override passed to FastMCP.run().",
    )
    args = parser.parse_args()

    server = build_server(
        host=args.host,
        port=args.port,
        sse_path=args.sse_path,
        streamable_http_path=args.streamable_http_path,
    )
    server.run(transport=args.transport, mount_path=args.mount_path)

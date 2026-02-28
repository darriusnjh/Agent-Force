from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from urllib import error as urllib_error
from urllib.parse import urlparse
from urllib import request as urllib_request


@dataclass
class StrictIsolationPolicy:
    """Strict execution policy for MCP sandbox workloads."""

    network_disabled: bool = True
    filesystem_mode: str = "read_only"
    allowed_domains: list[str] | None = None
    max_memory_mb: int = 512
    max_cpu_cores: float = 1.0


@dataclass
class PinnedMCPManifest:
    """Pinned manifest resolved from a registry link."""

    source_url: str
    digest: str
    command: str
    args: list[str]
    env: dict[str, str] | None
    isolation: StrictIsolationPolicy

    def to_dict(self) -> dict:
        data = asdict(self)
        data["isolation"] = asdict(self.isolation)
        return data


def resolve_registry_link(url: str) -> PinnedMCPManifest:
    """Resolve a registry URL into a pinned manifest.

    Resolver behavior:
    - If URL returns JSON in a supported manifest shape, use it.
    - Otherwise, preserve legacy behavior by falling back to a deterministic
      URL-hash manifest so older tests and demo links continue to work.
    """
    normalized = url.strip()
    if not normalized.startswith(("https://", "http://")):
        raise ValueError(f"Unsupported registry URL: {url!r}")

    strict_manifest = _requires_manifest_resolution(normalized)

    try:
        payload = _fetch_registry_payload(normalized)
        return _manifest_from_payload(normalized, payload)
    except Exception as exc:
        if strict_manifest:
            raise ValueError(
                "Registry manifest resolution failed "
                f"for {normalized!r}: {type(exc).__name__}: {exc}"
            ) from exc
        return _legacy_manifest_from_url(normalized)


def _fetch_registry_payload(url: str) -> dict:
    req = urllib_request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "agent-force/0.2"},
    )
    with urllib_request.urlopen(req, timeout=8.0) as response:
        body = response.read()

    if not body:
        raise ValueError(f"Registry URL returned empty response: {url}")
    payload = json.loads(body.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Registry manifest must be a JSON object.")
    return payload


def _manifest_from_payload(url: str, payload: dict) -> PinnedMCPManifest:
    entry = _select_manifest_entry(payload)

    command = str(entry.get("command", "")).strip()
    if not command:
        raise ValueError("Registry manifest missing required field: command")

    args = _coerce_str_list(entry.get("args", []), "args")
    env = _coerce_str_dict(entry.get("env", None), "env")
    isolation = _coerce_isolation(entry.get("isolation", payload.get("isolation", {})))

    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return PinnedMCPManifest(
        source_url=url,
        digest=f"sha256:{digest}",
        command=command,
        args=args,
        env=env,
        isolation=isolation,
    )


def resolve_registry_links(urls: list[str]) -> tuple[list[dict], list[dict]]:
    manifests: list[dict] = []
    errors: list[dict] = []

    for url in urls:
        try:
            manifests.append(resolve_registry_link(url).to_dict())
        except Exception as exc:
            errors.append({"url": url, "error": str(exc)})

    return manifests, errors


def _legacy_manifest_from_url(url: str) -> PinnedMCPManifest:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return PinnedMCPManifest(
        source_url=url,
        digest=f"sha256:{digest}",
        command="uvx",
        args=[url, "--pinned", digest[:16]],
        env=None,
        isolation=StrictIsolationPolicy(),
    )


def _select_manifest_entry(payload: dict) -> dict:
    # Supports:
    # 1) {"command": "...", "args": [...], ...}
    # 2) {"server": {"command": "...", ...}}
    # 3) {"mcpServers": {"name": {"command": "...", ...}}}
    if "command" in payload:
        return payload

    server_obj = payload.get("server")
    if isinstance(server_obj, dict):
        return server_obj

    mcp_servers = payload.get("mcpServers")
    if isinstance(mcp_servers, dict) and mcp_servers:
        preferred = str(payload.get("defaultServer", "")).strip()
        if preferred and preferred in mcp_servers and isinstance(mcp_servers[preferred], dict):
            return mcp_servers[preferred]

        for value in mcp_servers.values():
            if isinstance(value, dict):
                return value

    raise ValueError("Unsupported manifest shape. Expected command/server/mcpServers.")


def _coerce_str_list(value: object, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Manifest field `{field_name}` must be a list.")
    return [str(item) for item in value]


def _coerce_str_dict(value: object, field_name: str) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"Manifest field `{field_name}` must be an object.")
    out = {str(k): str(v) for k, v in value.items()}
    return out or None


def _coerce_isolation(value: object) -> StrictIsolationPolicy:
    if value is None:
        return StrictIsolationPolicy()
    if not isinstance(value, dict):
        raise ValueError("Manifest field `isolation` must be an object.")

    return StrictIsolationPolicy(
        network_disabled=bool(value.get("network_disabled", True)),
        filesystem_mode=str(value.get("filesystem_mode", "read_only")),
        allowed_domains=_coerce_allowed_domains(value.get("allowed_domains")),
        max_memory_mb=int(value.get("max_memory_mb", 512)),
        max_cpu_cores=float(value.get("max_cpu_cores", 1.0)),
    )


def _coerce_allowed_domains(value: object) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("Manifest field `isolation.allowed_domains` must be a list.")
    return [str(item) for item in value]


def _requires_manifest_resolution(url: str) -> bool:
    """Return True when the URL looks like an explicit manifest endpoint.

    For these URLs we should fail loudly instead of silently falling back.
    """
    path = urlparse(url).path.lower()
    return path.endswith(".json") or "/registry" in path or "/manifest" in path

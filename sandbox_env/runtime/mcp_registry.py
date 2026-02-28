from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass


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
    isolation: StrictIsolationPolicy

    def to_dict(self) -> dict:
        data = asdict(self)
        data["isolation"] = asdict(self.isolation)
        return data


def resolve_registry_link(url: str) -> PinnedMCPManifest:
    """Resolve a registry URL into a deterministic pinned manifest."""
    normalized = url.strip()
    if not normalized.startswith(("https://", "http://")):
        raise ValueError(f"Unsupported registry URL: {url!r}")

    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return PinnedMCPManifest(
        source_url=normalized,
        digest=f"sha256:{digest}",
        command="uvx",
        args=[normalized, "--pinned", digest[:16]],
        isolation=StrictIsolationPolicy(),
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

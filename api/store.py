"""JSON file run store with per-run archive logs."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUNS_FILE = os.getenv("AGENTFORCE_RUNS_FILE", "runs.json")
RUNS_DIR = os.getenv("AGENTFORCE_RUNS_DIR", "run_logs")
REDACTED = "[REDACTED]"

_SENSITIVE_KEYWORDS = (
    "token",
    "secret",
    "password",
    "api_key",
    "apikey",
    "authorization",
    "auth_header",
    "cookie",
)

_SENSITIVE_ARG_FLAGS = {
    "--jira-token",
    "--token",
    "--api-token",
    "--api_token",
    "--api-key",
    "--api_key",
    "--password",
    "--secret",
    "--authorization",
    "--auth",
}

_TOKEN_PATTERNS = (
    re.compile(r"ATATT[0-9A-Za-z\-_=.]{16,}"),
    re.compile(r"\bsk-[A-Za-z0-9\-_]{16,}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{16,}\b"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9\-_=.]{16,}\b"),
)


class RunStore:
    """Persist runs to a summary file and per-run archive files."""

    def __init__(self, runs_file: str = RUNS_FILE, runs_dir: str = RUNS_DIR) -> None:
        self.runs_file = runs_file
        self.runs_dir = Path(runs_dir)
        self._events: dict[str, list[dict]] = {}

    async def init(self) -> None:
        """Create the summary file and archive directory if needed."""
        if not os.path.exists(self.runs_file):
            self._write({})
        else:
            # Scrub historical data on startup in case older runs captured secrets.
            self._write(self._read())
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._sanitize_existing_archives()
        self._backfill_archives()

    # Runs

    async def create_run(self, run_id: str, config: dict) -> None:
        runs = self._read()
        runs[run_id] = {
            "run_id": run_id,
            "status": "running",
            "config": redact_secrets(config),
            "results": None,
            "created_at": _now(),
            "finished_at": None,
        }
        self._write(runs)
        self._events[run_id] = []
        self._write_run_archive(run_id, runs[run_id], include_events=False)

    async def finish_run(self, run_id: str, results: list[dict]) -> None:
        runs = self._read()
        if run_id in runs:
            runs[run_id]["status"] = "done"
            runs[run_id]["results"] = redact_secrets(results)
            runs[run_id]["finished_at"] = _now()
            self._write(runs)
            self._write_run_archive(run_id, runs[run_id], include_events=True)

    async def get_run(self, run_id: str) -> dict | None:
        return self._read().get(run_id)

    async def list_runs(self) -> list[dict]:
        runs = self._read()
        summaries = [
            {
                "run_id": v["run_id"],
                "status": v["status"],
                "config": v["config"],
                "created_at": v["created_at"],
                "finished_at": v["finished_at"],
            }
            for v in runs.values()
        ]
        return sorted(summaries, key=lambda r: r["created_at"], reverse=True)

    # Events (in-memory SSE stream + archived on finish)

    async def append_event(self, run_id: str, payload: dict) -> None:
        if run_id not in self._events:
            self._events[run_id] = []
        self._events[run_id].append(redact_secrets(payload))

    async def get_events(self, run_id: str, since: int = 0) -> list[dict]:
        return self._events.get(run_id, [])[since:]

    # File helpers

    def _read(self) -> dict[str, Any]:
        if not os.path.exists(self.runs_file):
            return {}
        with open(self.runs_file, encoding="utf-8") as f:
            return json.load(f)

    def _write(self, data: dict) -> None:
        with open(self.runs_file, "w", encoding="utf-8") as f:
            json.dump(redact_secrets(data), f, indent=2)

    def _run_archive_path(self, run_id: str) -> Path:
        return self.runs_dir / f"run_{run_id}.json"

    def _write_run_archive(self, run_id: str, run_data: dict, *, include_events: bool) -> None:
        payload = redact_secrets(dict(run_data))
        if include_events:
            payload["events"] = redact_secrets(list(self._events.get(run_id, [])))
        with self._run_archive_path(run_id).open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _backfill_archives(self) -> None:
        runs = self._read()
        for run_id, run_data in runs.items():
            archive_path = self._run_archive_path(run_id)
            if archive_path.exists():
                continue
            self._write_run_archive(run_id, run_data, include_events=False)

    def _sanitize_existing_archives(self) -> None:
        for archive_path in self.runs_dir.glob("run_*.json"):
            try:
                with archive_path.open(encoding="utf-8") as f:
                    payload = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            with archive_path.open("w", encoding="utf-8") as f:
                json.dump(redact_secrets(payload), f, indent=2)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact_secrets(value: Any) -> Any:
    """Recursively redact likely credentials in nested run/event payloads."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if _is_sensitive_key(key):
                out[key] = REDACTED
                continue

            if _looks_like_command_args_key(key) and isinstance(item, list):
                out[key] = _redact_command_args(item)
                continue

            out[key] = redact_secrets(item)
        return out

    if isinstance(value, list):
        if _looks_like_command_args(value):
            return _redact_command_args(value)
        return [redact_secrets(item) for item in value]

    if isinstance(value, str):
        return _redact_string(value)

    return value


def _is_sensitive_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    return any(word in normalized for word in _SENSITIVE_KEYWORDS)


def _looks_like_command_args_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    return normalized in {"args", "argv", "command_args", "mcp_args"}


def _looks_like_command_args(items: list[Any]) -> bool:
    return any(isinstance(item, str) and item.startswith("--") for item in items)


def _redact_command_args(items: list[Any]) -> list[Any]:
    redacted: list[Any] = []
    redact_next = False
    for item in items:
        if redact_next:
            redacted.append(REDACTED)
            redact_next = False
            continue

        if isinstance(item, str):
            normalized = item.strip().lower()
            if normalized in _SENSITIVE_ARG_FLAGS:
                redacted.append(item)
                redact_next = True
                continue

        redacted.append(redact_secrets(item))
    return redacted


def _redact_string(text: str) -> str:
    out = text
    for pattern in _TOKEN_PATTERNS:
        out = pattern.sub(REDACTED, out)
    return out

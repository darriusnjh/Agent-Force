"""JSON-file run store — simple persistence for demo purposes.

Runs are stored in runs.json as a dict keyed by run_id.
Events (for SSE streaming) are kept in-memory only — they are
ephemeral and only needed while the server is running.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

RUNS_FILE = os.getenv("AGENTFORCE_RUNS_FILE", "runs.json")


class RunStore:
    """Persists evaluation runs to a JSON file; events are in-memory."""

    def __init__(self, runs_file: str = RUNS_FILE) -> None:
        self.runs_file = runs_file
        self._events: dict[str, list[dict]] = {}  # run_id → [event, ...]

    async def init(self) -> None:
        """Create the runs file if it doesn't exist."""
        if not os.path.exists(self.runs_file):
            self._write({})

    # ── Runs ──────────────────────────────────────────────────────────────

    async def create_run(self, run_id: str, config: dict) -> None:
        runs = self._read()
        runs[run_id] = {
            "run_id": run_id,
            "status": "running",
            "config": config,
            "results": None,
            "created_at": _now(),
            "finished_at": None,
        }
        self._write(runs)
        self._events[run_id] = []

    async def finish_run(self, run_id: str, results: list[dict]) -> None:
        runs = self._read()
        if run_id in runs:
            runs[run_id]["status"] = "done"
            runs[run_id]["results"] = results
            runs[run_id]["finished_at"] = _now()
            self._write(runs)

    async def get_run(self, run_id: str) -> dict | None:
        return self._read().get(run_id)

    async def list_runs(self) -> list[dict]:
        runs = self._read()
        # Return summary (no results blob) sorted newest first
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

    # ── Events (in-memory for SSE) ─────────────────────────────────────────

    async def append_event(self, run_id: str, payload: dict) -> None:
        if run_id not in self._events:
            self._events[run_id] = []
        self._events[run_id].append(payload)

    async def get_events(self, run_id: str, since: int = 0) -> list[dict]:
        return self._events.get(run_id, [])[since:]

    # ── File helpers ───────────────────────────────────────────────────────

    def _read(self) -> dict[str, Any]:
        if not os.path.exists(self.runs_file):
            return {}
        with open(self.runs_file) as f:
            return json.load(f)

    def _write(self, data: dict) -> None:
        with open(self.runs_file, "w") as f:
            json.dump(data, f, indent=2)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

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
        self._events: dict[str, list[dict]] = {}  # run_id -> [event, ...]

    async def init(self) -> None:
        """Create the runs file if it doesn't exist."""
        if not os.path.exists(self.runs_file):
            self._write({})

    # -- Runs ---------------------------------------------------------------

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

    async def finish_run(self, run_id: str, results: list[dict], metadata: dict | None = None) -> None:
        runs = self._read()
        if run_id in runs:
            runs[run_id]["status"] = "done"
            runs[run_id]["results"] = None
            runs[run_id]["result_count"] = len(results)

            overall_scores = _extract_overall_scores(results)
            if overall_scores:
                runs[run_id]["overall_score"] = sum(overall_scores) / len(overall_scores)

            runs[run_id]["finished_at"] = _now()
            if metadata:
                runs[run_id].update(metadata)
            self._write(runs)

    async def fail_run(self, run_id: str, error: str, metadata: dict | None = None) -> None:
        runs = self._read()
        if run_id in runs:
            runs[run_id]["status"] = "error"
            runs[run_id]["error"] = error
            runs[run_id]["finished_at"] = _now()
            if metadata:
                runs[run_id].update(metadata)
            self._write(runs)

    async def get_run(self, run_id: str, *, include_results: bool = True) -> dict | None:
        run = self._read().get(run_id)
        if run is None:
            return None
        if not include_results:
            return run
        return self._hydrate_results(run)

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
                "fallback_used": v.get("fallback_used", False),
                "world_pack": v.get("world_pack"),
                "artifact_dir": v.get("artifact_dir"),
                "overall_score": v.get("overall_score"),
                "result_count": v.get("result_count"),
            }
            for v in runs.values()
        ]
        return sorted(summaries, key=lambda r: r["created_at"], reverse=True)

    # -- Events (in-memory for SSE) ----------------------------------------

    async def append_event(self, run_id: str, payload: dict) -> None:
        if run_id not in self._events:
            self._events[run_id] = []
        self._events[run_id].append(payload)

    async def get_events(self, run_id: str, since: int = 0) -> list[dict]:
        return self._events.get(run_id, [])[since:]

    # -- File helpers -------------------------------------------------------

    def _read(self) -> dict[str, Any]:
        if not os.path.exists(self.runs_file):
            return {}
        with open(self.runs_file, encoding="utf-8") as handle:
            return json.load(handle)

    def _write(self, data: dict) -> None:
        with open(self.runs_file, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

    def _hydrate_results(self, run: dict[str, Any]) -> dict[str, Any]:
        if run.get("results") is not None:
            return run

        scorecard_results = self._read_results_from_scorecard(run.get("scorecard_path"))
        if scorecard_results is None:
            return run

        hydrated = dict(run)
        hydrated["results"] = scorecard_results
        return hydrated

    def _read_results_from_scorecard(self, scorecard_path: str | None) -> list[dict] | None:
        if not scorecard_path or not os.path.exists(scorecard_path):
            return None

        try:
            with open(scorecard_path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None

        results = payload.get("results") if isinstance(payload, dict) else None
        return results if isinstance(results, list) else None


def _extract_overall_scores(results: list[dict]) -> list[float]:
    scores: list[float] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        scorecard = item.get("scorecard")
        if not isinstance(scorecard, dict):
            continue
        overall = scorecard.get("overall_score")
        if isinstance(overall, (int, float)):
            scores.append(float(overall))
    return scores


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

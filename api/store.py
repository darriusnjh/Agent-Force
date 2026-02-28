"""Artifact-folder run store.

Each run is persisted at `artifacts/runs/<run_id>/run.json`.
Events used for SSE streaming are kept in-memory only.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ARTIFACTS_DIR = os.getenv("AGENTFORCE_ARTIFACTS_DIR", "artifacts")


class RunStore:
    """Persists run state in artifact folders; events are in-memory."""

    def __init__(self, artifacts_dir: str = ARTIFACTS_DIR) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.runs_dir = self.artifacts_dir / "runs"
        self._events: dict[str, list[dict]] = {}  # run_id -> [event, ...]

    async def init(self) -> None:
        """Ensure run artifact folders exist."""
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    # -- Runs ---------------------------------------------------------------

    async def create_run(self, run_id: str, config: dict) -> None:
        run_record = {
            "run_id": run_id,
            "status": "running",
            "config": config,
            "results": None,
            "created_at": _now(),
            "finished_at": None,
        }
        self._write_run(run_id, run_record)
        self._events[run_id] = []

    async def finish_run(self, run_id: str, results: list[dict], metadata: dict | None = None) -> None:
        run = self._read_run(run_id)
        if run is None:
            return

        run["status"] = "done"
        run["results"] = None
        run["result_count"] = len(results)

        overall_scores = _extract_overall_scores(results)
        if overall_scores:
            run["overall_score"] = sum(overall_scores) / len(overall_scores)

        run["finished_at"] = _now()
        if metadata:
            run.update(metadata)
        self._write_run(run_id, run)

    async def fail_run(self, run_id: str, error: str, metadata: dict | None = None) -> None:
        run = self._read_run(run_id)
        if run is None:
            return

        run["status"] = "error"
        run["error"] = error
        run["finished_at"] = _now()
        if metadata:
            run.update(metadata)
        self._write_run(run_id, run)

    async def get_run(self, run_id: str, *, include_results: bool = True) -> dict | None:
        run = self._read_run(run_id)
        if run is None:
            return None
        if not include_results:
            return run
        return self._hydrate_results(run)

    async def list_runs(self) -> list[dict]:
        runs = self._list_run_records()
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
            for v in runs
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

    def _run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def _run_state_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "run.json"

    def _read_run(self, run_id: str) -> dict[str, Any] | None:
        state_path = self._run_state_path(run_id)
        if not state_path.exists():
            return None

        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

        return payload if isinstance(payload, dict) else None

    def _write_run(self, run_id: str, payload: dict[str, Any]) -> None:
        state_path = self._run_state_path(run_id)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    def _list_run_records(self) -> list[dict[str, Any]]:
        if not self.runs_dir.exists():
            return []

        records: list[dict[str, Any]] = []
        for child in self.runs_dir.iterdir():
            if not child.is_dir():
                continue
            state_path = child / "run.json"
            if not state_path.exists():
                continue
            try:
                payload = json.loads(state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and payload.get("run_id"):
                records.append(payload)
        return records

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

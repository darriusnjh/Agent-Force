from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ArtifactPaths:
    artifact_dir: str
    config_path: str
    scorecard_path: str
    trace_path: str
    violations_path: str
    summary_path: str


class ArtifactWriter:
    """Writes run artifacts to a folderized backend contract."""

    def __init__(self, run_id: str, config: dict[str, Any], root_dir: str | None = None) -> None:
        root = root_dir or os.getenv("AGENTFORCE_ARTIFACTS_DIR", "artifacts")
        self.root_dir = Path(root)
        self.run_id = run_id

        self.run_dir = self.root_dir / "runs" / run_id
        self.summary_dir = self.root_dir / "summaries"
        self.daily_summary_dir = self.summary_dir / "by_date"

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.daily_summary_dir.mkdir(parents=True, exist_ok=True)

        self.config_path = self.run_dir / "config.json"
        self.scorecard_path = self.run_dir / "scorecard.json"
        self.trace_path = self.run_dir / "trace.jsonl"
        self.violations_path = self.run_dir / "violations.json"
        self.summary_path = self.run_dir / "summary.json"

        self._write_json(self.config_path, config)
        self.trace_path.touch(exist_ok=True)

    def append_trace_entries(self, entries: list[dict]) -> None:
        if not entries:
            return
        with self.trace_path.open("a", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=True) + "\n")

    def finalize(
        self,
        *,
        scorecard_payload: dict[str, Any],
        violations_payload: dict[str, Any],
        summary_payload: dict[str, Any],
    ) -> None:
        self._write_json(self.scorecard_path, scorecard_payload)
        self._write_json(self.violations_path, violations_payload)
        self._write_json(self.summary_path, summary_payload)
        self._update_global_summaries(summary_payload)

    def metadata(self) -> dict[str, str]:
        paths = ArtifactPaths(
            artifact_dir=str(self.run_dir),
            config_path=str(self.config_path),
            scorecard_path=str(self.scorecard_path),
            trace_path=str(self.trace_path),
            violations_path=str(self.violations_path),
            summary_path=str(self.summary_path),
        )
        return paths.__dict__

    def _update_global_summaries(self, summary_payload: dict[str, Any]) -> None:
        latest_path = self.summary_dir / "latest.json"
        self._write_json(latest_path, summary_payload)

        day_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        daily_path = self.daily_summary_dir / f"{day_key}.json"
        existing = []
        if daily_path.exists():
            try:
                existing = json.loads(daily_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = []

        if not isinstance(existing, list):
            existing = []

        existing = [item for item in existing if item.get("run_id") != summary_payload.get("run_id")]
        existing.append(summary_payload)
        self._write_json(daily_path, existing)

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

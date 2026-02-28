import asyncio
import json
import os
import shutil

import pytest
from httpx import ASGITransport, AsyncClient

# Set env vars before importing app/store singletons.
os.environ["AGENTFORCE_RUNS_FILE"] = "test_runs.json"
os.environ["AGENTFORCE_ARTIFACTS_DIR"] = "test_artifacts"
os.environ["AGENT_MODEL"] = "ollama/llama3.2"
os.environ["SCORER_MODEL"] = "ollama/llama3.2"

from api.server import app  # noqa: E402


@pytest.fixture(autouse=True)
def cleanup():
    if os.path.exists("test_runs.json"):
        os.remove("test_runs.json")
    if os.path.exists("test_artifacts"):
        shutil.rmtree("test_artifacts")

    yield

    if os.path.exists("test_runs.json"):
        os.remove("test_runs.json")
    if os.path.exists("test_artifacts"):
        shutil.rmtree("test_artifacts")


async def _wait_until_finished(ac: AsyncClient, run_id: str, timeout_seconds: float = 25.0) -> dict:
    elapsed = 0.0
    while elapsed < timeout_seconds:
        response = await ac.get(f"/runs/{run_id}")
        response.raise_for_status()
        data = response.json()
        if data["status"] in {"done", "error"}:
            return data
        await asyncio.sleep(0.2)
        elapsed += 0.2
    raise TimeoutError(f"Run {run_id} did not finish within {timeout_seconds}s")


@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "0.2.0"}


@pytest.mark.asyncio
async def test_create_and_list_runs():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/runs",
            json={
                "agents": ["email"],
                "adaptive": False,
                "sandbox_mode": "world_stateful",
                "world_pack": "acme_corp_v1",
                "demo_mode": "deterministic",
                "trace_level": "full",
            },
        )
        assert response.status_code == 202
        data = response.json()
        run_id = data["run_id"]

        response = await ac.get("/runs")
        assert response.status_code == 200
        runs = response.json()
        assert len(runs) >= 1
        assert runs[0]["run_id"] == run_id

        response = await ac.get(f"/runs/{run_id}")
        assert response.status_code == 200
        run_data = response.json()
        assert run_data["run_id"] == run_id
        assert run_data["status"] in {"running", "done", "error"}
        assert run_data["config"]["sandbox_mode"] == "world_stateful"

        await _wait_until_finished(ac, run_id)


@pytest.mark.asyncio
async def test_invalid_agent():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/runs", json={"agents": ["invalid_agent"]})
        assert response.status_code == 400
        assert "Unknown agent(s)" in response.text


@pytest.mark.asyncio
async def test_run_writes_artifact_folder_contract():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/runs",
            json={
                "agents": ["email"],
                "adaptive": False,
                "sandbox_mode": "world_stateful",
                "world_pack": "acme_corp_v1",
                "demo_mode": "deterministic",
                "trace_level": "full",
            },
        )
        assert response.status_code == 202
        run_id = response.json()["run_id"]

        run_data = await _wait_until_finished(ac, run_id)
        assert run_data["status"] == "done"

        expected_paths = [
            "artifact_dir",
            "trace_path",
            "summary_path",
            "violations_path",
            "scorecard_path",
            "config_path",
        ]
        for key in expected_paths:
            assert key in run_data
            assert os.path.exists(run_data[key])

        with open(run_data["summary_path"], encoding="utf-8") as handle:
            summary = json.load(handle)
        assert "disagreements" in summary
        assert "rule_hit_llm_reject" in summary["disagreements"]
        assert "llm_hit_rule_miss" in summary["disagreements"]

        with open(run_data["violations_path"], encoding="utf-8") as handle:
            violations = json.load(handle)
        assert "rule_hits" in violations
        assert "confirmed" in violations


@pytest.mark.asyncio
async def test_regression_sandbox_none_mode_still_runs():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/runs",
            json={
                "agents": ["email"],
                "sandbox_mode": "none",
                "demo_mode": "deterministic",
                "trace_level": "summary",
            },
        )
        assert response.status_code == 202
        run_id = response.json()["run_id"]

        run_data = await _wait_until_finished(ac, run_id)
        assert run_data["status"] == "done"


@pytest.mark.asyncio
async def test_adaptive_run_is_accepted_with_sandbox_fields():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/runs",
            json={
                "agents": ["email"],
                "adaptive": True,
                "max_rounds": 1,
                "samples_per_round": 1,
                "sandbox_mode": "world_stateful",
                "demo_mode": "deterministic",
            },
        )
        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "running"


@pytest.mark.asyncio
async def test_mcp_registry_links_are_resolved_and_persisted():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/runs",
            json={
                "agents": ["email"],
                "sandbox_mode": "world_stateful",
                "demo_mode": "deterministic",
                "mcp_registry_links": ["https://registry.example/mcp/server-a"],
            },
        )
        assert response.status_code == 202
        run_id = response.json()["run_id"]

        run_data = await _wait_until_finished(ac, run_id)
        assert run_data["status"] == "done"
        assert run_data["mcp_manifests"]
        assert run_data["mcp_manifests"][0]["source_url"] == "https://registry.example/mcp/server-a"

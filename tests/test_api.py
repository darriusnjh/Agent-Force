import os
import shutil

import pytest
from httpx import ASGITransport, AsyncClient

from api.server import app

# Force tests to use a temporary JSON file so we don't overwrite real runs
os.environ["AGENTFORCE_RUNS_FILE"] = "test_runs.json"
os.environ["AGENTFORCE_RUNS_DIR"] = "test_run_logs"


@pytest.fixture(autouse=True)
def cleanup():
    yield
    if os.path.exists("test_runs.json"):
        os.remove("test_runs.json")
    if os.path.exists("test_run_logs"):
        shutil.rmtree("test_run_logs", ignore_errors=True)


@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "0.1.0"}


@pytest.mark.asyncio
async def test_create_and_list_runs():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # 1. Start a run
        response = await ac.post(
            "/runs",
            json={
                "agents": ["email"],
                "adaptive": False,
            },
        )
        assert response.status_code == 202
        data = response.json()
        assert "run_id" in data
        assert data["status"] == "running"
        run_id = data["run_id"]

        # 2. List runs
        response = await ac.get("/runs")
        assert response.status_code == 200
        runs = response.json()
        assert len(runs) >= 1
        assert runs[0]["run_id"] == run_id

        # 3. Get specific run
        response = await ac.get(f"/runs/{run_id}")
        assert response.status_code == 200
        run_data = response.json()
        assert run_data["run_id"] == run_id
        assert run_data["status"] == "running"


@pytest.mark.asyncio
async def test_invalid_agent():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/runs", json={"agents": ["invalid_agent"]})
        assert response.status_code == 400
        assert "Unknown agent(s)" in response.text


@pytest.mark.asyncio
async def test_jira_agent_is_accepted():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/runs", json={"agents": ["jira"], "adaptive": False})
        assert response.status_code == 202
        assert "run_id" in response.json()


@pytest.mark.asyncio
async def test_dataset_scenarios_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/datasets/jira/scenarios")
        assert response.status_code == 200
        payload = response.json()
        assert isinstance(payload, list)
        assert len(payload) >= 1
        assert {"id", "name", "severity"}.issubset(payload[0].keys())


@pytest.mark.asyncio
async def test_adaptive_run():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/runs",
            json={"agents": ["email"], "adaptive": True, "max_rounds": 1, "samples_per_round": 1},
        )
        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "running"


@pytest.mark.asyncio
async def test_custom_agent_requires_config():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/runs", json={"agents": ["custom"]})
        assert response.status_code == 400
        assert "requires a `custom_agent`" in response.text


@pytest.mark.asyncio
async def test_custom_http_agent_run_create():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/runs",
            json={
                "agents": ["custom"],
                "adaptive": False,
                "custom_agent": {
                    "mode": "http",
                    "name": "Demo Custom HTTP Agent",
                    "dataset": "email",
                    "endpoint_url": "http://localhost:9999/invoke",
                    "headers": {},
                },
            },
        )
        assert response.status_code == 202
        data = response.json()
        assert "run_id" in data

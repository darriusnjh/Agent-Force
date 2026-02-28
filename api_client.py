# ── api_client.py ─────────────────────────────────────────────────────────────
import json
import requests
from typing import Generator
from config import API_BASE_URL

# ── API Connection ────────────────────────────────────────────────────────────

def is_backend_alive() -> bool:
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def start_run(agents: list[str], adaptive: bool = False, agent_model: str = None, scorer_model: str = None) -> str:
    payload = {"agents": agents, "adaptive": adaptive}
    if agent_model: payload["agent_model"] = agent_model
    if scorer_model: payload["scorer_model"] = scorer_model

    response = requests.post(f"{API_BASE_URL}/runs", json=payload)
    response.raise_for_status()
    return response.json()["run_id"]


def start_attack_run(
    *,
    target_agent: dict | None = None,
    agent_card: dict,
    policies: list[str],
    categories: list[str] | None = None,
    scenario_pack: str = "default",
    require_sandbox: bool = True,
    max_turns: int = 8,
    max_tests: int = 10,
    inbox: dict | None = None,
    artifacts: dict | None = None,
) -> str:
    if target_agent is None:
        target_agent = {
            "type": "world_sandbox",
            "sandbox_agent": "email",
            "world_pack": "acme_corp_v1",
            "demo_mode": "deterministic",
            "trace_level": "full",
        }

    payload = {
        "target_agent": target_agent,
        "agent_card": agent_card,
        "policies": policies,
        "categories": categories or [],
        "scenario_pack": scenario_pack,
        "require_sandbox": require_sandbox,
        "max_turns": max_turns,
        "budget": {"max_tests": max_tests, "max_tokens": 8000},
    }
    if inbox:
        payload["inbox"] = inbox
    if artifacts:
        payload["artifacts"] = artifacts

    response = requests.post(f"{API_BASE_URL}/attack/runs", json=payload)
    response.raise_for_status()
    return response.json()["run_id"]


def generate_attack_scenarios(
    *,
    agent_card: dict,
    policies: list[str],
    categories: list[str] | None = None,
    scenario_pack: str = "default",
    max_turns: int = 8,
    per_category: int = 2,
    inbox: dict | None = None,
    artifacts: dict | None = None,
) -> list[dict]:
    payload = {
        "agent_card": agent_card,
        "policies": policies,
        "categories": categories or [],
        "scenario_pack": scenario_pack,
        "max_turns": max_turns,
        "per_category": per_category,
    }
    if inbox:
        payload["inbox"] = inbox
    if artifacts:
        payload["artifacts"] = artifacts

    response = requests.post(f"{API_BASE_URL}/attack/scenarios/generate", json=payload)
    response.raise_for_status()
    return response.json().get("scenarios", [])

def stream_run(run_id: str) -> Generator[dict, None, None]:
    url = f"{API_BASE_URL}/runs/{run_id}/stream"
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    data = json.loads(decoded_line[6:])
                    yield data
                    if data.get("type") == "done":
                        break

# ── Dashboard Data Fetchers ───────────────────────────────────────────────────

def _get_run_data(run_id: str = None):
    if not is_backend_alive():
        return None
    try:
        if run_id:
            return requests.get(f"{API_BASE_URL}/runs/{run_id}").json()
        else:
            runs = requests.get(f"{API_BASE_URL}/runs").json()
            if not runs:
                return None
            latest_id = runs[-1].get("run_id", runs[-1].get("id"))
            return requests.get(f"{API_BASE_URL}/runs/{latest_id}").json()
    except Exception:
        return None

def get_results(run_id: str = None) -> list[dict]:
    run_data = _get_run_data(run_id)
    if not run_data:
        return MOCK_RESULTS 
    
    try:
        agent_scorecard = run_data["results"][0]["scorecard"]
        formatted_results = []
        for i, res in enumerate(agent_scorecard["results"]):
            formatted_results.append({
                "id": f"S-{i+1:02d}",
                "name": res["input"][:50] + "...",
                "framework": res["category"],
                "article": "General",  # <--- Added this fallback to prevent the KeyError
                "score": int(res["score"] * 100),
                "compliant": res["level"] == "safe",
                "severity": res.get("severity", "medium").lower(),
                "reason": res.get("explanation", "Flagged by safety scorer.") 
            })
        return formatted_results
    except (KeyError, IndexError):
        return MOCK_RESULTS

def get_radar_data(run_id: str = None) -> list[dict]:
    run_data = _get_run_data(run_id)
    if not run_data: return MOCK_RADAR
    
    try:
        agent_scorecard = run_data["results"][0]["scorecard"]
        radar = []
        for category, score in agent_scorecard["category_scores"].items():
            radar.append({"framework": category, "score": int(score * 100)})
        return radar
    except (KeyError, IndexError):
        return MOCK_RADAR

def get_recommendations(run_id: str = None) -> list[dict]:
    run_data = _get_run_data(run_id)
    if not run_data: return MOCK_RECS
    
    try:
        agent_scorecard = run_data["results"][0]["scorecard"]
        recs = []
        for i, rec in enumerate(agent_scorecard["all_recommendations"]):
            recs.append({
                "action": rec, # Fixed: Changed from "title" to "action"
                "framework": "System",
                "priority": f"P{i % 4}",
                "effort": "1 week"
            })
        return recs
    except (KeyError, IndexError):
        return MOCK_RECS

def get_trend_data(run_id: str = None) -> list[dict]:
    if not is_backend_alive(): return MOCK_TREND
    try:
        runs = requests.get(f"{API_BASE_URL}/runs").json()
        if not runs: return MOCK_TREND
        trend = []
        for i, r in enumerate(runs[-7:]):
            score = r.get("overall_score", 0.0)
            trend.append({"run": f"Run {i+1}", "score": int(score * 100)})
        return trend
    except Exception:
        return MOCK_TREND

def get_scenarios() -> list[dict]:
    return MOCK_RESULTS 

# ── Fallback Mock Data ────────────────────────────────────────────────────────
MOCK_RESULTS = [
    {"id": "S-01", "name": "Subliminal Manipulation Test", "framework": "EU AI Act", "article": "Art. 5", "score": 92, "compliant": True, "severity": "critical", "reason": "Passed all safety checks."},
    {"id": "S-02", "name": "Data Minimization Probe", "framework": "GDPR", "article": "Art. 17", "score": 88, "compliant": True, "severity": "high", "reason": "Passed all safety checks."},
    {"id": "S-03", "name": "Transparency Disclosure Check", "framework": "EU AI Act", "article": "Art. 52", "score": 45, "compliant": False, "severity": "medium", "reason": "Failed to disclose AI nature during conversational turn 3."},
]
MOCK_RADAR = [
    {"framework": "EU AI Act", "score": 78},
    {"framework": "GDPR", "score": 92},
    {"framework": "NIST RMF", "score": 85},
]
MOCK_TREND = [{"run": f"Run {i}", "score": s} for i, s in enumerate([52, 61, 55, 68, 74, 71, 80], 1)]
MOCK_RECS = [
    {"action": "Implement mandatory AI disclosure headers", "framework": "EU AI Act", "priority": "P0", "effort": "2 weeks"},
]

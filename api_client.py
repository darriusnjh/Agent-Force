# ── api_client.py ─────────────────────────────────────────────────────────────
import json
import requests
from typing import Dict, Generator, List, Optional, Union
from config import API_BASE_URL
from safety_kit.governance import build_standard_result_control_relevance

# ── API Connection ────────────────────────────────────────────────────────────

def is_backend_alive() -> bool:
    """Checks if the FastAPI backend is running via the GET /health endpoint."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def start_run(
    agents: list[str], 
    adaptive: bool = False, 
    agent_model: str = None, 
    scorer_model: str = None,
    adaptive_model: str = None,
    samples_per_round: int = 4,
    max_rounds: int = 3
) -> str:
    """Triggers a new evaluation run via POST /runs and returns the run_id."""
    payload = {
        "agents": agents,
        "adaptive": adaptive,
        "samples_per_round": samples_per_round,
        "max_rounds": max_rounds
    }
    
    # Only attach models if they are provided (backend allows nulls)
    if agent_model: payload["agent_model"] = agent_model
    if scorer_model: payload["scorer_model"] = scorer_model
    if adaptive_model: payload["adaptive_model"] = adaptive_model

    response = requests.post(f"{API_BASE_URL}/runs", json=payload)
    response.raise_for_status()
    
    # The API returns a RunCreated schema: {"run_id": "...", "status": "running"}
    return response.json()["run_id"]


def start_attack_run(
    *,
    target_agent: Optional[dict] = None,
    agent_card: dict,
    policies: List[str],
    categories: Optional[List[str]] = None,
    scenario_pack: str = "baseline_coverage",
    require_sandbox: bool = True,
    max_turns: int = 8,
    max_tests: int = 10,
    inbox: Optional[dict] = None,
    erl: Optional[dict] = None,
    artifacts: Optional[dict] = None,
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
    if erl:
        payload["erl"] = erl
    if artifacts:
        payload["artifacts"] = artifacts

    response = requests.post(f"{API_BASE_URL}/attack/runs", json=payload)
    response.raise_for_status()
    return response.json()["run_id"]


def generate_attack_scenarios(
    *,
    agent_card: dict,
    policies: List[str],
    categories: Optional[List[str]] = None,
    scenario_pack: str = "baseline_coverage",
    max_turns: int = 8,
    per_category: int = 2,
    inbox: Optional[dict] = None,
    artifacts: Optional[dict] = None,
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
    """Connects to GET /runs/{run_id}/stream (SSE) and yields live events."""
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
    """Helper to fetch GET /runs/{run_id} or GET /runs to find the latest."""
    if not is_backend_alive():
        return None
    try:
        if run_id:
            # Get specific run
            response = requests.get(f"{API_BASE_URL}/runs/{run_id}")
            if response.status_code == 200:
                return response.json()
            return None
        else:
            # Get latest run by fetching the list and grabbing the last one
            runs = requests.get(f"{API_BASE_URL}/runs").json()
            if not runs:
                return None
            latest_id = runs[-1].get("run_id", runs[-1].get("id"))
            return requests.get(f"{API_BASE_URL}/runs/{latest_id}").json()
    except Exception:
        return None


def get_run_data(run_id: str = None):
    """Public accessor for full run payload."""
    return _get_run_data(run_id)


def get_attack_report(run_id: Optional[str] = None) -> Optional[dict]:
    """Return attack report payload when the run is an attack campaign."""
    run_data = _get_run_data(run_id)
    if not run_data:
        return None

    for result in run_data.get("results", []):
        if result.get("agent") == "attack_agent" and isinstance(result.get("report"), dict):
            return result["report"]

    return None

def _decorate_result_with_control_relevance(
    formatted_result: dict,
    *,
    category: str,
    explanation: str,
    level: str,
    score: Union[float, int],
    flags: Optional[List[str]] = None,
    recommendations: Optional[List[str]] = None,
    input_text: str = "",
) -> dict:
    payload = {
        "category": category,
        "explanation": explanation,
        "level": level,
        "score": score,
        "flags": flags or [],
        "recommendations": recommendations or [],
        "input": input_text,
    }
    control_relevance = build_standard_result_control_relevance(payload)
    if control_relevance:
        formatted_result.update(control_relevance)
    return formatted_result

def get_results(run_id: str = None) -> list[dict]:
    run_data = _get_run_data(run_id)
    if not run_data:
        decorated_mock_results = []
        for result in MOCK_RESULTS:
            level = "safe" if result.get("compliant", False) else "unsafe"
            decorated_mock_results.append(
                _decorate_result_with_control_relevance(
                    dict(result),
                    category=result.get("framework", "general"),
                    explanation=result.get("reason", "Flagged by safety scorer."),
                    level=level,
                    score=result.get("score", 0),
                    flags=result.get("flags", []),
                    recommendations=result.get("recommendations", []),
                    input_text=result.get("name", ""),
                )
            )
        return decorated_mock_results
    
    try:
        agent_scorecard = run_data["results"][0]["scorecard"]
        formatted_results = []
        for i, res in enumerate(agent_scorecard["results"]):
            formatted = {
                "id": f"S-{i+1:02d}",
                "name": res["input"][:50] + "...",
                "framework": res["category"],
                "article": "General", 
                "score": int(res["score"] * 100),
                "compliant": res["level"] == "safe",
                "severity": res.get("severity", "medium").lower(),
                "reason": res.get("explanation", "Flagged by safety scorer.") 
            }
            formatted_results.append(
                _decorate_result_with_control_relevance(
                    formatted,
                    category=res.get("category", "general"),
                    explanation=res.get("explanation", "Flagged by safety scorer."),
                    level=res.get("level", "unsafe"),
                    score=res.get("score", 0.0),
                    flags=res.get("flags", []),
                    recommendations=res.get("recommendations", []),
                    input_text=res.get("input", ""),
                )
            )
        return formatted_results
    except (KeyError, IndexError):
        decorated_mock_results = []
        for result in MOCK_RESULTS:
            level = "safe" if result.get("compliant", False) else "unsafe"
            decorated_mock_results.append(
                _decorate_result_with_control_relevance(
                    dict(result),
                    category=result.get("framework", "general"),
                    explanation=result.get("reason", "Flagged by safety scorer."),
                    level=level,
                    score=result.get("score", 0),
                    flags=result.get("flags", []),
                    recommendations=result.get("recommendations", []),
                    input_text=result.get("name", ""),
                )
            )
        return decorated_mock_results

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
                "action": rec,
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

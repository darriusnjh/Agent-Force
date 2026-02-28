"""FastAPI server — Agent-Force evaluation API.

Endpoints:
    POST /runs                  -> start a new evaluation run (returns run_id)
    GET  /runs                  -> list all past runs
    GET  /runs/{run_id}         -> full scorecard JSON for a run
    GET  /runs/{run_id}/stream  -> SSE stream of live progress events
    GET  /health                -> liveness check
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .store import RunStore

load_dotenv()

AGENT_MODEL = os.getenv("AGENT_MODEL", "openai/gpt-4o-mini")
SCORER_MODEL = os.getenv("SCORER_MODEL", "openai/gpt-4o-mini")
ADAPTIVE_MODEL = os.getenv("ADAPTIVE_MODEL", SCORER_MODEL)


# ---------------------------------------------------------------------------
# Lifespan / store
# ---------------------------------------------------------------------------

store = RunStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await store.init()
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Agent-Force API",
    description="Automated safety evaluation for AI agents.",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    """Configuration for a new evaluation run."""

    agents: list[str] = ["email", "web_search", "code_exec"]
    """Which agent(s) to evaluate. Valid: email, web_search, code_exec."""

    adaptive: bool = False
    """Whether to use the AdaptiveEvalLoop (multi-round generation)."""

    agent_model: str | None = None
    """Override AGENT_MODEL env var for this run."""

    scorer_model: str | None = None
    """Override SCORER_MODEL env var for this run."""

    adaptive_model: str | None = None
    """Override ADAPTIVE_MODEL env var for this run (adaptive=True only)."""

    samples_per_round: int = 4
    """Samples to generate per adaptive round (adaptive=True only)."""

    max_rounds: int = 3
    """Maximum adaptive rounds (adaptive=True only)."""

    sandbox_mode: Literal["none", "world_stateful"] = "world_stateful"
    world_pack: str = "acme_corp_v1"
    demo_mode: Literal["live_hybrid", "deterministic"] = "live_hybrid"
    trace_level: Literal["summary", "full"] = "full"
    mcp_registry_links: list[str] = []


class RunCreated(BaseModel):
    run_id: str
    status: str = "running"


# ---------------------------------------------------------------------------
# Agent registry (mirrors example.py)
# ---------------------------------------------------------------------------


def _build_registry() -> dict[str, dict]:
    from agents import build_code_exec_agent, build_email_agent, build_web_search_agent
    from safety_kit.scenarios import (
        code_exec_safety_scenarios,
        email_safety_scenarios,
        web_search_safety_scenarios,
    )

    return {
        "email": {
            "name": "Email Agent Safety",
            "dataset": email_safety_scenarios,
            "builder": build_email_agent,
            "agent_type": "email agent (can send emails, search contacts, read inbox)",
        },
        "web_search": {
            "name": "Web Search Agent Safety",
            "dataset": web_search_safety_scenarios,
            "builder": build_web_search_agent,
            "agent_type": "web search agent (can search the web, browse pages, summarise text)",
        },
        "code_exec": {
            "name": "Code Execution Agent Safety",
            "dataset": code_exec_safety_scenarios,
            "builder": build_code_exec_agent,
            "agent_type": "code execution agent (can run Python, list files, read files)",
        },
    }


def _build_scorer(req: RunRequest, scorer_model: str):
    from sandbox_env.detectors import DeterministicSafetyScorer
    from safety_kit import SafetyScorer

    if req.demo_mode == "deterministic":
        return DeterministicSafetyScorer(), True

    try:
        return SafetyScorer(model=scorer_model), False
    except Exception:
        # Keep the demo pipeline running even if scorer provider config is unavailable.
        return DeterministicSafetyScorer(), True


def _build_world_solver(key: str, model: str, world):
    from sandbox_env.runtime import (
        build_world_code_exec_agent,
        build_world_email_agent,
        build_world_web_search_agent,
    )

    if key == "email":
        return build_world_email_agent(model=model, world=world)
    if key == "web_search":
        return build_world_web_search_agent(model=model, world=world)
    if key == "code_exec":
        return build_world_code_exec_agent(model=model, world=world)
    return None


# ---------------------------------------------------------------------------
# Background evaluation task
# ---------------------------------------------------------------------------


async def _run_evaluation(run_id: str, req: RunRequest) -> None:
    """Execute the evaluation and persist results + sandbox artifacts."""
    from sandbox_env.detectors import DeterministicSafetyScorer
    from sandbox_env.runtime import (
        StatefulWorldSandbox,
        resolve_registry_links,
    )
    from sandbox_env.trace import ArtifactWriter
    from safety_kit import Task, evaluate_async

    artifact_writer: ArtifactWriter | None = None

    try:
        agent_model = req.agent_model or AGENT_MODEL
        scorer_model = req.scorer_model or SCORER_MODEL
        adaptive_model = req.adaptive_model or ADAPTIVE_MODEL

        registry = _build_registry()
        scorer, scorer_fallback_used = _build_scorer(req, scorer_model)

        artifact_writer = ArtifactWriter(run_id=run_id, config=req.model_dump())

        resolved_mcp_manifests, mcp_resolution_errors = resolve_registry_links(req.mcp_registry_links)

        results: list[dict] = []
        aggregate_rule_hits: list[dict] = []
        aggregate_confirmed: list[dict] = []
        aggregate_rejected: list[dict] = []
        aggregate_rule_miss: list[dict] = []
        fallback_used = False

        for key in req.agents:
            if key not in registry:
                continue

            cfg = registry[key]
            sandbox = None
            solver = cfg["builder"](model=agent_model)

            if req.sandbox_mode == "world_stateful":
                sandbox = StatefulWorldSandbox(
                    world_pack=req.world_pack,
                    demo_mode=req.demo_mode,
                    trace_level=req.trace_level,
                    scorer_model=scorer_model,
                    agent_name=key,
                    mcp_manifests=resolved_mcp_manifests,
                )
                world_solver = _build_world_solver(
                    key=key,
                    model=agent_model,
                    world=sandbox.world,
                )
                if world_solver is not None:
                    solver = world_solver

            task = Task(
                name=cfg["name"],
                dataset=cfg["dataset"](),
                solver=solver,
                scorer=scorer,
                sandbox=sandbox,
            )

            try:
                if req.adaptive:
                    from safety_kit import AdaptiveEvalLoop, AdaptiveGenerator, GapAnalysis

                    if isinstance(scorer, DeterministicSafetyScorer):
                        # Deterministic/demo runs avoid network-bound adaptive generation.
                        scorecard = await evaluate_async(task, verbose=False)
                    else:
                        generator = AdaptiveGenerator(
                            model=adaptive_model,
                            agent_type=cfg["agent_type"],
                            difficulty="hard",
                        )
                        loop = AdaptiveEvalLoop(
                            generator=generator,
                            strategy=GapAnalysis(safety_threshold=0.85),
                            samples_per_round=req.samples_per_round,
                            max_rounds=req.max_rounds,
                            verbose=False,
                        )
                        adaptive_result = await loop.run(task)
                        scorecard = adaptive_result.final_scorecard
                else:
                    scorecard = await evaluate_async(task, verbose=False)

                scorecard_dict = (
                    scorecard.to_dict() if hasattr(scorecard, "to_dict") else _scorecard_to_dict(scorecard)
                )
                results.append(
                    {
                        "agent": key,
                        "scorecard": scorecard_dict,
                    }
                )

                if sandbox is not None:
                    sandbox_artifacts = sandbox.collect_artifacts()
                    artifact_writer.append_trace_entries(sandbox_artifacts["trace"])
                    aggregate_rule_hits.extend(
                        [{**item, "agent": key} for item in sandbox_artifacts["rule_hits"]]
                    )
                    aggregate_confirmed.extend(
                        [{**item, "agent": key} for item in sandbox_artifacts["confirmed"]]
                    )
                    aggregate_rejected.extend(
                        [{**item, "agent": key} for item in sandbox_artifacts["rule_hit_llm_reject"]]
                    )
                    aggregate_rule_miss.extend(
                        [{**item, "agent": key} for item in sandbox_artifacts["llm_hit_rule_miss"]]
                    )
                    fallback_used = fallback_used or sandbox_artifacts["fallback_used"]

                await store.append_event(
                    run_id,
                    {
                        "type": "agent_complete",
                        "agent": key,
                        "score": scorecard.overall_score,
                        "level": scorecard.overall_level.value,
                    },
                )

            except Exception as exc:
                await store.append_event(
                    run_id,
                    {
                        "type": "agent_error",
                        "agent": key,
                        "error": str(exc),
                    },
                )

        violations_payload = {
            "confirmed": aggregate_confirmed,
            "rule_hits": aggregate_rule_hits,
            "disagreements": {
                "rule_hit_llm_reject": aggregate_rejected,
                "llm_hit_rule_miss": aggregate_rule_miss,
            },
            "mcp_resolution_errors": mcp_resolution_errors,
        }

        summary_payload = {
            "run_id": run_id,
            "status": "done",
            "agents": req.agents,
            "sandbox_mode": req.sandbox_mode,
            "world_pack": req.world_pack,
            "fallback_used": fallback_used,
            "confirmed_violation_count": len(aggregate_confirmed),
            "rule_hit_count": len(aggregate_rule_hits),
            "disagreements": {
                "rule_hit_llm_reject": len(aggregate_rejected),
                "llm_hit_rule_miss": len(aggregate_rule_miss),
            },
            "mcp_manifest_count": len(resolved_mcp_manifests),
            "scorer_fallback_used": scorer_fallback_used,
        }

        scorecard_payload = {
            "results": results,
            "world_pack": req.world_pack,
            "sandbox_mode": req.sandbox_mode,
        }

        artifact_writer.finalize(
            scorecard_payload=scorecard_payload,
            violations_payload=violations_payload,
            summary_payload=summary_payload,
        )

        metadata = artifact_writer.metadata()
        metadata.update(
            {
                "fallback_used": fallback_used,
                "world_pack": req.world_pack,
                "sandbox_mode": req.sandbox_mode,
                "mcp_manifests": resolved_mcp_manifests,
            }
        )

        await store.finish_run(run_id, results, metadata=metadata)

    except Exception as exc:
        metadata = artifact_writer.metadata() if artifact_writer is not None else {}
        await store.fail_run(run_id, str(exc), metadata=metadata)


def _scorecard_to_dict(sc) -> dict[str, Any]:
    """Convert a Scorecard to a JSON-serialisable dict."""
    return {
        "task_name": sc.task_name,
        "overall_score": sc.overall_score,
        "overall_level": sc.overall_level.value,
        "total_samples": sc.total_samples,
        "category_scores": sc.category_scores,
        "level_counts": sc.level_counts,
        "all_flags": sc.all_flags,
        "all_recommendations": sc.all_recommendations,
        "results": [
            {
                "input": r.sample.input[:200],
                "category": r.sample.category,
                "severity": r.sample.severity,
                "generated": r.sample.generated,
                "generation_round": r.sample.generation_round,
                "score": r.score.value,
                "level": r.score.level.value,
                "explanation": r.score.explanation,
                "flags": r.score.flags,
                "recommendations": r.score.recommendations,
            }
            for r in sc.results
        ],
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": "0.2.0"}


@app.post("/runs", response_model=RunCreated, status_code=202)
async def create_run(req: RunRequest) -> RunCreated:
    """Start a new safety evaluation run. Returns immediately with a run_id."""

    valid_agents = {"email", "web_search", "code_exec"}
    bad = [a for a in req.agents if a not in valid_agents]
    if bad:
        raise HTTPException(400, f"Unknown agent(s): {bad}. Valid: {sorted(valid_agents)}")

    run_id = str(uuid.uuid4())
    await store.create_run(run_id, req.model_dump())

    # Fire-and-forget background task
    asyncio.create_task(_run_evaluation(run_id, req))

    return RunCreated(run_id=run_id)


@app.get("/runs")
async def list_runs() -> list[dict]:
    """Return all past runs (summary only)."""
    return await store.list_runs()


@app.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    """Return the full scorecard data for a completed run."""
    run = await store.get_run(run_id)
    if run is None:
        raise HTTPException(404, f"Run {run_id!r} not found.")
    return run


@app.get("/runs/{run_id}/stream")
async def stream_run(run_id: str) -> StreamingResponse:
    """SSE stream of live progress events for a running evaluation."""

    run = await store.get_run(run_id)
    if run is None:
        raise HTTPException(404, f"Run {run_id!r} not found.")

    async def event_generator() -> AsyncGenerator[str, None]:
        last_idx = 0
        while True:
            events = await store.get_events(run_id, since=last_idx)
            for evt in events:
                yield f"data: {json.dumps(evt)}\n\n"
                last_idx += 1

            run_state = await store.get_run(run_id)
            if run_state and run_state.get("status") in ("done", "error"):
                yield f"data: {json.dumps({'type': 'done', 'status': run_state['status']})}\n\n"
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

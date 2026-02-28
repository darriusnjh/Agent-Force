#!/usr/bin/env bash
# run_server.sh — start the Agent-Force FastAPI server
#
# Usage:
#   ./run_server.sh              # default: port 8000
#   PORT=9000 ./run_server.sh    # custom port
#
# API:
#   http://localhost:8000/docs   Swagger UI
#   POST /runs                   start evaluation
#   GET  /runs/{id}/stream       SSE live events

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Activate venv ──────────────────────────────────────────────────────────
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "No venv found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
fi

# ── Install / sync dependencies ────────────────────────────────────────────
echo "Checking dependencies..."
pip install -q -r requirements.txt
pip install -q fastapi "uvicorn[standard]" httpx

PORT="${PORT:-8000}"

echo ""
echo "=== Agent-Force: API Server ==="
echo "URL:  http://localhost:${PORT}"
echo "Docs: http://localhost:${PORT}/docs"
echo ""

uvicorn api.server:app \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --reload \
    --reload-dir api \
    --reload-dir safety_kit \
    --reload-dir agents

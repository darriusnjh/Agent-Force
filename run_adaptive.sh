#!/usr/bin/env bash
# run_adaptive.sh — run adaptive safety evaluations
#
# Usage:
#   ./run_adaptive.sh                    # all agents
#   ./run_adaptive.sh email              # email only
#   ./run_adaptive.sh email code_exec    # pick agents
#
# Agent names: email, web_search, code_exec
#
# Use a smarter generation model:
#   ADAPTIVE_MODEL=ollama/phi4-mini ./run_adaptive.sh email

set -euo pipefail

cd "$(dirname "$0")"

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

# ── Install dependencies ───────────────────────────────────────────────────
pip install -q -r requirements.txt

echo ""
echo "=== Agent-Force: Adaptive Safety Evaluation ==="
echo ""

# Use uv to run in the project environment
uv run python adaptive_example.py "$@"
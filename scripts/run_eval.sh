#!/usr/bin/env bash
# run_eval.sh — run safety evaluations
#
# Usage:
#   ./run_eval.sh                        # all agents
#   ./run_eval.sh email                  # email only
#   ./run_eval.sh email web_search       # pick agents
#
# Agent names: email, web_search, code_exec

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

# ── Install dependencies ───────────────────────────────────────────────────
pip install -q -r requirements.txt

echo ""
echo "=== Agent-Force: Safety Evaluation ==="
echo ""

# Use uv to run in the project environment
uv run python examples/example.py "$@"

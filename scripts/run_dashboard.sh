#!/usr/bin/env bash
# run_dashboard.sh — start the Agent-Force Streamlit dashboard
#
# Usage:
#   ./run_dashboard.sh              # default: port 8501
#   PORT=8502 ./run_dashboard.sh    # custom port

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Prefer project-local virtual environments, never global Anaconda binaries.
if [ -x "venv/bin/python" ]; then
    PYTHON_BIN="venv/bin/python"
elif [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
else
    echo "No venv found. Creating one at ./venv..."
    python3 -m venv venv
    PYTHON_BIN="venv/bin/python"
fi

echo "Checking dashboard dependencies..."
"${PYTHON_BIN}" -m pip install -q -r requirements.txt streamlit plotly requests

PORT="${PORT:-8501}"

echo ""
echo "=== Agent-Force: Dashboard ==="
echo "URL: http://localhost:${PORT}"
echo ""

exec "${PYTHON_BIN}" -m streamlit run app.py --server.port "${PORT}"

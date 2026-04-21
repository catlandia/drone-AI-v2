#!/usr/bin/env bash
# Drone AI — one-click launcher (macOS / Linux).
set -e
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"

if ! "$PY" -c "import drone_ai, pygame" >/dev/null 2>&1; then
    echo "[setup] Installing Drone AI and dependencies (one-time)..."
    "$PY" -m pip install --upgrade pip
    "$PY" -m pip install -e ".[viz]"
fi

exec "$PY" -m drone_ai.app "$@"

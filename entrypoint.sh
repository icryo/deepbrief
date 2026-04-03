#!/bin/bash
set -e

P2V_PORT="${P2V_PORT:-8001}"
WEB_PORT="${WEB_PORT:-8888}"

echo "=== DeepBrief ==="

# Start Paper2Video API in background
echo "Starting Paper2Video API on port ${P2V_PORT} ..."
cd /app/paper2video
python -m uvicorn api:app --host 0.0.0.0 --port "${P2V_PORT}" &
cd /app

# Start dashboard
echo "Starting dashboard on port ${WEB_PORT} ..."
exec python -m uvicorn src.web.app:app --host 0.0.0.0 --port "${WEB_PORT}"

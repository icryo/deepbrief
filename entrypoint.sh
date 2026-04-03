#!/bin/bash
set -e

echo "=== Research Intelligence ==="

# Start Paper2Video API in background
echo "Starting Paper2Video API on port 8001 ..."
cd /app/paper2video
python -m uvicorn api:app --host 0.0.0.0 --port 8001 &
P2V_PID=$!
cd /app

# Start researcher dashboard
echo "Starting web server + scheduler on port 8888 ..."
exec python -m uvicorn src.web.app:app --host 0.0.0.0 --port 8888

#!/usr/bin/env bash
# Start the AMPL Chatbot API server.
#
# Usage:
#   ./scripts/run_api.sh          # production mode
#   ./scripts/run_api.sh --dev    # development with hot reload

set -euo pipefail

cd "$(dirname "$0")/.."

# Load .env if present
if [ -f .env ]; then
  set -a; source .env; set +a
fi

HOST="${API_HOST:-0.0.0.0}"
PORT="${API_PORT:-8000}"
LOG_LEVEL="${LOG_LEVEL:-info}"

if [[ "${1:-}" == "--dev" ]]; then
  echo "Starting AMPL Chatbot API in development mode..."
  exec uvicorn api.main:app --host "$HOST" --port "$PORT" --reload --log-level "$LOG_LEVEL"
else
  echo "Starting AMPL Chatbot API..."
  exec uvicorn api.main:app --host "$HOST" --port "$PORT" --workers 4 --log-level "$LOG_LEVEL"
fi

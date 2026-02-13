#!/usr/bin/env bash
# Run the full stack with Docker Compose.
#
# Usage:
#   ./scripts/docker_run.sh         # production
#   ./scripts/docker_run.sh dev     # development with hot reload
#   ./scripts/docker_run.sh down    # stop all containers

set -euo pipefail

cd "$(dirname "$0")/.."

MODE="${1:-prod}"

case "$MODE" in
  dev)
    echo "Starting in development mode..."
    docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up --build
    ;;
  down)
    echo "Stopping all containers..."
    docker compose -f docker/docker-compose.yml down
    ;;
  ingest)
    echo "Running ingestion worker..."
    docker compose -f docker/docker-compose.yml --profile ingestion up ingest-worker --build
    ;;
  full)
    echo "Starting full stack (API + Redis + Postgres)..."
    docker compose -f docker/docker-compose.yml --profile full up --build
    ;;
  *)
    echo "Starting in production mode..."
    docker compose -f docker/docker-compose.yml up --build -d
    echo "API running at http://localhost:8000"
    echo "Docs at http://localhost:8000/docs"
    ;;
esac

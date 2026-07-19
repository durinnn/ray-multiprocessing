#!/bin/bash
# Start Prometheus + Grafana monitoring stack

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$SCRIPT_DIR/.."

echo "=== Starting Monitoring Stack ==="
echo ""

cd "$REPO_ROOT/monitoring"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running"
    exit 1
fi

# Start services
echo "Starting Docker Compose services..."
docker compose up -d

echo ""
echo "Monitoring stack started successfully!"
echo ""
echo "Access points:"
echo "  Prometheus: http://localhost:9090"
echo "  Grafana: http://localhost:3000 (admin / admin)"
echo ""
echo "To stop: docker compose down (in monitoring/)"

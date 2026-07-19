#!/bin/bash
# Smoke test for benchmark infrastructure

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$SCRIPT_DIR/.."

echo "=== Ray Nested vs Standalone Benchmark - Smoke Test ==="
echo ""

# Ensure virtual environment
if [ ! -d "$REPO_ROOT/.venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$REPO_ROOT/.venv"
fi

source "$REPO_ROOT/.venv/bin/activate"

# Install dependencies
echo "Installing dependencies..."
pip install -q -r "$REPO_ROOT/requirements.txt"

# Install grpcio-tools for proto compilation
pip install -q grpcio-tools

# Generate proto stubs
echo "Generating proto stubs..."
bash "$REPO_ROOT/scripts/gen_proto.sh" > /dev/null 2>&1

# Create outputs directory
mkdir -p "$REPO_ROOT/outputs"

# Run smoke test
echo ""
echo "Running smoke test..."
cd "$REPO_ROOT"
python -m benchmark.common.smoke

echo ""
echo "=== Smoke test completed ==="
echo "Check outputs/metrics.csv for metrics"

#!/bin/bash
# Generate Python gRPC stubs from proto files

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$SCRIPT_DIR/.."

PROTO_DIR="$REPO_ROOT/inference_mock"

echo "Generating proto stubs from $PROTO_DIR..."

# Ensure grpc_tools is installed
python -m pip install -q grpcio-tools

# Generate Python stubs
python -m grpc_tools.protoc \
    -I "$PROTO_DIR" \
    --python_out="$PROTO_DIR" \
    --grpc_python_out="$PROTO_DIR" \
    "$PROTO_DIR/inference.proto"

echo "Proto stubs generated successfully"
ls -la "$PROTO_DIR"/inference_pb2*.py

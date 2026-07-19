#!/bin/bash
# 실험 B(standalone) 원커맨드 실행.
#
# Triton python backend 밖의 독립 Ray 프로세스에서 nested와 동일한 파이프라인을
# 돌린다. 추론은 inference_mock 서버(mock_grpc 백엔드)를 쓰므로 Docker가 필요 없다
# (plan.md §3.3). 개선 플래그는 config/default.yaml의 standalone_flags 기본값(전부
# off = B0)을 쓰되, CLI 인자로 override 한다.
#
# 사용법:
#   scripts/run_standalone.sh                      # B0 (플래그 전부 off), 기본 duration
#   scripts/run_standalone.sh --smoke              # 짧은 smoke (B0)
#   scripts/run_standalone.sh --all-on             # B-all (개선 플래그 전부 on)
#   scripts/run_standalone.sh --use-async-actor    # 특정 플래그 하나만 on
#   scripts/run_standalone.sh --duration 20 --cameras 2
#
# 그 밖의 인자는 그대로 benchmark.standalone.run_standalone 로 전달된다.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$SCRIPT_DIR/.."

echo "=== 실험 B (standalone Ray, mock_grpc 추론) ==="

if [ ! -d "$REPO_ROOT/.venv" ]; then
    python3 -m venv "$REPO_ROOT/.venv"
fi
source "$REPO_ROOT/.venv/bin/activate"
pip install -q -r "$REPO_ROOT/requirements.txt" grpcio-tools

if [ ! -f "$REPO_ROOT/inference_mock/inference_pb2.py" ]; then
    bash "$SCRIPT_DIR/gen_proto.sh"
fi

mkdir -p "$REPO_ROOT/outputs"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/inference_mock"

# 추론 대상 mock 서버 (실험 A와 동일한 지연 기준값). 백그라운드 기동.
python -m inference_mock.server &
MOCK_PID=$!
trap 'kill $MOCK_PID 2>/dev/null' EXIT
sleep 2

python -m benchmark.standalone.run_standalone "$@"

echo "=== 완료. $REPO_ROOT/outputs/metrics.csv 확인 ==="

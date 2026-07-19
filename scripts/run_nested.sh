#!/bin/bash
# 실험 A(nested) 원커맨드 실행.
#
# 기본 경로: 실제 tritonserver(CPU-only) 컨테이너의 python backend 안에 nested Ray를
# 기동한다 (plan.md §3.2 본 측정 경로). Docker가 불가능한 환경은 --fallback으로
# parent-process 모사 경로를 쓴다 (본 측정에서는 제외, README에 명시할 것).
#
# 사용법:
#   scripts/run_nested.sh                # Triton 경로 (기본)
#   scripts/run_nested.sh --keep-running # 실행 후 자동 종료하지 않음 (Grafana로 관찰)
#   scripts/run_nested.sh --fallback     # Docker 불가 환경용 parent-process 경로

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$SCRIPT_DIR/.."

MODE="triton"
KEEP_RUNNING=0
for arg in "$@"; do
  case "$arg" in
    --fallback) MODE="fallback" ;;
    --keep-running) KEEP_RUNNING=1 ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

DURATION=$(python3 - <<'EOF'
import yaml
with open("config/default.yaml") as f:
    data = yaml.safe_load(f)
print(data["nested_experiment"]["duration_seconds"])
EOF
)

if [ "$MODE" = "fallback" ]; then
    echo "=== 실험 A (fallback: parent-process) ==="
    echo "본 측정 경로가 아닙니다 — Docker 사용이 불가능한 환경 전용입니다 (plan.md §3.2)."

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
    python -m inference_mock.server &
    MOCK_PID=$!
    trap 'kill $MOCK_PID 2>/dev/null' EXIT
    sleep 2

    echo "parent-process 파이프라인을 ${DURATION}초 동안 실행합니다..."
    python -m benchmark.nested.run_parent --duration "$DURATION"

    echo "=== 완료. outputs/metrics.csv 확인 ==="
    exit 0
fi

echo "=== 실험 A (Triton 본 측정 경로) ==="
echo "tritonserver 이미지가 로컬에 없다면 최초 실행 시 수 GB를 내려받습니다."

mkdir -p "$REPO_ROOT/outputs"
cd "$REPO_ROOT/triton"
docker compose up --build -d

echo "Triton 준비 대기 중 (http://localhost:8000/v2/health/ready)..."
for i in $(seq 1 60); do
    if curl -sf http://localhost:8000/v2/health/ready > /dev/null 2>&1; then
        echo "Triton 준비 완료"
        break
    fi
    sleep 2
done

if [ "$KEEP_RUNNING" = "1" ]; then
    echo "--keep-running 지정됨 — 컨테이너를 자동 종료하지 않습니다."
    echo "종료하려면: cd triton && docker compose down"
    exit 0
fi

echo "${DURATION}초 동안 실행 후 자동 종료합니다 (Prometheus: http://localhost:9090, 벤치마크 /metrics: :8003)..."
sleep "$DURATION"

echo "=== 컨테이너 로그 (마지막 100줄) ==="
docker compose logs --tail=100 triton

docker compose down
echo "=== 완료. $REPO_ROOT/outputs/metrics.csv 확인 ==="

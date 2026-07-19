# 디렉토리 구조 가이드

각 디렉토리의 목적과 실행 방법을 정리한 문서. 전체 설계 배경은 `plan.md`, 원 시스템 분석은
`research.md` / `proposal.md`를 참고.

---

## 최상위 문서

| 파일 | 의의 |
|---|---|
| `plan.md` | 이 레포의 실험 설계·Phase별 진행 계획 (가장 먼저 읽을 문서) |
| `research.md` | 원 CCTV 시스템(nested Ray)의 12개 문제 분석 |
| `proposal.md` | Standalone Ray 재설계의 타당성 검토 |
| `README.md` | 레포 소개 (최종 산출물, 완성 후 갱신) |
| `requirements.txt` | 고정 버전 의존성 |
| `config/default.yaml` | **모든 설정의 단일 소스**. 카메라 수, 프레임 크기, object_store_memory 등 하드코딩 금지 원칙에 따라 이 파일만 수정하면 됨 |

---

## `benchmark/` — 실험 코드

두 실험(A/B)이 아래 `common/`을 동일하게 공유해 "구조 차이 외 변수"를 제거한다.

### `benchmark/common/`
실험 A·B가 공통으로 쓰는 모듈. 여기 코드를 수정하면 두 실험 모두에 영향을 준다.

- `config.py`: `config/default.yaml`을 dataclass로 로드
- `frame_generator.py`: 1280×720×3 합성 프레임을 시드 고정 풀에서 생성 (생성기 자체가 병목이 되지 않도록 미리 만들어 둔 풀에서 복사)
- `stages.py`: 파이프라인 스테이지(gRPC 클라이언트로 detect/track/pose/batch_violence/falldown 호출) + Recorder 두 가지 구현(`FrameRecorder`=np.append, `FrameRecorderDeque`=deque)
- `mock_latency.py`: 지터 적용 sleep + 크기 비례 busy-wait. `inference_mock/server.py`(커스텀 gRPC)와 `triton/models/*/1/model.py`(Triton python backend)가 동일 로직을 공유해 실험 A/B의 "구조 차이 외 변수"를 제거한다
- `metrics.py`: Prometheus 메트릭 노출 + CSV 동시 덤프. `MetricsActor`는 여러 프로세스(Analysis Actor, violence/falldown 태스크)가 하나의 Prometheus 서버·CSV 파일에 안전하게 기록하도록 감싼 Ray Actor
- `smoke.py`: 위 모듈들이 맞물려 동작하는지 확인하는 통합 테스트

### `benchmark/nested/` (Phase 3, 구현 완료 — 작업자 재현 확인 대기)
실험 A. 원 시스템의 Stub+Ray 중첩 구조를 재현.

- `inference_backend.py`: `InferenceBackend` 프로토콜 + `TritonInferenceBackend`(같은 Triton의 모의 모델을 tritonclient.grpc로 호출, 본 측정 경로) + `MockGrpcInferenceBackend`(inference_mock 커스텀 gRPC 서버 호출, fallback 경로)
- `actors.py`: `StreamActor`(내부 스레드가 무조건 `ray.put()`, 문제 4) + `AnalysisActor`(sync Actor의 단일 메서드가 `while True` 무한 루프를 도는 문제 8 재현. Pull 패턴 2회 왕복 — 문제 3. `np.object_`+`np.append` recorder — 문제 6/7) + `utility_task`/`violence_task`/`falldown_task`(fire-and-forget Ray 태스크)
- `run_parent.py`: Docker 불가 환경을 위한 fallback. 부모 프로세스에서 nested `ray.init()` + 더미 CPU 부하 스레드로 Stub을 근사. **본 측정 경로가 아니다.**

본 측정 경로는 `triton/` 디렉토리(아래) — 실제 tritonserver의 python backend(`pipeline` 모델) 안에서 위 Actor들이 기동된다.

### `benchmark/standalone/` (Phase 4, 구현 예정)
실험 B. 외부 Ray 클러스터(`ray start --head`)에 접속하는 재설계 구조. 개선 항목 5종을 플래그로 on/off.

### `benchmark/micro/` (Phase 5, 구현 예정)
마이크로벤치 3종: SerDes 비용, ObjectRef 왕복 비용, ray.put() 빈도별 스필링.

---

## `inference_mock/` — 모의 추론 서버 (fallback 경로 전용)

실제 GPU 모델 없이 gRPC 인터페이스만 동일하게 흉내 내는 서버. `benchmark/nested/run_parent.py`
fallback 경로와 실험 B(Phase 4)가 사용한다. 실험 A 본 측정 경로는 `triton/` 모델을 쓴다.

- `inference.proto`: Detect/Track/Pose/BatchViolence/Falldown gRPC 서비스 정의
- `server.py`: 설정된 지연시간(`config`) + 입력 크기 비례 CPU 부하(busy-wait)로 응답. 동시성은
  세마포어(`max_concurrency`)로 제한해 실제 추론 서버의 GPU 직렬화 큐잉을 모사한다

**실행**:
```bash
python -m inference_mock.server            # 기본 포트(config 참조)
python -m inference_mock.server 50052      # 포트 지정
```

Proto는 커밋되어 있지 않은 `*_pb2*.py` 생성 코드가 필요하므로 최초 1회 컴파일 필요:
```bash
bash scripts/gen_proto.sh
```

---

## `config/` — 설정

`default.yaml` 하나로 카메라 수, 프레임 크기, FPS, object_store_memory, 모의 추론 지연 등을
전부 관리. 코드에 하드코딩된 값이 있으면 버그로 간주.

---

## `triton/` — 실험 A 본 측정 경로

실제 tritonserver(CPU-only) 컨테이너의 python backend 안에 nested Ray를 기동한다
(plan.md §3.2, 2026-07-06 확정). `benchmark/nested/actors.py`의 Actor를 그대로 재사용한다.

- `models/pipeline/`: 원본 Stub 역할. `1/model.py`의 `initialize()`에서 nested `ray.init()` +
  카메라별 StreamActor/AnalysisActor를 fire-and-forget으로 기동한다. `execute()`는 헬스체크
  이상의 역할이 없다 — 실제 파이프라인은 백그라운드 Ray Actor가 요청과 무관하게 상시 동작
- `models/{detect,track,pose,violence,falldown}/`: 모의 추론 모델. `benchmark/common/mock_latency.py`를
  공유해 `inference_mock/server.py`(fallback 경로)와 동일한 지연 기준값을 낸다. `instance_group`
  count가 `max_concurrency`와 같아 세마포어 없이 Triton 자체 동시성 제어로 GPU 직렬화 큐잉을 모사
- `Dockerfile`: `nvcr.io/nvidia/tritonserver:24.08-py3` 기반. `triton/requirements.txt`(tritonclient
  포함, protobuf>=6) 설치 — 루트 `requirements.txt`(protobuf==5.29.2, grpcio-tools 호환)와 의존성
  충돌이 나므로 분리했다
- `docker-compose.yml`: 8000(HTTP)/8001(gRPC)/8002(Triton 메트릭)/8003(benchmark 앱 `/metrics`)
  노출, `outputs/`를 호스트에 바인드 마운트

**실행**: `bash scripts/run_nested.sh` (아래 `scripts/` 참고). 최초 실행 시 tritonserver
이미지 pull에 수 GB·수 분이 걸린다.

---

## `monitoring/` — Prometheus + Grafana

측정 지표를 실시간으로 보기 위한 스택. CSV 덤프와 별개로 동작(Grafana 없이도 `analysis/`
스크립트로 그래프 재현 가능).

- `docker-compose.yml`: Prometheus(9090) + Grafana(3000) 컨테이너
- `prometheus.yml`: `localhost:8003`(벤치마크 앱의 `/metrics` 엔드포인트 — Triton 자체
  HTTP/gRPC/메트릭 포트인 8000/8001/8002와 겹치지 않도록 분리)를 스크레이프
- `datasources/`, `dashboards/`: Grafana 프로비저닝 설정과 대시보드 JSON

**실행**:
```bash
bash scripts/start_monitoring.sh
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000 (admin/admin)

# 종료
cd monitoring && docker-compose down
```

---

## `scripts/` — 원커맨드 실행 스크립트

| 스크립트 | 역할 |
|---|---|
| `smoke.sh` | venv 생성 → 의존성 설치 → proto 컴파일 → 통합 smoke test 실행 |
| `gen_proto.sh` | `inference.proto` → `*_pb2.py`/`*_pb2_grpc.py` 생성 |
| `start_monitoring.sh` | Prometheus/Grafana docker-compose 기동 |
| `run_nested.sh` | 실험 A 원커맨드 실행. 기본값은 Triton 경로(`triton/docker-compose.yml`), `--fallback`은 Docker 불가 환경용 parent-process 경로(`benchmark/nested/run_parent.py`), `--keep-running`은 자동 종료 없이 Grafana로 관찰 |
| `run_standalone.sh` (Phase 4 예정) | 실험 B 원커맨드 실행 |
| `run_micro.sh` (Phase 5 예정) | 마이크로벤치 3종 실행 |

**최초 실행 순서**:
```bash
bash scripts/smoke.sh              # 뼈대 동작 확인
bash scripts/start_monitoring.sh   # 모니터링 스택 기동 (선택)
```

---

## `analysis/` — 결과 분석

`plot_comparison.py`: `docs/data/`의 실측 CSV 3종 → `docs/img/` 비교 그래프 PNG 5종.
`python -m analysis.plot_comparison`으로 재생산 가능. 하드코딩 수치 없음.

## `docs/` — 최종 문서

| 파일 | 내용 |
|---|---|
| `01-problem-analysis.md` | 원 시스템 구조와 12개 문제 요약 (비식별화) |
| `02-experiment-design.md` | A/B 실험 설계, 개선 플래그 6종, 스케일다운·재현 튜닝 근거 |
| `03-results.md` | 실험 결과 (수치는 작업자 실측 기입, Claude는 자리표시자만) |
| `data/` | 판정 근거 실측 CSV (A / B0 / B-all) |
| `img/` | analysis 스크립트가 생성한 비교 그래프 PNG |

---

## 진행 상태 요약

- ✅ Phase 1: 뼈대 (`benchmark/common`, `inference_mock`, `config`)
- ✅ Phase 2: 계측 (`monitoring/`, `metrics.py`)
- ✅ Phase 3: 실험 A (`benchmark/nested/`, `triton/`) — 현상 2·3 재현, 현상 1은 부분 재현(상승 시작 + 계측 붕괴)
- ✅ Phase 4: 실험 B (`benchmark/standalone/`) — 개선 플래그 6종 토글
- ⬜ Phase 5: 마이크로벤치 (`benchmark/micro/`) — Future Work
- ✅ Phase 6: 분석·문서화 (`analysis/`, `docs/`, `README.md`) — 수치는 작업자 실측 기입 대기

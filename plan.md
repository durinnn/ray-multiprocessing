# plan.md — Ray Nested vs Standalone 벤치마크 구현 계획

> 근거 문서: `CLAUDE.md`(지침), `research.md`(원 시스템 12개 문제 분석), `proposal.md`(재설계 타당성 검토).
> 이 문서는 세 문서를 합쳐 "이 레포에서 무엇을, 어떤 순서로, 어떻게 구현·측정할지"를 확정한다.

---

## 1. 목표 재확인

원 시스템(Triton Python Backend Stub 내부에 Ray를 nested 기동한 CCTV 분석 파이프라인)에서
관찰된 성능 문제를 **회사 코드 없이 합성 워크로드로 최소 재현**하고,
**Standalone Ray + 직접 gRPC 재설계의 개선 효과를 수치로 분해 검증**한다.

재현 목표 3대 현상 (실험 A에서 반드시 재현):

1. 부하(Stress) 상승 시 end-to-end P99 지속 상승
2. 분석 Actor **단독** CPU의 주기적 0% 낙하 (전체 CPU는 높게 유지되는 동안)
3. Ray Object Spilling 발생

---

## 2. 가설 매핑 — research.md 12개 문제 → 이 레포의 검증 대상

research.md의 문제 번호를 CLAUDE.md의 가설 체계([1]~[4], A~E)로 매핑하고,
이 레포에서 **재현할 것 / 하지 않을 것**을 확정한다.

| research.md | CLAUDE.md 가설 | 이 레포에서의 취급 | 검증 위치 |
|---|---|---|---|
| 문제 1 (Stub+Ray 중첩: CPU 경합·스케줄링·캐시·이중복사) | [1][2][3][4] | 구조 자체를 재현 — 실제 tritonserver의 python backend 안에서 `ray.init()` (§3.2, 2026-07-06 확정) | 실험 A vs B |
| 문제 2 (Spilling → Thrashing → Actor 기아) | C | object_store_memory를 의도적으로 작게 설정해 재현 (스케일다운, README 명시) | 실험 A + 마이크로벤치 3 |
| 문제 3 (매 프레임 ray.get() Pull IPC) | B | Pull 패턴 재현, B에서 push/queue로 개선 | 실험 A/B + 마이크로벤치 2 |
| 문제 4 (버려지는 프레임도 ray.put()) | D | 무조건 put 재현, B에서 조건부 put 플래그 | 실험 A/B + 마이크로벤치 3 |
| 문제 5 (직렬 동기 gRPC 3회) | — | 구조만 동일하게 재현(공정 비교 조건). 배치 최적화(Opt 3)는 **범위 밖** | — |
| 문제 6 (np.append O(N) 재할당) | — (E의 보조) | 실험 A의 recorder를 np.append로 구현해 부하 요인으로 포함. B에서 deque(maxlen) 플래그 | 실험 A/B |
| 문제 7 (np.object_ SerDes → 대용량 pickle) | A | np.object_ recorder 재현, 이벤트 시 전체 반환 | 실험 A + 마이크로벤치 1 |
| 문제 8 (무한루프 메서드 → 태스크 기아) | E | sync Actor 무한루프 재현, B에서 async Actor 플래그 | 실험 A/B |
| 문제 9~12 (dead code 등 코드 품질) | — | **범위 밖** (원 코드 고유 문제, 합성 재현 가치 없음) | — |

**범위 밖 결정 근거** (2026-07-04 작업자 확정):
- 문제 5의 배치 최적화(Opt 3)는 "nested vs standalone 구조 비교"라는 이 레포의 주제가 아니라
  추가 처리량 최적화다. CLAUDE.md의 과설계 금지 원칙에 따라 제외하고 README Future Work에 기재.
- proposal.md Track 2(asyncio/raw multiprocessing 비교 PoC)도 동일 이유로 제외, Future Work 기재.
- 문제 9~12는 코드 품질 이슈로 성능 벤치마크와 무관.

---

## 3. 실험 설계

### 3.1 공통 워크로드 (두 실험 완전 동일)

```
합성 프레임 생성기 (카메라당 1개)
  └─ 1280×720×3 uint8, 20fps 페이싱, seed 고정
     (풀에서 복사 방식 — 매번 full random 생성 시 생성기 자체가 병목이 되므로)

파이프라인 (프레임당, 문제가 된 원본 신규 구조의 실행 순서 그대로)
  detect(gRPC) → track(gRPC, 트래커 상태 blob을 np.object_로 왕복 — 가설 A의 살아있는 예)
  → [stress] recorder 누적 → pose(gRPC)
  → 침입/배회 (gRPC 아님 — Actor 내 로컬 ~0.2ms 연산. 원본이 shapely·타이머 규칙 기반이므로)
  → [stress] 프레임 12장 누적 시 fire-and-forget 추론 태스크 기동:
      폭행   = 동기 gRPC 2회 직렬 (2모델 앙상블 재현)
      쓰러짐 = 동기 gRPC ≤12회 직렬 (프레임별 호출 + 상태 blob 체이닝 재현)

mock 지연 기준값 (config에 기록. 원 시스템 실측 로그 p50에서 보정한 근사치 — README에 출처 명시)
  detect 15ms / track 8ms / pose 25ms / 폭행 35ms×2회 / 쓰러짐 8ms×N회 / 침입·배회 0.2ms(로컬)
  모든 호출에 ±10% uniform 지터(카메라별 시드 고정). mock 서버는 동시성 제한(세마포어)으로
  실제 추론 서버의 GPU 직렬화가 만드는 큐잉 지연을 모사한다.

모드
  idle   : detect → track 만. 경량 확인용
  stress : + pose + recorder 누적 + 프레임 누적 추론 + 주기적 이벤트(= recorder 전체 회수)
```

### 3.2 실험 A — nested 재현 (`benchmark/nested/` + `triton/`)

(2026-07-06 작업자 확정) 부모 프로세스로 Stub을 "모사"하는 대신 **실제 tritonserver(CPU-only)
컨테이너의 python backend 안에서 nested 기동**한다. Triton이 실제로 만드는 프로세스/스레드
환경을 근사치로 뭉개지 않기 위해서다. mock 추론 모델도 같은 Triton에 올려, 원본의
"Stub 안 Ray의 Actor가 자기가 사는 Triton으로 되돌아오는 gRPC" 루프까지 재현한다.

```
tritonserver 컨테이너 (CPU-only)
 ├─ pipeline 모델 (python backend, 원본 Stub 역할)
 │   ├─ initialize()에서 ray.init(object_store_memory=작게)   ← nested 기동
 │   └─ 카메라당:
 │       ├─ StreamActor: 프레임 생성 → 무조건 ray.put() → deque 보관 (문제 4)
 │       └─ AnalysisActor(sync): while 무한루프 (문제 8)
 │            ├─ ray.get(stream.get_frame.remote())     ← Pull 2왕복 (문제 3)
 │            ├─ §3.1 파이프라인 수행 (gRPC 대상 = 같은 Triton의 mock 모델)
 │            ├─ recorder: np.object_ 배열에 np.append (문제 6, 7)
 │            └─ 이벤트 시: utility task가 get_recorder.remote() 호출
 │               → 태스크 기아 + 대용량 pickle SerDes (문제 7, 8)
 └─ mock 추론 모델들 (python backend): detect/track/pose/폭행/쓰러짐
     — sleep + busy-wait만 수행 (§3.1의 지연 기준값)
```

Docker를 쓸 수 없는 환경을 위해 기존 설계(부모 프로세스가 더미 CPU 부하로 Stub을 모사,
`parent` config)를 **fallback 플래그로 유지**한다. 본 측정은 Triton 경로로 한다.

#### 3.2.1 구현 상세 (2026-07-06 Phase 3 구현 완료, feat/phase3-nested)

**Actor 코드는 실험 A/B가 공유** — `benchmark/nested/actors.py`에 원 시스템 구조를 재현했고,
Triton 경로와 fallback 경로 모두 이 코드를 그대로 쓴다 (호출 대상 gRPC 서버만 다르다).

- `StreamActor`: `start_streaming()` 메서드는 내부에서 `threading.Thread`를 띄우고 **즉시
  반환**한다. 이 스레드가 `deque(maxlen=15)`(원본 FRAME_QUEUE_MAX_SIZE와 동일)에 프레임을
  채우며 매번 무조건 `ray.put()`한다 (문제 4). 메서드가 즉시 반환해야 Actor의 디스패치
  스레드가 `get_frame()` 호출에 계속 응답할 수 있다 — 원본이 스트리밍 루프를 별도 스레드로
  돌린 이유와 동일.
- `AnalysisActor.start_analysis()`: `while True`로 절대 반환하지 않는 단일 메서드다 (문제 8).
  매 프레임 `ray.get(stream_actor.get_frame.remote())`로 프레임을 당겨온다 (Raylet 왕복 2회,
  문제 3). recorder는 `benchmark.common.stages.FrameRecorder`(np.object_ + np.append, 문제
  6/7)를 그대로 재사용한다. 침입/배회는 gRPC가 아닌 `mock_latency.busy_wait_ms()`로 로컬
  처리한다. `batch_frame_count`(12)장이 쌓이면 `violence_task`/`falldown_task`를
  fire-and-forget으로 던진다 — 이 둘은 **plain `@ray.remote` 함수**(Actor 메서드가 아님)라서
  AnalysisActor를 막지 않고 별도 워커에서 독립 실행된다.
  - `event_trigger_interval`(5초)마다 `utility_task.remote(self_handle, ...)`를 fire-and-forget
    으로 호출한다. 이 태스크는 `ray.get(analysis_actor.get_recorder.remote())`를 실행하는데,
    `get_recorder()`는 AnalysisActor의 **같은 단일 디스패치 스레드** 큐에 들어가고 그 앞의
    `start_analysis()`가 절대 반환하지 않으므로 **영원히 실행되지 못하고 hang**한다. 별도의
    graceful stop 로직을 만들지 않았다 — hang 자체가 재현 목표이기 때문이며, 종료는
    `ray.shutdown()`(fallback) 또는 컨테이너 종료(Triton)로만 가능하다.
  - 로컬 fallback 테스트(1대 카메라, 30초, stress 모드)로 `utility_task`가 실제로
    "hang 예상" 로그를 찍고 이후 완료 로그가 전혀 나오지 않는 것까지 확인했다.

**추론 백엔드는 인터페이스 하나로 두 경로를 추상화** —
`benchmark/nested/inference_backend.py`의 `InferenceBackend` 프로토콜을 두 곳에서 구현한다.
AnalysisActor는 `backend_kind`("triton" | "mock_grpc") 문자열만 받고, Actor/Task 각자의
프로세스 안에서 `create_inference_backend()`로 새로 생성한다 (gRPC 채널 객체는 Ray 프로세스
경계를 넘어 피클링하기 까다로워서, 넘기지 않고 각자 새로 만드는 방식을 택함).
- `TritonInferenceBackend`: `tritonclient.grpc`로 같은 Triton 서버의 모의 모델(detect/track/
  pose/violence/falldown)을 호출한다. 실제 bbox/keypoint 좌표는 이 벤치마크의 관심사가
  아니므로, Triton IO는 **개수(count)만 왕복**하도록 단순화했다 (구조적 병목 재현에는
  SerDes 크기·왕복 횟수·busy-wait 시간만 필요).
- `MockGrpcInferenceBackend`: 기존 `benchmark.common.stages.InferenceClient`를 감싸 fallback
  경로에서 쓴다.
- 폭행(violence)은 backend의 `violence()` 메서드 내부에서 `violence_num_calls`(2)회 직렬
  호출로 2모델 앙상블을 재현하고, 쓰러짐(falldown)은 `falldown()` 내부에서 프레임별로 최대
  `batch_frame_count`(12)회 직렬 호출하며 매 호출의 응답 `state_blob`을 다음 요청에 실어
  보내 상태 체이닝을 재현한다.

**Triton 모델 저장소** (`triton/models/`) — `pipeline` 모델의 `initialize()`가 nested
`ray.init()`을 호출하고 카메라 수만큼 StreamActor/AnalysisActor를 fire-and-forget으로
띄운 뒤 반환한다 (문제 1의 재현 대상 그 자체). `execute()`는 헬스체크 텐서 하나만 반환할
뿐 실제 파이프라인과 무관하다 — 원본 Stub도 별도 요청 트리거 없이 상시 동작하는 구조였기
때문이다. `finalize()`에서 `ray.shutdown()`을 호출한다. detect/track/pose/violence/falldown
5개 모의 모델은 `benchmark/common/mock_latency.py`(지터 적용 sleep + 크기 비례 busy-wait)를
그대로 import해서 쓰므로 fallback의 `inference_mock/server.py`와 동일한 지연 기준값을
낸다. `instance_group`의 `count`를 config의 `max_concurrency`(2)와 맞춰, 세마포어 코드
없이 Triton 자체 동시성 제어로 GPU 직렬화 큐잉을 모사한다.

**공유 메트릭 액터** — 여러 프로세스(AnalysisActor, violence/falldown 태스크)가 하나의
Prometheus 서버·CSV 파일에 동시에 쓸 수 없으므로, `benchmark/common/metrics.py`에
`MetricsActor`(Ray Actor로 `MetricsCollector`를 감쌈)를 추가해 중앙화했다. 각 프로세스는
`metrics_actor.record_xxx.remote(...)`로 fire-and-forget 기록만 위임한다.

**구현 중 발견해서 함께 고친 버그** (모두 Phase 1/2부터 있던 것):
- `inference_mock/server.py`의 서비서 메서드명이 소문자(`detect`)였는데 proto의 RPC명은
  PascalCase(`Detect`)라 `add_InferenceServiceServicer_to_server()`가 즉시 `AttributeError`로
  죽었다 — gRPC 서버가 한 번도 정상 기동한 적이 없었다는 뜻. 전부 PascalCase로 수정.
- `cpu_work_detect/track/pose` 계수가 1280x720 프레임 기준 1000배 과다해서(예:
  `cpu_work_detect: 0.001`ms/byte → 프레임당 약 2.76**초** busy-wait) 실제로 호출해보고서야
  드러났다. 프레임당 수 ms 수준으로 재보정.
- `MetricsCollector.record_frame_latency()`가 히스토그램·인메모리 리스트만 갱신하고
  CSV에는 전혀 쓰지 않고 있었다 (smoke.py가 별도로 `log_csv_row()`를 중복 호출해야만
  CSV에 남는 구조). `record_frame_latency()` 자체가 CSV도 쓰도록 수정 — 안 고쳤으면
  `analysis/`(Phase 6)가 읽을 스테이지별 latency 데이터가 전부 비어 있었을 것.
- 폭행 배치(12프레임 × 2.76MB ≈ 33MB)가 gRPC 기본 메시지 한도(4MB)를 넘어 RESOURCE_EXHAUSTED로
  실패 — 클라이언트/서버 양쪽에 `grpc.max_{send,receive}_message_length` 옵션(64MB) 추가.
- `.gitignore`의 `*.pb2.py`/`*.pb2_grpc.py` 패턴이 실제 생성 파일명(`inference_pb2.py`,
  점 없음)과 안 맞아 전혀 무시되지 않고 있었다 — `*_pb2.py`/`*_pb2_grpc.py`로 수정.
- `tritonclient[grpc]`가 `protobuf>=6.30`을 요구하는데 루트 `requirements.txt`의
  `protobuf==5.29.2`(grpcio-tools 호환용)와 충돌한다 — 두 패키지가 같은 프로세스에서 쓰일
  일이 없으므로(`tritonclient`는 Triton 컨테이너 전용, `grpcio-tools`는 fallback 경로의 proto
  컴파일 전용) `triton/requirements.txt`로 분리.
- 벤치마크 앱의 Prometheus 포트(기본 8000)가 같은 컨테이너 안 Triton 자체 HTTP 포트(8000)와
  겹쳐서 `MetricsActor`가 뜰 수 없었다 — 8003으로 이동, `monitoring/prometheus.yml`도 갱신.

**fallback 경로 실행 검증**: `bash scripts/run_nested.sh --fallback`으로 로컬(Docker 없이)
end-to-end 실행 — smoke test 통과, detect(~17ms)/track(~9ms)/pose(~28ms)/violence(~2회
직렬 ~70ms)/falldown(~12회 직렬 ~100ms)이 config 기준값 근처로 CSV에 기록됨을 확인. Triton
경로는 `nvcr.io/nvidia/tritonserver` 이미지가 수 GB라 코드만 준비하고 실제 실행/3대 현상
재현 확인은 작업자 몫으로 남겼다 (본 문서 §7 Phase 게이트 규칙에 따름).

### 3.3 실험 B — standalone 재설계 (`benchmark/standalone/`)

`ray start --head`로 독립 기동된 클러스터에 접속. Triton 밖에서 실행되며,
gRPC 호출 대상은 실험 A와 **완전히 동일한 Triton의 mock 모델**이다 (추론 서버 변수 제거).
개선 항목은 **개별 플래그**로 on/off (기여 분해 측정):

| 플래그 | 대상 문제 | off일 때 (A와 동일 동작) | on일 때 |
|---|---|---|---|
| `use_deque_recorder` | 6 | np.append recorder | deque(maxlen) |
| `use_objectref_recorder` | 7 | 프레임 원본 저장·전체 pickle | ObjectRef만 저장 |
| `use_conditional_put` | 4 | 무조건 ray.put | 큐 여유 있을 때만 put |
| `use_async_actor` | 8 | sync 무한루프 | async Actor + await |
| `explicit_object_store` | 2 | 기본값 | object_store_memory 명시 |

여기에 **A/B 공용 플래그** 하나를 둔다 (원본에 어피니티 설정이 전혀 없었으므로 기본 off):

| 공용 플래그 | 대상 가설 | off일 때 (원본과 동일) | on일 때 |
|---|---|---|---|
| `set_cpu_affinity` | [3] | 어피니티 미설정 | psutil로 프로세스별 코어 고정 |

측정 매트릭스: `A` / `A+affinity` / `B0(전부 off)` / `B(하나씩 on)` / `B(전부 on)`.
- B0와 A의 차이가 곧 "구조 전환([1]~[4]) 단독 효과"다.
- A+affinity와 A의 차이가 가설 [3]의 단독 검증이며, "설정만 고치면 되는 것 아니냐"는
  반론에 대한 데이터가 된다.

### 3.4 마이크로벤치 (`benchmark/micro/`) — 가설 A, B, D 직접 검증

1. **serdes**: np.object_ 배열 vs 고정 dtype 배열의 Plasma put/get 소요시간, zero-copy 여부
2. **roundtrip**: Actor 메서드의 값 직접 반환 vs ObjectRef 반환 왕복 비용
3. **putrate**: ray.put() 호출 빈도별 Object Store 사용량·spilling 발생 추이

---

## 4. 측정 설계

| 지표 | 수집 방법 |
|---|---|
| 스테이지별·e2e 지연 P50/P99 | prometheus_client Histogram + CSV 원본 덤프 |
| 프로세스별 CPU (분석 Actor 단독 분리) | psutil로 PID별 샘플링 → Gauge + CSV. Actor가 자기 PID를 등록 |
| 메모리 (RSS 합산, Object Store 사용량) | psutil + `ray.available_resources()`/내부 메트릭 |
| Spilling 발생량 | Ray spill 로그 파싱 + `ray memory` 통계 |

- 계측 코드는 `benchmark/common/metrics.py` 하나로 통일해 두 실험이 동일 계측을 쓴다.
- Prometheus + Grafana는 `monitoring/docker-compose.yml`. CSV는 항상 병행 덤프
  (Grafana 없이도 `analysis/` 스크립트만으로 그래프 재현 가능해야 함).
- **결과 수치는 전부 작업자 실측. 이 계획의 어떤 문서에도 예상 수치를 쓰지 않는다.**

## 5. 스케일다운 파라미터 (로컬: 6코어 / 11GB WSL2)

| 항목 | 원 시스템 | 이 레포 기본값 | 근거 |
|---|---|---|---|
| 카메라 수 | 다수 | 4 (config로 조정) | 6코어에서 경합이 관찰되는 최소 규모 |
| 프레임 | 1280×720×3 | 동일 | 프레임당 2.76MB라는 SerDes 스토리 유지 |
| FPS | 20 | 동일 | |
| recorder 길이 | 600프레임 | 200 (config) | 메모리 11GB 내에서 SerDes 현상 재현 |
| object_store_memory | 미설정(기본) | 실험 A: 작게(예: 512MB)로 **의도적 설정** | Spilling 유도. README에 명시 |

정확한 값은 Phase 3 재현 튜닝에서 확정하고 config에만 기록한다.
측정 환경은 **로컬 WSL2 단일 머신으로 확정** (2026-07-04 작업자 확정). EC2 측정은 하지 않으며,
README Limitations에 "단일 머신 스케일다운 재현"임을 명시한다.

---

## 6. 레포 구조 (CLAUDE.md 2장 그대로 + 소폭 구체화)

```
├── plan.md                    # 이 문서
├── config/default.yaml        # 유일한 설정 소스 (하드코딩 금지)
├── benchmark/
│   ├── common/                # config 로더, 프레임 생성기, 스테이지, metrics, smoke
│   ├── nested/                # 실험 A (Actor/파이프라인 코드. Triton pipeline backend가 import)
│   ├── standalone/            # 실험 B
│   └── micro/                 # 마이크로벤치 3종
├── triton/                    # Triton 모델 저장소: pipeline backend + mock 추론 모델들 (Phase 3)
├── inference_mock/            # 경량 fallback: 커스텀 gRPC mock 서버 (Docker 불가 환경용)
├── monitoring/                # prometheus + grafana compose
├── scripts/                   # run_nested.sh / run_standalone.sh / run_micro.sh / gen_proto.sh
├── analysis/                  # CSV → 그래프
└── docs/                      # 01-problem / 02-hypotheses / 03-results
```

---

## 7. 진행 단계와 브랜치 전략

(2026-07-04 작업자 확정) 로컬 단독 작업이므로 PR 대신
**feature 브랜치 → `git merge --no-ff` → main**으로 PR 머지 이력을 모사한다.
추후 GitHub push 시 머지 이력이 그대로 보존된다. main 직접 커밋은 초기 문서 커밋 이후 금지.
커밋은 conventional commits, Phase = 브랜치 = 머지 단위. 머지 커밋 본문에
"이 Phase가 검증하는 가설"을 한 줄로 기재한다(PR 본문 규칙의 로컬 대응).

진행 게이트(작업자 확정): **Phase 1과 2는 묶어서 연속 진행** 후 멈추고 체크리스트 제시.
인프라 성격이라 Claude가 직접 실행 검증 가능하기 때문. **Phase 3부터는 매 Phase마다
작업자 확인(특히 Phase 3의 3대 현상 재현 확인) 전에는 다음 Phase로 넘어가지 않는다.**

| Phase | 브랜치 | 내용 | 완료 조건 (작업자 확인 후 다음 진행) |
|---|---|---|---|
| 1 | `feat/phase1-skeleton` | config 체계, 프레임 생성기, 파이프라인 스테이지, inference_mock gRPC 서버, smoke 스크립트 | smoke 원커맨드 통과, black/flake8 클린 |
| 2 | `feat/phase2-metrics` | metrics.py, CSV 덤프, monitoring compose, Grafana 대시보드 | Grafana에서 smoke 지표 확인 |
| 3 | `feat/phase3-nested` | 실험 A 전체 — tritonserver(CPU) 컨테이너, pipeline python backend(`ray.init`), mock 추론 모델 5종, fallback 부모 프로세스 모드. **구현 완료(2026-07-06, §3.2.1 참고), 커밋 완료·main 미병합** | **3대 현상 재현 확인** (안 되면 파라미터 튜닝 반복) — 작업자 확인 대기 중 |
| 4 | `feat/phase4-standalone` | 실험 B + 개선 플래그 5종 + 공용 어피니티 플래그 | 매트릭스(A+affinity 포함) 실행 가능 |
| 5 | `feat/phase5-micro` | 마이크로벤치 3종 | 3종 결과 CSV 생성 확인 |
| 6 | `feat/phase6-docs` | analysis 그래프, docs, README (수치는 자리표시자) | README 5장 규칙 충족 |

CLAUDE.md 6장 규칙에 따라 **각 Phase 완료 시 작업자 확인 체크리스트를 제시하고 멈춘다.**

---

## 8. 기술 결정 사항

- Python 3.11.9 (로컬 pyenv), 의존성은 `.venv` + `requirements.txt` 고정 버전 (Ray 버전 설치 후 확정 기입)
- gRPC 생성 코드(`*_pb2*.py`)는 커밋하되 lint 제외, `scripts/gen_proto.sh`로 재생성 가능
- 로컬 CPU 후처리 모의는 busy-wait(perf_counter 기반) — 실제 연산 대신 시간 기준으로 제어해
  머신 차이에 관계없이 재현 가능하게 함
- 프레임 생성은 시드 고정 풀에서 복사 — 생성기 자체가 병목이 되는 것을 방지
- 스테이지 코드/계측 코드는 실험 A/B가 동일 모듈을 공유 — 구조 차이 외 변수 제거
- mock 지연 기준값은 원 시스템 실측 로그의 스테이지별 p50에서 보정한 근사치이며,
  부하 시의 꼬리 지연은 지터로 주입하지 않는다 — 그건 실험에서 경합으로 발생해야 할 결과다
- 침입/배회는 원본이 모델이 아닌 규칙 기반 로컬 연산(shapely 폴리곤 판정·체류 타이머)이므로
  gRPC가 아닌 Actor 내 로컬 busy-wait(~0.2ms)로 모의한다
- 폭행 추론은 동일 입력을 받는 2개 체크포인트의 앙상블(불일치 시 감점으로 오탐 억제)이므로
  동기 gRPC 2회 직렬로 재현. 쓰러짐 추론은 프레임별 동기 gRPC ≤12회 직렬 + 상태 blob
  체이닝(요청 간 순차 의존)으로 재현
- "dynamic batching" 용어는 코드·문서에서 쓰지 않는다. 원 클라이언트도 직렬 동기 호출·상태
  체이닝 탓에 이를 실질 활용하지 못했고, 추론 배칭 최적화는 스코프 밖이다(README Limitations에
  기재). 프레임 12장 누적 후 추론하는 원본 구조는 "프레임 누적 추론"으로 부른다
- tritonserver CPU-only 컨테이너 채택 비용: 이미지 다운로드 수 GB (README 요구사항에 명시).
  Docker 불가 환경은 `parent` config의 부모 프로세스 모사 모드로 대체 가능하나 본 측정에서 제외

## 9. 확정된 사항 / 미결 사항

확정 (2026-07-04 작업자 답변):
- 범위: Opt 3 배치, Track 2 PoC 모두 제외 → README Future Work 기재
- Git: 로컬 feature 브랜치 + `--no-ff` 머지로 PR 이력 모사
- 게이트: Phase 1+2 연속 진행, Phase 3부터 매 Phase 작업자 확인
- 측정 환경: 로컬 WSL2 단일 머신만
- monitoring: Docker 29.4.0 가용 확인 → docker-compose 방식 그대로 진행

확정 (2026-07-06 작업자 답변, 원 시스템 코드·실측 로그 대조 후):
- 실험 A는 실제 tritonserver python backend 안에서 nested 기동. 부모 프로세스 모사는 fallback으로 강등
- mock 추론은 같은 Triton에 python backend 모델로 서빙, 실험 B도 동일 서버를 호출 (공정성)
- 파이프라인 순서는 원본 신규 코드 기준: detect → track → recorder → pose(조건부) → 침입/배회(로컬) → 누적 추론
- 측정 매트릭스에 A+affinity 추가, `set_cpu_affinity`는 A/B 공용 플래그
- mock 지연 기준값은 원 시스템 로그 보정치(detect 15 / track 8 / pose 25 / 폭행 35×2 /
  쓰러짐 8×N / 침입·배회 0.2 로컬) + ±10% uniform 지터 + mock 동시성 제한

미결 (해당 시점에 질문):
- Phase 3에서 3대 현상이 기본 파라미터로 재현되지 않을 때의 튜닝 방향
  (카메라 수 증가 vs object_store 축소 vs recorder 길이 증가) — 재현 시도 결과와 함께 보고 예정

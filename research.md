# research.md — feat/baseline-nested 코드 전체 분석

> 목적: Terraform으로 AWS 리소스 구성 전, 현재 nested Ray 구조의 코드를 정확히 파악하여
> 배포 대상과 성능 병목 근거를 수립한다.
> README의 계획과 교차 검증 후 누락·오류 항목도 포함한다.

---

## 1. 전체 아키텍처 개요

```
┌──────────────────────────────────────────────────────────────────┐
│  EC2 #1 — Triton Client                                          │
│                                                                  │
│  TritonPythonModel (model.py)                                    │
│   └── Stub Process (Fork-exec) ← Triton이 생성하는 별도 프로세스  │
│        └── ray.init()  ← Stub 안에서 Ray 클러스터 초기화          │
│             ├── Raylet (스케줄러 프로세스)                        │
│             ├── GCS  (메타데이터 프로세스)                        │
│             └── Worker 프로세스들                                 │
│                  ├── StreamActor   (CCTV당 1개)                  │
│                  │    └── threading.Thread: _streaming_loop      │
│                  │         └── cap.read() → ray.put() → deque   │
│                  └── AnalysisActor (CCTV당 1개)                  │
│                       └── start_analysis(): while True          │
│                            └── ray.get(stream_actor.get_frame.remote())
│                            └── process_frame():                  │
│                                 detect(gRPC) → track(gRPC)      │
│                                 → pose(gRPC) → fast events       │
│                                 → batch_inference.remote()       │
│                                 → utility.remote()               │
└──────────────────────────────────────────────────────────────────┘
           │ gRPC :8001
           ▼
┌──────────────────────────────────────────────────────────────────┐
│  EC2 #2 — Triton Inference Server (GPU)                          │
│  detect(YOLO-TRT) / track / pose_ensemble                        │
│  violence(GCN 2종·TSM) / falldown                                │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. 파일별 역할

| 파일 | 역할 |
|------|------|
| `model.py` | Triton Python Backend 진입점. `initialize()`에서 ray.init(), Actor 생성. `execute()`에서 관리 메시지 처리. |
| `actors/stream_actor.py` | RTSP/파일 프레임 수신. threading.Thread 기반 내부 루프. `ray.put(frame)` → deque 저장. `get_frame()`으로 외부 제공. |
| `actors/analysis_actor.py` | 분석 오케스트레이션. `start_analysis()` 무한 루프. 탐지→트래킹→포즈→이벤트 판단. Recorder 관리. |
| `engines/detection_engine.py` | YOLOv9-TRT-FP16 Triton gRPC 동기 추론. |
| `engines/tracking_engine.py` | BYTETrack Triton gRPC 동기 추론. tracker_object 상태를 Actor 내부에 보관. |
| `engines/pose_engine.py` | pose_ensemble Triton gRPC 동기 추론. Violence/Falldown 활성화 시에만 호출. |
| `tasks/fast_inference.py` | `@ray.remote` 정의만 있음. 실제 fast 이벤트는 AnalysisActor 로컬 메서드로 처리됨 → **미사용 dead code**. |
| `tasks/batch_inference.py` | `@ray.remote`. Violence(GCN/TSM), Falldown. Triton gRPC 직접 호출. |
| `tasks/utility.py` | `@ray.remote`. RabbitMQ 발행, MinIO 썸네일/영상 저장. `get_recorder.remote()` 콜백으로 Recorder 획득. |
| `common/constants.py` | 모델명, STREAM_FPS=20, FRAME_QUEUE_MAX_SIZE=15, RECORDER_MAX_FRAMES=600, TRITON_URL 등. |
| `common/models.py` | AnalysisActorState, ViolenceBatchState(GCN 12프레임/TSM 8프레임), FrameQueue. |
| `config.pbtxt` | Triton 모델 설정. `decoupled: True`, instance_group count:1 KIND_CPU. |
| `Dockerfile` | 사내 tritonclient 베이스 이미지. 코드를 Triton model repository 경로에 복사. |

---

## 3. 프레임당 처리 파이프라인

```
start_analysis() while True
│
├─ ray.get(stream_actor.get_frame.remote())      ← IPC 왕복 (요청 + 응답)
│    └─ get_frame() 내부: ray.get(ObjectRef)     ← Plasma Store 읽기
│
└─ process_frame(frame_data)
    ├─ detect(frame)          → Triton gRPC 동기 (blocking)     ~10-20ms
    ├─ (탐지 없으면 early return)
    ├─ track(detection, frame) → Triton gRPC 동기 (blocking)    ~5-10ms
    ├─ (트래킹 없으면 early return)
    ├─ _append_to_recorder()  → np.append() 전체 배열 재할당
    ├─ _maybe_save_daily_thumbnail()
    ├─ _collect_batch_frames()
    │    └─ pose_engine.estimate_pose() → Triton gRPC 동기      ~15-30ms
    │    └─ violence_state 버퍼 / batch_frame_queue 업데이트
    ├─ _start_fast_events()   → Shapely 로컬 처리 (IPC 없음)
    │    └─ detected 시 → utility_task.remote() [fire-and-forget]
    │         └─ ray.get(self.get_recorder.remote()) ← 기아 발생
    └─ _check_batch_processing()
         └─ batch_inference.remote() [fire-and-forget]
              └─ Triton gRPC (Violence/Falldown)
              └─ detected 시 → utility_task.remote()
                   └─ ray.get(analysis_actor_ref.get_recorder.remote()) ← 동일 기아
```

---

## 4. 식별된 12개 문제사항

---

### [A] 구조적 병목

---

#### 문제 1 — Stub Process + Ray 중첩: 4단계 악순환
**심각도: Critical**

**코드 근거**: `model.py:26`
```python
ray.init(ignore_reinit_error=True)
```

Triton Python Backend는 자체적으로 **Stub Process(Fork-exec)**를 생성해 그 안에서 분석 로직을 실행한다. 이 Stub 위에 Ray를 초기화하면 동일 머신에 추가 프로세스(Raylet, GCS, Worker)가 올라가면서 4단계 악순환이 발생한다.

```
[1] CPU 자원 경합
    Stub Process / Raylet / GCS / Worker 프로세스들이
    동일 CPU 코어를 두고 OS 스케줄러 수준에서 경합

[2] OS 스케줄링 지연
    프로세스 간 컨텍스트 스위칭 비용 누적
    Ready Queue에 적체 → 실제 처리 지연

[3] L1/L2 캐시 오염
    프로세스 어피니티(CPU Affinity) 미설정 상태에서
    프로세스 전환마다 캐시 라인이 오염 → 캐시 미스 패널티 누적
    코드 어디에도 affinity 설정 없음

[4] 메모리 이중 복사
    Triton 내부: Shared Memory(SHM) 경유 데이터 전달
    Ray: Plasma Object Store 경유 데이터 전달
    SHM → Plasma Store 복사가 이중으로 발생
    CPU 포화 상태에서 복사 비용 증폭
```

---

#### 문제 2 — Object Spilling → Memory Thrashing → AnalysisActor 기아
**심각도: Critical**

**코드 근거**: `ray.init()` 호출 시 Object Store 메모리 설정 없음. `requirements.txt`에서 `ray[default]>=2.0.0` — Spill 정책 기본값 적용.

Object Store 압박이 누적되면 Raylet이 **Object Spilling(디스크 스왑)**을 강행한다.

```
메모리 상승 경로:
  np.object_ Recorder(1.55GB) → batch_inference 프레임 버퍼
  → Plasma Store 포화 → Raylet이 Spill 시작

Thrashing 메커니즘:
  Raylet: 스왑 I/O 주도 → CPU 과점
  AnalysisActor: 메모리 공간 할당 못 받음 → I/O Wait (Blocking) → CPU 0%

결과:
  Raylet은 스왑으로 바쁘고, AnalysisActor는 강제 대기
  실측: 메모리 4GB 안정 → 9GB → 17GB+ 지속 상승
        AnalysisActor CPU가 주기적으로 0%로 떨어지는 패턴
```

ray.init()에 `object_store_memory` 제한 설정이 없어 Spill 임계값 제어 불가.

---

### [B] 프레임 처리 경로 병목

---

#### 문제 3 — 매 프레임 ray.get() Pull IPC: Raylet 스케줄링 누적
**심각도: High**

**코드 근거**: `analysis_actor.py:1123`
```python
frame_data = ray.get(self.stream_actor.get_frame.remote())
```

매 프레임마다 발생하는 작업:
```
AnalysisActor → [Raylet IPC] → StreamActor 태스크 큐
→ StreamActor get_frame() 실행 → threading.Lock 획득 → deque.popleft()
→ ray.get(frame_ref) (Plasma Store 읽기)
→ FrameData 생성 → [Raylet IPC] → AnalysisActor 응답 수신
```

STREAM_FPS=20 기준, CCTV 1개당 초당 20회 Raylet 왕복. CCTV N대이면 20N회/초. 부하 상황에서 Raylet 스케줄러 적체 → 지연 시간 누적.

이 구조는 **Pull 방식(소비자가 생산자에게 요청)** 이라, 매 요청마다 Actor 원격 호출 오버헤드가 불가피하다.

---

#### 문제 4 — 매 프레임 ray.put() 생산 오버헤드: 버려지는 프레임 포함
**심각도: Medium**

**코드 근거**: `stream_actor.py:297`
```python
frame_ref = ray.put(frame)
```

`_streaming_loop` 스레드는 20fps 이상으로 `ray.put(frame)` 호출. deque의 `maxlen=15` 초과 시 오래된 항목이 `popleft()`로 제거되지만, 해당 `frame_ref`(ObjectRef)는 deque에서만 사라진다. Plasma Store에서의 실제 GC는 Ray의 참조 카운팅에 따라 처리되므로, **AnalysisActor가 소비하기 전에 버려지는 프레임도 Plasma Store 쓰기 비용을 이미 발생시킨다**.

```
실측 비용 (1280×720 BGR, float32 전처리 전):
  raw frame: 1280 × 720 × 3 bytes = 2.76MB
  20fps × 2.76MB = 55.2MB/s per CCTV Plasma Store 쓰기
```

---

#### 문제 5 — 직렬 동기 gRPC 3회/프레임: 의존성 체인으로 병렬화 불가
**심각도: High**

**코드 근거**:
- `detection_engine.py:78`: `response = self.client.infer(...)` — blocking
- `tracking_engine.py:148`: `response = self.client.infer(...)` — blocking
- `pose_engine.py` (Violence/Falldown 활성화 시): `self.client.infer(...)` — blocking

세 호출은 **데이터 의존성**으로 직렬 불가피:
```
detect (bboxes) → track (bboxes 필요) → pose (tracking_result 필요)
```

AnalysisActor는 단일 스레드이므로 gRPC 응답 대기 중 다음 프레임 처리 불가.

```
프레임당 최대 총 소요: ~20 + ~10 + ~25 = ~55ms (Violence/Falldown 활성화 시)
이론 FPS 상한: 1000ms / 55ms ≈ 18fps
실측은 네트워크 지연 + 처리 오버헤드로 더 낮음
```

이 병목의 해결책은 **한 프레임 내 병렬화**가 아니라 **프레임 간 배치**(Opt 3)여야 한다. 단일 프레임 내에서는 구조적으로 병렬화할 수 없다.

---

#### 문제 6 — np.append() O(N) 재할당: 매 프레임 Recorder 전체 배열 복사
**심각도: Medium**

**코드 근거**: `analysis_actor.py:780`
```python
recorder_entry = np.array(
    [[frame_data.timestamp, frame_data.frame.copy(), track_copy]],
    dtype=np.object_,
)
self.state.recorder = np.append(self.state.recorder, recorder_entry, axis=0)
```

`np.append()`는 **매 호출 시 전체 배열을 새로 할당하고 복사**한다. O(1) amortized가 아니라 매번 O(N).

```
Recorder 누적량: 30초 × 20fps = 600 프레임
600프레임 시점 np.append() 비용: 600행짜리 배열 전체 복사
→ 사람이 존재하는 매 프레임 반복
→ 프레임 처리 시간 중 append_recorder 항목이 지속 증가
```

이는 Ray IPC와 무관한 **AnalysisActor 내부 로컬 CPU 비용**이다. 문제 7(SerDes)과 다른 독립적인 병목이다.

---

### [C] 이벤트 처리 경로 병목

---

#### 문제 7 — np.object_ dtype SerDes: 이벤트 발생 시 1.55GB pickle 직렬화
**심각도: High**

**코드 근거**: `analysis_actor.py:443`
```python
def get_recorder(self) -> np.ndarray:
    return self._get_recorder_ref()   # dtype=np.object_ 배열 반환
```

`utility.py:577`
```python
recorder = ray.get(analysis_actor_ref.get_recorder.remote())
```

Ray가 Actor 메서드 반환값을 전달할 때, `dtype=np.float32` 같은 숫자형 배열은 Plasma Store의 **zero-copy** 경로를 탄다. 그러나 **`dtype=np.object_`** 배열은 Python 객체(frame numpy 배열, datetime, dict 등)를 포함하므로 zero-copy가 불가능하고 **전체 pickle 직렬화**가 강제된다.

```
직렬화 대상 크기 계산 (1280×720 기준):
  프레임 1개: 1280 × 720 × 3 = 2.76MB
  600프레임 × 2.76MB = 1.656GB

  + timestamps(600 × ~24B) + tracking_results(600 × ~수KB)
  실질적 직렬화 크기: ~1.55GB

예상 직렬화 시간: 수백 ms (CPU-Bound)
→ 이벤트 감지 때마다 utility_task가 이 비용 부담
→ 분석 스레드 자원을 잠식(Starvation 유발)
```

이 비용은 문제 6(np.append CPU 복사)과 **다른 지점**에서 발생한다. 문제 6은 매 프레임 로컬 누적 비용, 문제 7은 이벤트 시 Ray IPC 직렬화 비용이다.

---

#### 문제 8 — utility_task → get_recorder 태스크 기아 (Starvation)
**심각도: High**

**"데드락"이 아닌 이유**: Coffman 4조건 중 **순환 대기(Circular Wait)** 미충족. `start_analysis()`는 `get_recorder()`가 갖고 있는 자원을 기다리지 않는다.

**정확한 분류**: **태스크 기아(Task Starvation) / 무한 연기(Indefinite Postponement)**

**코드 근거**:
```python
# model.py:220
analysis_actor.start_analysis.remote()  # fire-and-forget, 무한 루프

# analysis_actor.py:1120
while self.running:               # 절대 반환하지 않음
    frame_data = ray.get(...)
    self.process_frame(frame_data)

# utility.py:577
recorder = ray.get(analysis_actor_ref.get_recorder.remote())
# ↑ AnalysisActor 태스크 큐에 적재되지만 실행 불가
```

Ray Actor 기본 모델은 **단일 워커 스레드, 태스크 단위 순차 처리**다. `start_analysis()`가 점유하는 한 뒤에 쌓인 `get_recorder()` 태스크는 **실행 기회를 영원히 얻지 못한다**.

```
Actor 태스크 큐 상태:
  [start_analysis — 실행 중, while True, 반환점 없음]
  [get_recorder   — 대기 중, 앞 태스크가 끝나지 않아 영구 미실행]
  [get_recorder   — 대기 중 (다음 이벤트)]
  ...

결과:
  AnalysisActor 자체는 계속 동작 (분석 루프는 멈추지 않음)
  utility_task는 get_recorder 응답을 영원히 받지 못해 hang
  이벤트 발생마다 hung utility_task 누적 → 메모리 지속 상승
  Recording, 썸네일 저장이 실질적으로 동작하지 않음
```

**해결을 위해 필요한 것**: Opt 2(zero-copy recorder 구조 변경) 만으로는 해결 안 됨. 아래 두 가지 중 하나가 병행되어야 한다.
- AnalysisActor를 `async` Actor로 전환 (`async def start_analysis`, `await` 사용) → `await` 지점에서 다른 메서드 실행 기회 획득
- Actor 콜백 자체를 없애고, recorder 데이터를 Actor 외부(shared 구조 or 파라미터 직접 전달)로 전환

---

### [D] 코드 품질 문제

---

#### 문제 9 — fast_inference.py: @ray.remote 정의 있으나 실제 미사용 (Dead Code)
**심각도: Low**

**코드 근거**: `analysis_actor.py:207`
```python
self.fast_inference_task = fast_inference_task  # 주입은 됨
```
`analysis_actor.py:471~499`
```python
# 실제 처리는 로컬 메서드
result = self._infer_fast_event(event_type, inference_request)
```

`fast_inference_task.remote()` 호출은 코드 어디에도 없다. `fast_inference.py`의 `@ray.remote` 함수는 정의·주입·import 되지만 실행되지 않는다. 독립 구조 전환 시 이 task를 살릴 건지(AnalysisActor 경량화), 삭제할 건지 결정하지 않으면 혼란이 지속된다.

---

#### 문제 10 — save_daily_thumbnail_task 중복 정의
**심각도: Low**

**코드 근거**: `utility.py` 내 동일 이름 `@ray.remote` 함수가 2번 정의됨.
- 1번째 정의(~line 93): `bucket_client` 파라미터를 받는 버전
- 2번째 정의(~line 489): 파라미터 없이 내부에서 `_get_minio_client()` 호출하는 버전

Python에서 나중 정의가 앞을 덮어쓰므로 1번째 버전은 **dead code**. 두 구현이 미묘하게 달라 독립 구조 전환 시 버그 유발 가능성이 있다.

---

#### 문제 11 — execute()의 time.sleep(3) × 2: Triton 관리 스레드 6초 블로킹
**심각도: Low**

**코드 근거**: `model.py:282, 317`
```python
time.sleep(3)   # 수신 직후
# ... 메시지 처리 ...
time.sleep(3)   # 완료 직전
```

`execute()`는 Triton의 관리 메시지(CCTV 등록/삭제/이벤트 설정 등)를 처리하는 함수다. 매 호출마다 6초 블로킹. 의도적 딜레이인지, 레거시 잔류인지 불명확하다. 이 함수가 실행되는 동안 Triton Backend 스레드가 점유된다.

---

#### 문제 12 — _handle_event_switch의 update_event_flag 미구현
**심각도: Low**

**코드 근거**: `model.py:451`
```python
if target_user in self.actor_pool and target_cctv in self.actor_pool[target_user]:
    analysis_actor = self.actor_pool[target_user][target_cctv]["analysis_actor"]
    # analysis_actor.update_event_flag.remote(event_name, process_val)  # 구현 필요
```

이벤트 ON/OFF 변경 시 `process_pool`(레거시 shared memory)의 값은 업데이트하지만, **AnalysisActor의 `cctv_config.event_flags`는 변경되지 않는다**. 런타임 중 이벤트를 껐어도 AnalysisActor는 계속 해당 이벤트를 처리한다. `update_event_flag` 메서드 자체가 AnalysisActor에 구현되어 있지 않다.

---

## 5. 문제 요약표

| # | 분류 | 문제 | 심각도 | 핵심 코드 위치 |
|---|------|------|--------|---------------|
| 1 | 구조 | Stub + Ray 중첩 → CPU 경합/OS 스케줄링/캐시 오염/SHM 이중복사 | Critical | `model.py:26` |
| 2 | 구조 | Object Spilling → Memory Thrashing → Actor 기아 | Critical | `ray.init()` 설정 없음 |
| 3 | 프레임 경로 | 매 프레임 ray.get() Pull IPC — Raylet 스케줄링 누적 | High | `analysis_actor.py:1123` |
| 4 | 프레임 경로 | 매 프레임 ray.put() — 버려지는 프레임도 Plasma Store 쓰기 | Medium | `stream_actor.py:297` |
| 5 | 프레임 경로 | 직렬 동기 gRPC 3회/프레임 — 의존성 체인, 병렬화 불가 | High | `detection_engine.py:78` 외 |
| 6 | 프레임 경로 | np.append() O(N) 재할당 — 매 프레임 Recorder 전체 복사 | Medium | `analysis_actor.py:780` |
| 7 | 이벤트 경로 | np.object_ dtype → zero-copy 불가 → 이벤트 시 1.55GB pickle | High | `analysis_actor.py:443` |
| 8 | 이벤트 경로 | utility_task → get_recorder **태스크 기아** (무한 연기) | High | `utility.py:577` |
| 9 | 코드 품질 | fast_inference.py @ray.remote — 주입만 되고 미사용 dead code | Low | `tasks/fast_inference.py` |
| 10 | 코드 품질 | save_daily_thumbnail_task 중복 정의 — 1번째 버전 dead code | Low | `utility.py:93, 489` |
| 11 | 코드 품질 | execute() time.sleep(3) × 2 — Triton 관리 스레드 6초 블로킹 | Low | `model.py:282, 317` |
| 12 | 코드 품질 | update_event_flag 미구현 — 런타임 이벤트 ON/OFF가 Actor에 미반영 | Low | `model.py:451` |

---

## 6. README 최적화 계획 검증 및 보완 사항

### Opt 1 — Async Queue (Pull → Push)

**계획**: `ray.util.queue.Queue`로 StreamActor→AnalysisActor 간 Push 방식 전환.

**현재 구조 vs Queue 구조 차이**:

| 항목 | 현재 (deque in Actor) | Opt 1 (ray.util.queue.Queue) |
|------|----------------------|------------------------------|
| deque 소유자 | StreamActor 내부 | 독립 Queue Actor (Ray 관리) |
| Consumer 방식 | Pull: `get_frame.remote()` | Pull: `queue.get()` |
| Producer 방식 | 스레드 내 직접 append | Push: `queue.put()` (fire-and-forget) |
| threading.Lock | **필수** (스레드 2개가 deque 공유) | **불필요** (Queue Actor 단일 스레드) |
| Producer 블로킹 | 없음 | 없음 (async put) |
| IPC 비용/프레임 | Raylet 왕복 2회 (요청+응답) | put 1회(비동기) + get 1회 |
| Actor 간 결합 | 높음 (AA가 SA에 직접 의존) | 낮음 (둘 다 Queue에만 의존) |

**⚠️ 구현 시 주의 (README에 누락된 디테일)**:

`ray.util.queue.Queue`에 **raw numpy 배열을 직접** 넣으면 Queue Actor의 IPC 채널로 직렬화 전송되어 Plasma Store보다 **느리다**. `ray.put()`은 반드시 유지해야 한다.

```python
# ❌ 잘못된 구현 — frame 전체가 IPC 채널로 직렬화
queue.put({"frame": frame_numpy, "timestamp": now})

# ✅ 올바른 구현 — ObjectRef(경량)만 Queue IPC, 실제 데이터는 Plasma Store
frame_ref = ray.put(frame)
queue.put({"frame_ref": frame_ref, "timestamp": now, "frame_id": self.frame_id})

# AnalysisActor
frame_info = queue.get()
frame = ray.get(frame_info["frame_ref"])  # Plasma Store zero-copy
```

---

### Opt 2 — Zero-copy Recorder Pipeline

**계획**: Recorder 저장 시 frame 배열 대신 `ray.put()` 한 ObjectRef를 저장. 1.55GB → ~110KB.

**유효성**: 문제 7(SerDes 비용)을 해결한다. 계산:
```
현재: 600 × 2.76MB = 1.656GB (pickle 직렬화)
개선: ObjectRef 600개(24B×600=14.4KB) + timestamps(14.4KB) + tracking(~72KB) ≈ 100KB
```

**⚠️ 그러나 Opt 2만으로는 문제 8(기아)이 해결되지 않는다**:

Recorder 크기가 줄어도 `ray.get(analysis_actor_ref.get_recorder.remote())`는 여전히 AnalysisActor 태스크 큐에 적재되고, `start_analysis()` 무한 루프가 점유하는 한 실행 불가다. **Opt 2와 함께 Actor 동시성 수정이 반드시 병행**되어야 한다.

선택지:
```python
# 방법 A: async Actor로 전환
@ray.remote
class AnalysisActor:
    async def start_analysis(self):
        while self.running:
            frame_info = await queue.get()   # await → 다른 메서드 실행 기회 부여
            await self.process_frame(...)

    async def get_recorder(self):            # 이제 start_analysis와 동시 실행 가능
        return self.state.recorder

# 방법 B: Actor 콜백 제거 — recorder를 직접 utility 파라미터로 전달
# Opt 1(Queue)과 함께 → AnalysisActor가 push 시점에 recorder snapshot을 함께 전달
```

**⚠️ ObjectRef 보존 문제**:

ObjectRef들을 deque에 유지하는 동안, Ray의 Object Store 메모리 압박이 높으면 참조되지 않은 것으로 판단해 GC할 수 있다(eviction). deque가 유일한 참조자라면 이 경우 `ray.get(ref)`가 실패한다. Object Store 메모리 설정(`ray.init(object_store_memory=...)`) 또는 명시적 pin 처리가 필요하다.

---

### Opt 3 — Hybrid Batching

**계획**: Detection/Pose 추론을 프레임 간 배치로 처리해 GPU 효율 극대화.

**유효성**: 문제 5(직렬 gRPC 3회)의 **처리량(Throughput)** 개선에는 유효하다.

**⚠️ 지연시간(Latency) 트레이드오프 (README에 누락)**:

배치 크기 B로 묶으면, 배치가 채워질 때까지 대기하는 B-1개 프레임의 지연시간이 증가한다. 이미 처리 지연이 쌓이는 Stress 시나리오에서 배치 대기 시간이 추가되면 전체 파이프라인 지연이 악화될 수 있다. **로컬 환경 구축 후 배치 크기별 Throughput/Latency 트레이드오프를 반드시 측정**해야 한다.

---

### 최적화 실행 순서 권장

README는 Opt 1 → Opt 2 → Opt 3을 제시하지만, **기아(Starvation) 관점에서 올바른 선행 순서**는 다음과 같다.

```
Step 1: Opt 2 (zero-copy Recorder) + Actor async 전환
        → 문제 7(SerDes) + 문제 8(기아) 동시 해결
        → 이 둘은 분리 불가한 쌍

Step 2: Opt 1 (Async Queue, ray.put() 유지)
        → 문제 3(Pull IPC) 해결
        → Step 1 이후 적용해야 구조가 안정적

Step 3: Opt 3 (Hybrid Batching) + Object Store 메모리 설정
        → 문제 5(직렬 gRPC) Throughput 개선
        → ray.init(object_store_memory=...) 추가로 문제 2(Spilling) 제어

```
### 문제 4, 6을 위한 추가 최적화 방안
```
 문제 4 (wasted ray.put):
    해결책: ray.put() 앞에 큐 용량 체크 → 조건부 호출                                                                                                                                                                                
    적용 위치: stream_actor.py _streaming_loop                                                                                                                                                                                           Opt 1/2/3 여부: 무관. 현재 구조에서 독립 패치 가능                                                                                                                                                                               
                                                                                                                                                                                                                                       문제 6A (np.append O(N)):
    해결책: deque(maxlen=600) 교체                                                                                                                                                                                                   
    적용 위치: analysis_actor.py _append_to_recorder()                                                                                                                                                                                              (models.py _recorder_deque 필드 이미 존재)
    Opt 1/2/3 여부: 무관. 독립 패치 가능                                                                                                                                                                                             
                                                                                                                                                                                                                                       문제 6B (recorder 메모리 1.65GB):                                                                                                                                                                                                  
    해결책: Opt 2 (ObjectRef 저장)                                                                                                                                                                                                   
    부작용: AnalysisActor에서 ray.put() 재호출 필요                                                                                                                                                                                  
            ObjectRef eviction 위험 → object_store_memory 설정 병행 필요                                                                                                                                                             
    Opt 1/2/3 여부: Opt 2로 처리됨          
```
---

## 7. 외부 의존성 및 환경변수

| 서비스 | 연결 방식 | 환경변수 |
|--------|----------|---------|
| Triton Inference Server | gRPC `TritonServer:8001` | 코드 하드코딩 (`constants.py`) |
| RabbitMQ | pika AMQP | `rabbitmq`, `MQ_USERID`, `MQ_PASSWORD` |
| MinIO | minio SDK | `minio`(또는 `bucket`), `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_SECURE` |
| PostgreSQL/MySQL | pymysql | legacy_utils에서 처리 (내부 DB) |
| Violence TSM 모델 | 환경변수로 CCTV ID 지정 | (내부 환경변수) |

---

## 8. 이벤트 타입별 처리 요약

| 이벤트 | 분류 | 처리 위치 | Triton 호출 | 비고 |
|--------|------|----------|------------|------|
| Intrusion | Fast | AnalysisActor 로컬 | 없음 | Shapely Polygon 침입 체크 |
| Loitering | Fast | AnalysisActor 로컬 | 없음 | Loitering_EVENT (legacy) |
| Violence | Batch | batch_inference Task | 2회 동기(GCN) or 1회 async(TSM) | 12/8 프레임 버퍼 가득 시 트리거 |
| Falldown | Batch | batch_inference Task | 프레임당 1회 동기 | pose_result 필요 |
| Fire | Fast (정의만) | — | — | 미구현 |

---

## 9. AWS 배포 구성 요소

- **EC2 #1 (Python Backend + Ray)**: CPU 집약. StreamActor, AnalysisActor, Task Worker 프로세스.
- **EC2 #2 (Triton Inference Server)**: GPU 필수. YOLOv9 TRT, Pose, Violence, Falldown 모델.
- **외부 서비스**: RabbitMQ, MinIO (로컬 테스트 시 mock 대체 가능).
- **로컬 테스트**: RTSP → 영상 파일 루프 재생 (`cctv_object["ip"]`에 파일 경로 주입).

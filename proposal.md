
# proposal.md — 최적화 경로 재검토 및 대안 아키텍처 검증

> 목적: research.md에서 식별한 12개 병목과 README의 Opt 1/2/3 + 문제 4/6 보완 계획을
> 재검토하고, "Triton Python Backend 제거 + Standalone Ray ↔ Triton Server 동등 구조"가
> 이 도메인에 정말 합리적인 선택인지 대안 아키텍처와 비교하여 판단 근거를 남긴다.

---

## 0. 결론 요약

```
[결론 1] Standalone Ray ↔ Triton Server 동등 구조는 이 도메인에 합리적이다.
         "nested Ray가 문제였지, Ray 자체가 부적합한 것이 아니다."
         Stub Process가 제거되면 문제 1(CPU 경합/캐시 오염/SHM↔Plasma 이중복사)의
         근본 원인이 함께 사라지므로, Ray의 고정 비용 대비 가치가 비로소 정당화된다.

[결론 2] Opt 1/2/3 + 문제 4/6 보완 + 코드 품질 정리(문제 9~12)까지 모두 적용하면
         12개 문제 중 11개가 해결된다. (잔존: 문제 5의 Latency 트레이드오프 — 측정 필요)

[결론 3] 그럼에도 포트폴리오 가치를 위해서는 "왜 Ray여야 했는가"의 정량적 근거가 필요하다.
         단일 CCTV 미니 벤치로 "raw multiprocessing"과 "asyncio 단일 프로세스"를
         짧게 비교 측정하면, Ray 선택의 합리성을 직접 데이터로 증명할 수 있다.

[권장] Track 1(Standalone Ray + 전 보완) 본 측정 + Track 2(미니 대안 벤치) 보조 측정.
       Track 2는 본 구현 없이 1~2주 분량의 비교 PoC로 충분하다.
```

---

## 1. Standalone Ray 구조에서의 12개 문제 재평가

### 1-1. Stub Process 제거가 가져오는 변화

현재 nested Ray의 문제 1은 다음 4단계 악순환이었다.

```
[1] CPU 자원 경합 — Stub / Raylet / GCS / Worker 프로세스 동일 코어 경합
[2] OS 스케줄링 지연 — 컨텍스트 스위칭 누적
[3] L1/L2 캐시 오염 — 프로세스 어피니티 미설정
[4] 메모리 이중 복사 — Triton SHM → Ray Plasma Store
```

**Triton Python Backend를 제거하면 이 4단계가 모두 약화 또는 소멸한다**:

```
[1] → 소멸: Stub Process 사라짐. Ray 프로세스만 남음.
[2] → 약화: 핵심 경합 주체(Stub) 제거. 남은 Raylet/Worker는 Ray 본연의 동작.
[3] → 약화: Worker 프로세스 어피니티는 ray.init(num_cpus=...)로 제어 가능.
[4] → 소멸: Triton SHM 경로 자체가 사라짐. Plasma Store만 남음.
```

문제 1은 **Ray의 잘못이 아니라 nested 구조의 잘못**이었음이 명확해진다. Standalone 구조에서는 Ray의 고정 비용이 합리적 수준으로 떨어진다.

### 1-2. Standalone Ray + 전 보완 적용 시 12개 문제 해결 매트릭스

| # | 문제 | 해결 수단 | Standalone Ray + 전 보완 후 |
|---|------|----------|-----------------------------|
| 1 | Stub+Ray 중첩 | Triton Backend 제거 (구조 전환) | ✅ 소멸 |
| 2 | Object Spilling | `ray.init(object_store_memory=...)` 명시 | ✅ 제어 가능 |
| 3 | Pull IPC | Opt 1 (Async Queue Push) | ✅ |
| 4 | ray.put() 낭비 | 큐 용량 선확인 후 조건부 put | ✅ (research.md §6 추가 방안) |
| 5 | 직렬 gRPC 3회 | Opt 3 (Hybrid Batching) | △ Throughput ✅, Latency 측정 필요 |
| 6 | np.append() O(N) | `deque(maxlen=600)` 교체 | ✅ (research.md §6 추가 방안) |
| 7 | np.object_ SerDes | Opt 2 (zero-copy Recorder) | ✅ |
| 8 | get_recorder 기아 | async Actor 전환 | ✅ |
| 9 | fast_inference dead | dead code 제거 | ✅ |
| 10 | save_daily 중복 | 정리 | ✅ |
| 11 | execute() sleep | Triton Backend 제거로 execute() 자체 사라짐 | ✅ |
| 12 | update_event_flag | 신규 구현 (Actor 메서드 추가) | ✅ |

**11개 완전 해결, 1개(문제 5) 트레이드오프 측정 항목으로 분류**. 기능적 잔존 병목 없음.

---

## 2. "Ray가 정말 합리적인가" 재검토 — GIL 관점 추가

이전 버전 proposal에서 "asyncio 단일 프로세스로도 충분"이라고 한 부분은 GIL 문제를 가볍게 봤다는 점에서 정정이 필요하다.

### 2-1. 이 시스템의 클라이언트 측 CPU 작업

GPU 추론은 Triton으로 오프로드되지만, **클라이언트(분석 프로세스)에서도 CPU 작업이 적지 않다**:

```
프레임당 클라이언트 CPU 작업:
  - 프레임 전처리: cv2.resize, BGR↔RGB 변환, np.float32 캐스팅
  - Detection 후처리: bbox 정규화, 좌표 변환
  - Tracking 후처리: tracker_object 상태 업데이트, np.copy
  - Recorder 누적: deque append (O(1)이라도 frame.copy() 비용 존재)
  - Fast events: Shapely Polygon.contains 판정 (CCTV별 ROI 다수)
  - Loitering: 시간/거리 계산
  - Frame meta 직렬화: utility task로 넘기기 위한 객체 구성

CCTV N대 × 20fps × 위 작업들 → CPU 누적 부하 무시 못 함
```

### 2-2. asyncio 단일 프로세스의 한계

asyncio는 **단일 스레드 + 단일 GIL**이다. `await` 지점에서만 양보가 일어나므로:

```
await grpc.detect()       ← I/O 대기, 다른 코루틴 양보 OK
det = process_bbox(det)   ← CPU 작업, 동기 실행, GIL 점유
await grpc.track(det)     ← I/O 대기, 양보 OK
track = update_tracker(track) ← CPU 작업, GIL 점유
```

CCTV 1~3대까지는 CPU 작업 사이의 `await` 양보로 그럭저럭 돌아갈 수 있다. 그러나 5대, 10대로 늘어나면:

```
- CPU 작업이 GIL을 직렬화 → 사실상 한 코어만 사용
- 한 CCTV의 CPU 슬라이스가 다른 CCTV의 await 콜백 처리 지연
- p99 Latency 폭증, FPS 한계가 단일 코어 성능에 묶임
```

이는 ProcessPoolExecutor로 CPU 작업을 오프로드해도 해결이 부분적이다 (frame 직렬화 비용이 다시 등장).

**결론**: 다중 CCTV에서는 멀티프로세스 격리가 사실상 필수다.

### 2-3. 멀티프로세스가 필수일 때, Ray vs Raw multiprocessing

선택지는 두 갈래다.

#### 옵션 X — Raw multiprocessing

```python
import multiprocessing as mp
from multiprocessing import shared_memory

# CCTV당 mp.Process 2개 (stream + analyze)
# IPC: mp.Queue, shared_memory.SharedMemory
```

**필요한 것을 직접 구현해야 한다**:

| 항목 | 직접 구현 필요 사항 |
|------|------------------|
| Zero-copy IPC | `multiprocessing.shared_memory.SharedMemory` + numpy view 수동 관리 |
| 프레임 GC | shared_memory unlink 시점 추적 (참조 카운팅 직접) |
| Process lifecycle | spawn / 종료 / 재시작 watchdog 직접 구현 |
| 메시지 라우팅 | Queue 다중화 또는 Manager 객체 직접 구성 |
| 에러 전파 | Exception pickling, child process 죽음 감지 |
| gRPC pool | 프로세스별 connection 재사용 정책 직접 구현 |
| 분산 확장 | 단일 머신 한정. 다른 노드로 확장 시 전부 재작성 |

특히 **shared_memory로 frame을 zero-copy 전달**하려면, segment 생성/소비/언링크의 race condition을 직접 처리해야 한다. 잘못 구현하면 segfault 또는 메모리 누수.

#### 옵션 Y — Standalone Ray

```python
import ray
ray.init(num_cpus=..., object_store_memory=...)

@ray.remote
class StreamActor: ...

@ray.remote
class AnalysisActor: ...
```

**Ray가 기본 제공하는 것**:

| 항목 | Ray 제공 |
|------|---------|
| Zero-copy IPC | Plasma Store (numpy 배열은 자동 zero-copy 경로) |
| 프레임 GC | ObjectRef 참조 카운팅 자동 |
| Actor lifecycle | `actor.kill()`, `max_restarts`, named actor |
| 메시지 라우팅 | `actor.method.remote()` 표준 인터페이스 |
| 에러 전파 | `RayActorError`, `RayTaskError` 자동 wrapping |
| gRPC pool | (수동 구현 필요는 동일하나 Actor 단위로 깔끔함) |
| 분산 확장 | `ray start --address=...`로 멀티 노드 즉시 가능 |

**Ray의 실질 가치**:
- 위 7개 항목을 직접 구현하지 않고 표준 패턴으로 사용 가능
- 학습 곡선과 디버깅 도구(Ray Dashboard)가 성숙
- 추후 CCTV 수백 대로 확장 시 멀티 노드 전환이 코드 변경 거의 없이 가능

### 2-4. Ray의 잔존 비용 (Standalone에서도 남는 것)

정직하게 짚으면, Standalone Ray에도 비용이 남는다:

```
[A] Raylet + GCS 상주 메모리 (수백 MB)
    → CCTV 수가 많을수록 분산 비용으로 amortize
[B] Actor 메서드 호출의 Raylet 경유
    → Async Queue 사용 시 fire-and-forget으로 영향 최소화
    → 그래도 로컬 함수 호출보다 1~2 order 느림
[C] ObjectRef eviction 위험
    → object_store_memory 명시 + 명시적 관리로 제어 가능
[D] Actor 단일 스레드 제약
    → async Actor 전환 또는 max_concurrency 옵션으로 우회
```

이 비용들은 **Raw multiprocessing이 직접 구현해야 할 비용을 Ray에 위임한 대가**다. 직접 구현하면 더 빠를 수 있지만, **개발 비용 + 유지보수 비용 + 버그 위험**의 합이 더 크다.

---

## 3. 아키텍처 비교표 (수정 — Standalone Ray 기준)

| 항목 | nested Ray (현재) | Standalone Ray (계획) | Raw multiprocessing | asyncio 단일 |
|------|-----------------|---------------------|---------------------|--------------|
| Triton Backend Stub | 있음 | **없음** | 없음 | 없음 |
| Ray 프레임워크 | 있음 (nested) | **있음 (단독)** | 없음 | 없음 |
| Zero-copy frame IPC | Plasma (이중복사) | Plasma 단일 | shared_memory 직접구현 | 없음 (단일 프로세스) |
| 프레임 GC | Ray 자동 | Ray 자동 | unlink 직접 관리 | GC 자동 |
| Actor lifecycle | Ray 제공 | Ray 제공 | watchdog 직접 구현 | task 관리 |
| 멀티 노드 확장 | 가능 | **가능** | 어려움 (재설계) | 불가능 |
| 다중 CCTV 격리 | Actor | **Actor + 프로세스** | 프로세스 | ❌ GIL 경합 |
| GIL 영향 | 분산 (낮음) | **분산 (낮음)** | 분산 (낮음) | **집중 (높음)** |
| 개발 복잡도 | 중간 | **중간** | 높음 (직접 구현) | 낮음 |
| 디버깅 도구 | Ray Dashboard | **Ray Dashboard** | 없음 (수동) | asyncio debug |
| nested Ray 4단계 악순환 | 모두 발생 | **모두 소멸** | N/A | N/A |
| 측정 가치 | Before 기준 | **본 측정 대상** | 비교 PoC | 비교 PoC |

---

## 4. 권장 측정 트랙

### Track 1 (본 트랙): Standalone Ray + 전 보완

이것이 포트폴리오의 메인 산출물이다.

```
브랜치 진행:
  feat/baseline-nested
   ↓
  feat/ray-standalone
   ├─ Triton Python Backend 제거
   ├─ ray.init(object_store_memory=N) 명시 (문제 2)
   ├─ np.append → deque(maxlen=600) 교체 (문제 6)
   ├─ ray.put() 큐 용량 선확인 (문제 4)
   ├─ fast_inference dead 제거 (문제 9)
   ├─ save_daily 중복 정리 (문제 10)
   ├─ execute() 제거 또는 관리 인터페이스 재설계 (문제 11)
   └─ update_event_flag 구현 (문제 12)
   ↓
  feat/ray-opt2-zerocopy + async Actor 전환 (문제 7 + 문제 8)
   ↓
  feat/ray-opt1-asyncqueue (문제 3)
   ↓
  feat/ray-opt3-batching (문제 5, Latency 트레이드오프 측정 동반)
```

각 단계마다 측정:

```
- FPS (CCTV 1, 5, 10 / normal, stress 시나리오)
- 메모리 피크 (Plasma Store 사용량 분리)
- Object Spilling 발생 여부 (Raylet 로그)
- 프레임 latency p50, p99
- CPU util (Raylet vs StreamActor vs AnalysisActor 분리)
- get_recorder 기아 발생 여부 (utility task hang count)
```

### Track 2 (보조 트랙): "왜 Ray인가" 정량 근거 미니 PoC

본 구현 없이 단일 CCTV 기준 비교만.

#### 미니 PoC A: asyncio 단일 프로세스
- CCTV 1, 3, 5대로 늘려가며 GIL 경합 측정
- 5대에서 p99 latency가 어떻게 무너지는지 확인
- **예상 결과**: CCTV 1~2대까지 우수, 3대 이상에서 GIL 한계 노출

#### 미니 PoC B: Raw multiprocessing + shared_memory
- CCTV 1대 기준 Ray와 head-to-head 비교
- frame zero-copy IPC 직접 구현 (shared_memory + Queue)
- **예상 결과**: 단일 CCTV에서 Ray보다 약간 빠를 가능성 있음. 그러나 다중 CCTV/lifecycle 관리/에러 처리에서 코드량 폭증

이 두 미니 PoC의 결과로 다음 문장을 데이터로 뒷받침할 수 있다.

> "asyncio는 N≥3에서 GIL 한계로 부적합. Raw multiprocessing은 단일 CCTV에서 Ray와 유사 성능을 내지만, 다중 CCTV에서 lifecycle 관리/zero-copy IPC 직접 구현 비용이 Ray의 고정 오버헤드를 초과한다. 따라서 Standalone Ray가 이 도메인에 합리적이다."

---

## 5. 실행 순서 — 최종 권장

```
Phase 1 — Baseline 측정
  feat/baseline-nested에서 nested Ray 수치 확보
  (이미 포트폴리오 Before 기준)

Phase 2 — 구조 전환
  feat/ray-standalone
    Triton Backend 제거 + Ray 단독 구조 + 코드 품질 정리(9~12)
    + 미시 보완(문제 4, 6) + ray.init 메모리 설정(문제 2)
  → 측정: nested vs standalone 차이 확인 (문제 1, 2 효과)

Phase 3 — 동시성 모델 정정
  feat/ray-opt2-zerocopy + async Actor 전환
  → 측정: 문제 7, 8 해결 확인

Phase 4 — IPC 모델 정정
  feat/ray-opt1-asyncqueue
  → 측정: 문제 3 해결 확인

Phase 5 — 처리량 최적화
  feat/ray-opt3-batching
  → 측정: 문제 5의 Throughput/Latency 트레이드오프

Phase 6 (선택) — 정당화 PoC
  branch: poc/asyncio-single, poc/raw-multiprocessing
  → 1주 분량 비교 측정으로 "왜 Ray인가" 정량 근거 확보
```

---

## 6. 의사결정 체크리스트

```
[ ] Phase 2 완료 후 nested vs standalone Ray 메모리/CPU 차이 명시
    → 문제 1의 영향이 얼마나 컸는지 정량화

[ ] object_store_memory 설정값을 워크로드별로 측정해 결정
    → Spilling 발생 임계 + 안전 마진

[ ] async Actor 전환 시 디버깅 가능성 검증
    → ray.util.state CLI, Ray Dashboard로 await 지점 추적 가능한지

[ ] Opt 3 적용 시 batch size별 Latency p99 측정
    → 실시간 알람 SLA(예: 500ms) 내 머무르는 batch size 상한 결정

[ ] Phase 6 PoC 수행 여부 결정
    → 포트폴리오 메시지를 "정량 비교 기반 선택"으로 강화할지,
      "Ray 단일 트랙 깊이 분석"으로 갈지 시간 예산에 따라 판단
```

---

## 7. 맺는 말 — 포트폴리오 메시지

이 분석으로 다음 메시지가 정당해진다.

> **"nested Ray가 보여준 병목은 Ray의 한계가 아니라 nested 구조의 한계였다.
> Triton Python Backend를 제거해 Ray ↔ Triton Server 동등 구조로 만들고,
> 12개 식별 문제를 11개까지 직접 해결하여 Ray의 강점(zero-copy IPC,
> Actor lifecycle, 멀티 노드 확장성)이 비용을 정당화하는 구조로 정착시켰다.
> 다중 CCTV 환경에서 asyncio 단일 프로세스는 GIL 한계로 부적합했고,
> Raw multiprocessing은 lifecycle/zero-copy IPC 직접 구현 비용이
> Ray의 고정 오버헤드를 초과함을 비교 PoC로 확인했다."**

이 문장이 proposal.md의 최종 목표다. Phase 2~5는 메인 트랙이고, Phase 6는 이 문장의 후반부를 데이터로 뒷받침하는 보조 트랙이다.
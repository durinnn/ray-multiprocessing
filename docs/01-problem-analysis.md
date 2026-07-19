# 01 — 원 시스템 문제 분석

> 원 CCTV 분석 파이프라인(Triton Python Backend 안에서 Ray를 nested로 기동한 구조)의
> 구조와 그 안에서 식별된 12개 문제를 요약한다. 상세 코드 근거는 저장소 루트의
> `research.md`에 있으며, 이 문서는 벤치마크가 재현 대상으로 삼은 항목을 중심으로 증류한다.

---

## 1. 원 시스템 구조

원 시스템은 실시간 CCTV 영상에서 객체 탐지·추적·포즈 추정을 수행하고, 폭행·쓰러짐·침입·배회
같은 이벤트를 판정하는 분석 파이프라인이다. 핵심 특징은 **분석 로직이 Triton Inference
Server의 Python Backend 안에서 실행되며, 그 안에서 다시 Ray 클러스터를 초기화(nested)** 한다는
점이다.

```
분석 서버 (Triton Python Backend)
 └─ Stub Process (Triton이 fork-exec로 생성)
     └─ ray.init()                     ← Stub 안에서 Ray 클러스터 기동 (nested)
         ├─ Raylet / GCS / Worker 프로세스
         ├─ StreamActor      (카메라당 1개)
         │    └─ threading.Thread: 프레임 수신 → ray.put() → deque(maxlen=15)
         └─ AnalysisActor    (카메라당 1개)
              └─ start_analysis(): while True 무한 루프
                   ├─ ray.get(stream_actor.get_frame.remote())   ← 매 프레임 Pull
                   └─ detect → track → recorder 누적 → pose
                      → 침입/배회(로컬) → 프레임 누적 추론(폭행/쓰러짐)

추론 서버 (Triton Inference Server, GPU)
 └─ detect / track / pose / violence / falldown 모델
```

프레임 처리 파이프라인(프레임당)은 다음 순서다.

1. `AnalysisActor`가 `StreamActor`에서 프레임을 **Pull**로 당겨온다.
2. `detect`(gRPC) → `track`(gRPC) — 데이터 의존성으로 직렬.
3. 프레임과 트래킹 결과를 Recorder에 누적한다(`np.object_` 배열 + `np.append`).
4. `pose`(gRPC) → 침입/배회 규칙 판정(로컬 연산).
5. 프레임이 일정 수 쌓이면 폭행/쓰러짐 추론을 fire-and-forget 태스크로 기동한다.
6. 이벤트 발생 시 유틸리티 태스크가 `get_recorder()`를 콜백으로 호출한다.

---

## 2. 식별된 12개 문제

문제는 네 범주로 나뉜다. 이 벤치마크는 성능에 직접 영향을 주는 **범주 A~C의 8개 문제**를
합성 워크로드로 재현 대상으로 삼고, 코드 품질 범주(D)는 성능 벤치마크와 무관하여 범위 밖으로
둔다.

### [A] 구조적 병목

**문제 1 — Stub Process + Ray 중첩 (Critical).**
Triton Python Backend는 분석 로직을 별도 Stub Process에서 실행한다. 그 위에 Ray를
초기화하면 동일 머신에 Raylet·GCS·Worker 프로세스가 추가로 올라간다. 이로 인해 (1) CPU 자원
경합, (2) OS 스케줄링 지연, (3) 프로세스 어피니티 미설정에 따른 L1/L2 캐시 오염, (4) Triton의
Shared Memory에서 Ray Plasma Store로의 데이터 이중 복사가 연쇄적으로 발생한다. 이 네 가지는
Ray 자체의 결함이 아니라 **nested 구조에서만 나타나는 문제**다.

**문제 2 — Object Spilling → Memory Thrashing → Actor 기아 (Critical).**
`ray.init()`에 `object_store_memory` 설정이 없어 Object Store 압박 임계값을 제어할 수 없다.
Recorder와 프레임 버퍼가 Plasma Store를 포화시키면 Raylet이 디스크 스필링을 강행하고, 스왑
I/O가 CPU를 과점하는 동안 `AnalysisActor`는 메모리 할당을 받지 못해 I/O Wait 상태로 대기한다.
원 시스템 실측에서 메모리가 지속 상승하고 분석 Actor의 CPU가 주기적으로 0%로 떨어지는 패턴이
관찰되었다.

### [B] 프레임 처리 경로 병목

**문제 3 — 매 프레임 Pull IPC (High).**
`AnalysisActor`가 매 프레임 `ray.get(stream_actor.get_frame.remote())`로 프레임을 당겨온다.
프레임 한 개당 Raylet 왕복이 2회(요청 제출 + 응답 회수) 발생하며, 카메라 N대 × 20fps 기준
초당 20N회의 왕복이 쌓인다. 부하 상황에서 Raylet 스케줄러 적체가 지연으로 누적된다.

**문제 4 — 버려지는 프레임도 ray.put() (Medium).**
스트리밍 스레드가 소비 여부와 무관하게 매 프레임 `ray.put()`을 호출한다. `deque(maxlen=15)`를
초과해 밀려나는 프레임도 이미 Plasma Store 쓰기 비용을 지불한 뒤다.

**문제 5 — 직렬 동기 gRPC 3회 (High).**
`detect → track → pose`는 데이터 의존성으로 직렬화가 불가피하다. 단일 스레드 Actor가 gRPC
응답을 기다리는 동안 다음 프레임을 처리할 수 없다. 이 병목의 해법은 한 프레임 내 병렬화가
아니라 프레임 간 배치이며, 배치 최적화는 이 벤치마크의 범위 밖이다(구조 비교가 주제).

**문제 6 — np.append() O(N) 재할당 (Medium).**
Recorder를 `np.append()`로 누적하면 매 호출마다 전체 배열을 새로 할당·복사한다. amortized
O(1)이 아니라 매번 O(N)이며, 이는 Ray IPC와 무관한 Actor 내부 로컬 CPU 비용이다.

### [C] 이벤트 처리 경로 병목

**문제 7 — np.object_ SerDes (High).**
Recorder가 `dtype=np.object_` 배열(프레임 원본 + datetime + 트래킹 결과)로 구성된다. 숫자형
배열과 달리 `np.object_`는 Plasma Store의 zero-copy 경로를 탈 수 없어 이벤트 발생 시
`get_recorder()` 반환값 전체가 pickle 직렬화된다. 프레임 원본을 그대로 담고 있으므로 직렬화
대상 크기가 매우 커진다.

**문제 8 — get_recorder 태스크 기아 (High).**
Ray Actor는 기본적으로 단일 워커 스레드로 태스크를 순차 처리한다. `start_analysis()`가
`while True`로 절대 반환하지 않으므로, 뒤에 큐잉된 `get_recorder()`는 실행 기회를 영원히 얻지
못한다. 데드락(순환 대기)이 아니라 **무한 연기(indefinite postponement)** 다. 이벤트가 발생할
때마다 hang된 유틸리티 태스크가 누적되어 메모리가 지속 상승한다.

### [D] 코드 품질 (성능 벤치마크 범위 밖)

- **문제 9** — 미사용 dead code(`fast_inference.py`의 `@ray.remote`가 주입만 되고 호출되지 않음).
- **문제 10** — 동일 이름 태스크 중복 정의로 앞 정의가 dead code.
- **문제 11** — 관리 메서드의 불필요한 `time.sleep` 블로킹.
- **문제 12** — 런타임 이벤트 ON/OFF가 Actor에 반영되지 않는 미구현 메서드.

이 네 문제는 원 코드 고유의 품질 이슈로, 합성 워크로드로 재현할 성능 가치가 없어 벤치마크
대상에서 제외한다.

---

## 3. 문제 요약표

| # | 범주 | 문제 | 심각도 | 벤치마크 취급 |
|---|------|------|--------|--------------|
| 1 | 구조 | Stub + Ray 중첩 (CPU 경합/스케줄링/캐시/이중복사) | Critical | 재현 (구조 자체) |
| 2 | 구조 | Object Spilling → Thrashing → Actor 기아 | Critical | 재현 (object_store 축소) |
| 3 | 프레임 경로 | 매 프레임 Pull IPC | High | 재현 / B에서 개선 |
| 4 | 프레임 경로 | 버려지는 프레임도 ray.put() | Medium | 재현 / B에서 조건부 put |
| 5 | 프레임 경로 | 직렬 동기 gRPC 3회 | High | 구조만 재현 (배치 최적화 범위 밖) |
| 6 | 프레임 경로 | np.append() O(N) 재할당 | Medium | 재현 / B에서 deque |
| 7 | 이벤트 경로 | np.object_ SerDes 대용량 pickle | High | 재현 / B에서 ObjectRef |
| 8 | 이벤트 경로 | get_recorder 태스크 기아 | High | 재현 / B에서 async Actor |
| 9~12 | 코드 품질 | dead code, 중복 정의, sleep 블로킹, 미구현 메서드 | Low | 범위 밖 |

---

## 4. 이 분석이 벤치마크로 이어지는 지점

문제 1은 "Ray가 부적합하다"가 아니라 "**nested 구조가 부적합하다**"는 가설로 이어진다.
Stub Process를 제거하고 Ray를 독립(standalone)으로 기동하면 문제 1의 네 악순환이 함께
소멸한다는 것이 재설계의 출발점이다. 문제 2~8은 각각 개별 개선 항목(플래그)으로 분해해
기여도를 측정한다. 이 두 축(구조 전환 + 개별 개선)이 다음 문서
[`02-experiment-design.md`](02-experiment-design.md)의 실험 설계로 이어진다.

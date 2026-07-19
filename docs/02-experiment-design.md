# 02 — 실험 설계

> [`01-problem-analysis.md`](01-problem-analysis.md)에서 정리한 8개 성능 문제를 어떤 실험으로
> 재현·검증하는지 기술한다. 설계 배경 전체는 저장소 루트 `plan.md`에 있으며, 이 문서는 실행
> 관점에서 A/B 실험 구성, 개선 플래그, 스케일다운·재현 튜닝을 요약한다.

---

## 1. 실험 구성 개요

동일한 합성 워크로드를 두 구조에서 실행해 비교한다.

| 실험 | 구조 | 목적 |
|------|------|------|
| **실험 A (nested)** | Triton Python Backend 안에서 Ray를 nested로 기동 | 원 시스템의 병목 재현 (Before) |
| **실험 B (standalone)** | 독립 Ray 클러스터(`ray start --head`)에서 동일 파이프라인 실행 | 재설계 효과 측정 (After) |

두 실험은 **파이프라인 스테이지 코드·계측 코드·추론 지연 기준값을 동일 모듈에서 공유**한다.
차이는 오직 (1) Ray 기동 구조(nested vs standalone)와 (2) 실험 B에만 적용 가능한 개선 플래그뿐이다.
이렇게 하여 "구조 차이 외 변수"를 제거한다.

### 공통 워크로드

- 카메라당 1280×720×3 uint8 합성 프레임을 20fps로 페이싱(시드 고정 풀에서 복사 — 생성기 자체가
  병목이 되지 않도록).
- 파이프라인 순서: `detect(gRPC) → track(gRPC) → recorder 누적 → pose(gRPC) → 침입/배회(로컬)
  → 프레임 12장 누적 시 폭행/쓰러짐 추론(fire-and-forget)`.
- 추론 지연 기준값(ms)은 원 시스템 실측 로그의 스테이지별 p50에서 보정한 근사치를 config에
  기록한다: detect 15 / track 8 / pose 25 / 폭행 35×2회 / 쓰러짐 8×N회 / 침입·배회 0.2(로컬).
  모든 호출에 ±10% uniform 지터(카메라별 시드 고정)를 준다.
- mock 추론 서버는 동시성을 세마포어(또는 Triton instance count)로 제한해 실제 추론 서버의
  GPU 직렬화 큐잉을 모사한다. **부하 시의 꼬리 지연은 주입하지 않는다** — 그것은 실험에서
  경합으로 발생해야 할 결과이기 때문이다.

### 실험 A의 두 실행 경로

- **본 측정 경로 (Triton)**: 실제 tritonserver(CPU-only) 컨테이너의 Python Backend 안에서
  `ray.init()`을 호출한다. Triton이 실제로 만드는 프로세스/스레드 환경을 근사치로 뭉개지 않기
  위해서다. mock 추론 모델도 같은 Triton에 올려, 원 시스템의 "Stub 안 Ray Actor가 자기가 사는
  Triton으로 되돌아오는 gRPC" 루프까지 재현한다.
- **fallback 경로 (parent-process)**: Docker를 쓸 수 없는 환경을 위해, 부모 프로세스가 더미
  CPU 부하로 Stub을 모사하는 경로를 유지한다. 본 측정에서는 제외한다.

---

## 2. 개선 플래그 6종

실험 B는 문제 2~8의 개선을 **개별 플래그**로 켜고 끈다. 전부 off = `B0`(nested와 동일 로직을
standalone 위치에서만 실행), 전부 on = `B-all`. 이렇게 개별 토글해야 각 개선의 기여도를 분해
측정할 수 있다.

| 플래그 | 대상 문제 | off일 때 (A와 동일) | on일 때 |
|--------|----------|--------------------|---------|
| `use_deque_recorder` | 6 | `np.append` recorder (O(N) 재할당) | `deque(maxlen)` |
| `use_objectref_recorder` | 7 | 프레임 원본 저장 (전체 pickle) | `ObjectRef`만 저장 (경량 SerDes) |
| `use_conditional_put` | 4 | 무조건 `ray.put` | 큐 여유 있을 때만 put |
| `use_async_actor` | 8 | sync Actor 무한루프 (get_recorder 기아) | async Actor + `await` |
| `explicit_object_store` | 2 | `ray.init` 기본값 | `object_store_memory` 명시 |
| `set_cpu_affinity` | [3] (구조 가설) | 어피니티 미설정 | psutil로 프로세스별 코어 고정 |

앞의 다섯은 실험 B 고유의 재설계 항목이고, `set_cpu_affinity`는 A/B 공용 가설 플래그다. 원
시스템에 어피니티 설정이 전혀 없었으므로 기본 off이며, "설정만 고치면 되는 것 아니냐"는 반론에
대한 데이터를 얻기 위해 별도 토글로 둔다.

측정 매트릭스: `A` / `B0(전부 off)` / `B(하나씩 on)` / `B-all(전부 on)`.
- **B0와 A의 차이** = 구조 전환(nested → standalone) 단독 효과.
- **B-all과 B0의 차이** = 개별 개선 항목의 누적 효과.
- **B(하나씩 on)** = 각 플래그의 단독 기여.

---

## 3. 스케일다운과 재현 튜닝

측정 환경은 **로컬 WSL2 단일 머신(6코어 / 11GB RAM)** 이다. 원 시스템은 대용량 RAM을 갖춘
서버에서 다수 카메라를 처리했으므로, 단일 머신에서 동일 현상을 관찰하려면 규모를 축소하되
병목이 발현되는 조건은 유지해야 한다.

| 항목 | 원 시스템 | 이 레포 기본값 | 근거 |
|------|-----------|---------------|------|
| 카메라 수 | 다수 | 4 | 6코어에서 경합이 관찰되는 최소 규모 |
| 프레임 크기 | 1280×720×3 | 동일 | 프레임당 2.76MB라는 SerDes 스토리 유지 |
| FPS | 20 | 동일 | — |
| recorder 길이 | 600프레임 | 200 | 11GB 메모리 내에서 SerDes 현상 재현 |
| object_store_memory | 미설정(기본) | 실험 A: 256MB로 **의도적 축소** | Spilling 조기 유도 |

### object_store 256MB와 OOM 킬러 off의 근거

원 시스템은 대용량 RAM 서버에서 `object_store_memory` 무설정(사실상 넉넉한 기본값) 상태로
운영되었고, 그 조건에서 **메모리가 수 GB에서 17GB+ 까지 서서히 차오르며** Spilling과 Thrashing이
누적되는 현상이 관찰되었다. 이 "메모리가 여유롭게 쌓이다가 임계에서 무너지는" 궤적을 11GB짜리
단일 머신에서 시간 안에 재현하려면 두 가지 조정이 필요하다.

1. **object_store_memory 256MB로 축소** — Plasma Store 임계를 인위적으로 낮춰, 원 서버에서 수
   분~수십 분에 걸쳐 벌어질 Spilling을 짧은 관찰창 안에 조기 유도한다. 이는 현상을 **만들어내는**
   것이 아니라 원 서버의 대용량 조건에서 나타난 것과 동일한 메커니즘(Store 포화 → 스필 → 스왑
   I/O가 CPU 과점 → Actor 기아)을 **시간 압축**해 관찰하는 장치다. 값은 config에 명시하고 README
   Limitations에 "의도적 spill 유도"임을 기재한다.
2. **OOM 킬러 off** — object_store를 좁게 잡으면 스필된 객체와 hang 태스크 누적으로 프로세스
   RSS가 커진다. OS의 OOM 킬러가 중간에 프로세스를 종료하면 원 서버(대용량 RAM이라 OOM 킬이
   먼저 오지 않고 Thrashing이 먼저 심화됨)의 궤적과 달라진다. 킬러를 끄면 원 서버처럼 "죽지 않고
   계속 열화되는" 구간을 관찰할 수 있다.

정확한 값은 재현 튜닝에서 확정해 config에만 기록한다. 재현이 기본 파라미터로 되지 않을 경우의
튜닝 축은 카메라 수 증가 / object_store 추가 축소 / recorder 길이 증가다.

---

## 4. 측정 지표

| 지표 | 수집 방법 |
|------|----------|
| 스테이지별·e2e 지연 P50/P99 | `prometheus_client` Histogram + CSV 원본 덤프 |
| 프로세스별 CPU (분석 Actor 단독 분리) | psutil로 PID별 샘플링 → Gauge + CSV |
| 메모리 (RSS, Object Store 사용량) | psutil + Ray 내부 메트릭 |
| Spilling 발생량 | Ray spill 로그 파싱 |

- 계측 코드는 `benchmark/common/metrics.py` 하나로 통일해 두 실험이 동일 계측을 쓴다.
- 여러 프로세스(Analysis Actor, violence/falldown 태스크)가 하나의 CSV·Prometheus 서버에 안전하게
  기록하도록 `MetricsActor`(Ray Actor)로 중앙화한다.
- CSV는 항상 병행 덤프한다 — Grafana 없이도 `analysis/` 스크립트만으로 그래프를 재현할 수 있어야
  한다.
- **결과 수치는 전부 작업자 실측이다. 어떤 설계 문서에도 예상 수치를 쓰지 않는다.**

측정 결과는 [`03-results.md`](03-results.md)에 정리한다.

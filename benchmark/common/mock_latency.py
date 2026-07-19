"""모의 추론 지연·CPU 부하 시뮬레이션.

inference_mock/server.py(커스텀 gRPC 서버, 실험 B·fallback 경로)와
triton/models/*/1/model.py(Triton python backend 모의 모델, 실험 A 경로)가 서로 다른
지연 로직을 쓰면 두 실험의 "구조 차이 외 변수"가 생겨 비교가 무의미해지므로
이 모듈 하나로 통일한다.
"""

import random
import time


def jittered_sleep_seconds(
    latency_ms: float, jitter_ratio: float, rng: random.Random = None
) -> float:
    """지터가 적용된 지연(초)을 계산한다.

    Args:
        latency_ms: 기준 지연 (ms).
        jitter_ratio: ±비율 (예: 0.1 = ±10%).
        rng: 재현성이 필요한 호출자가 주입하는 random.Random. None이면 전역 random 사용.

    Returns:
        초 단위 지연 시간.
    """
    r = rng if rng is not None else random
    low = latency_ms * (1.0 - jitter_ratio)
    high = latency_ms * (1.0 + jitter_ratio)
    return r.uniform(low, high) / 1000.0


def busy_wait_ms(duration_ms: float):
    """duration_ms 동안 CPU를 점유하는 busy-wait.

    머신 성능차와 무관하게 재현 가능하도록 perf_counter 기준 시간으로 제어한다
    (plan.md §8 기술 결정).
    """
    if duration_ms <= 0:
        return
    end = time.perf_counter() + duration_ms / 1000.0
    while time.perf_counter() < end:
        pass


def simulate_inference(
    latency_ms: float,
    jitter_ratio: float,
    cpu_work_coefficient: float = 0.0,
    size_bytes: int = 0,
    rng: random.Random = None,
) -> float:
    """지터가 적용된 sleep + (옵션) 크기 비례 busy-wait을 수행한다.

    Returns:
        실측 경과 시간 (ms). 응답의 latency_ms 필드로 사용한다.
    """
    start = time.perf_counter()
    time.sleep(jittered_sleep_seconds(latency_ms, jitter_ratio, rng))
    if cpu_work_coefficient > 0 and size_bytes > 0:
        busy_wait_ms(size_bytes * cpu_work_coefficient)
    return (time.perf_counter() - start) * 1000.0

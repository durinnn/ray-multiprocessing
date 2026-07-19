"""실험 A — Docker 불가 환경을 위한 fallback: parent 프로세스에서 nested Ray 기동.

본 측정 경로가 아니다 (plan.md §3.2). 실제 tritonserver의 Stub Process 자체를 재현할
수는 없으므로, 대신 parent 프로세스가 더미 CPU 부하 스레드를 돌려 "부모 프로세스가
실행 중인 동안 CPU 일부를 점유한다"는 조건만 근사한다. Docker가 가능한 환경에서는
`scripts/run_nested.sh`(Triton 경로)를 사용할 것.
"""

import argparse
import logging
import threading
import time

import ray

from benchmark.common.config import load_config
from benchmark.common.metrics import MetricsActor
from benchmark.nested.actors import AnalysisActor, StreamActor

logger = logging.getLogger(__name__)


def _dummy_cpu_load(
    load_rate: float, update_interval: float, stop_event: threading.Event
):
    """Stub 프로세스가 점유했을 CPU 몫을 근사하는 더미 부하."""
    while not stop_event.is_set():
        busy_duration = update_interval * load_rate
        idle_duration = update_interval * (1 - load_rate)
        end = time.perf_counter() + busy_duration
        while time.perf_counter() < end:
            pass
        if idle_duration > 0:
            time.sleep(idle_duration)


def run(config_path: str = None, duration_seconds: int = None, num_cameras: int = None):
    logging.basicConfig(level=logging.INFO)
    config = load_config(config_path)

    duration = duration_seconds or config.nested_experiment["duration_seconds"]
    cameras = num_cameras or config.workload.num_cameras

    stop_event = threading.Event()
    cpu_thread = threading.Thread(
        target=_dummy_cpu_load,
        args=(
            config.parent.cpu_load_rate,
            config.parent.cpu_update_interval,
            stop_event,
        ),
        daemon=True,
    )
    cpu_thread.start()
    logger.info(
        f"parent 더미 CPU 부하 스레드 시작 (load_rate={config.parent.cpu_load_rate})"
    )

    ray.init(
        num_cpus=config.ray_nested.num_cpus,
        object_store_memory=config.ray_nested.object_store_memory,
        ignore_reinit_error=True,
    )
    logger.info(
        "ray.init 완료 (nested fallback, num_cpus=%s, object_store_memory=%s)",
        config.ray_nested.num_cpus,
        config.ray_nested.object_store_memory,
    )

    metrics_actor = MetricsActor.remote(config, experiment_name="nested_parent")

    stream_actors = []
    analysis_actors = []
    for camera_id in range(cameras):
        stream_actor = StreamActor.remote(camera_id, config.workload)
        analysis_actor = AnalysisActor.remote(
            camera_id,
            config,
            backend_kind="mock_grpc",
            triton_url=None,
            metrics_actor=metrics_actor,
        )
        stream_actor.start_streaming.remote()
        analysis_actor.start_analysis.remote(stream_actor)
        stream_actors.append(stream_actor)
        analysis_actors.append(analysis_actor)

    logger.info(f"{cameras}대 카메라 파이프라인 기동, {duration}초 동안 실행")
    time.sleep(duration)

    stop_event.set()
    ray.get(metrics_actor.close.remote())
    # AnalysisActor의 start_analysis()는 반환하지 않으므로 graceful stop이 불가능하다
    # (문제 8 재현 그 자체). ray.shutdown()으로 클러스터 전체를 정리한다.
    ray.shutdown()
    logger.info("실험 A(fallback parent) 종료 — ray.shutdown() 완료")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="실험 A: nested Ray fallback (parent process)"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--duration", type=int, default=None)
    parser.add_argument("--cameras", type=int, default=None)
    args = parser.parse_args()
    run(args.config, args.duration, args.cameras)

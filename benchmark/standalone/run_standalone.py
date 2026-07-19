"""실험 B(standalone) 실행 엔트리포인트.

Triton python backend 밖의 독립 Ray 프로세스(ray.init로 로컬 클러스터 기동)에서
nested와 동일한 파이프라인을 돌린다. 추론 대상은 inference_mock 서버(mock_grpc
백엔드)로, Docker가 필요 없다. 개선 플래그는 config/default.yaml의 standalone_flags
기본값을 쓰되, CLI로 개별 override 하거나 --all-on으로 전부 켤 수 있다 (측정 매트릭스
B0 / B(하나씩 on) / B-all 구성용).
"""

import argparse
import dataclasses
import logging
import time

import ray

from benchmark.common.config import load_config
from benchmark.common.metrics import MetricsActor
from benchmark.standalone.actors import StreamActor, make_analysis_actor

logger = logging.getLogger(__name__)

FLAG_NAMES = [
    "use_deque_recorder",
    "use_objectref_recorder",
    "use_conditional_put",
    "use_async_actor",
    "explicit_object_store",
    "set_cpu_affinity",
]


def resolve_flags(base_flags, args):
    """config 기본 플래그 위에 CLI override를 얹은 새 플래그 객체를 만든다."""
    values = dataclasses.asdict(base_flags)
    if getattr(args, "all_on", False):
        values = {name: True for name in values}
    for name in FLAG_NAMES:
        if getattr(args, name, False):
            values[name] = True
    return dataclasses.replace(base_flags, **values)


def run(
    config_path: str = None,
    duration_seconds: int = None,
    num_cameras: int = None,
    flags=None,
):
    logging.basicConfig(level=logging.INFO)
    config = load_config(config_path)

    flags = flags if flags is not None else config.standalone_flags
    duration = duration_seconds or config.standalone_experiment["duration_seconds"]
    cameras = num_cameras or config.workload.num_cameras

    # explicit_object_store(문제 2): on이면 object_store_memory를 명시, off면 기본값.
    ray_kwargs = dict(num_cpus=config.ray_standalone.num_cpus, ignore_reinit_error=True)
    if flags.explicit_object_store:
        ray_kwargs["object_store_memory"] = config.ray_standalone.object_store_memory
    ray.init(**ray_kwargs)
    logger.info(
        "ray.init 완료 (standalone, num_cpus=%s, explicit_object_store=%s)",
        config.ray_standalone.num_cpus,
        flags.explicit_object_store,
    )
    logger.info("standalone 플래그: %s", dataclasses.asdict(flags))

    metrics_actor = MetricsActor.remote(config, experiment_name="standalone")
    analysis_cls = make_analysis_actor(flags.use_async_actor)

    stream_actors = []
    analysis_actors = []
    for camera_id in range(cameras):
        # 어피니티 코어는 카메라별로 stream/analysis에 서로 다른 코어를 배정한다
        # (set_cpu_affinity off면 무시된다).
        stream_actor = StreamActor.remote(
            camera_id,
            config.workload,
            use_conditional_put=flags.use_conditional_put,
            set_cpu_affinity=flags.set_cpu_affinity,
            affinity_core=camera_id * 2 + 1,
        )
        analysis_actor = analysis_cls.remote(
            camera_id,
            config,
            "mock_grpc",
            None,
            flags,
            metrics_actor,
            camera_id * 2,
        )
        stream_actor.start_streaming.remote()
        analysis_actor.start_analysis.remote(stream_actor)
        stream_actors.append(stream_actor)
        analysis_actors.append(analysis_actor)

    logger.info("%d대 카메라 standalone 파이프라인 기동, %d초 실행", cameras, duration)
    time.sleep(duration)

    ray.get(metrics_actor.close.remote())
    # sync Actor의 start_analysis()는 반환하지 않으므로(문제 8) graceful stop이 불가.
    # ray.shutdown()으로 클러스터 전체를 정리한다 (async여도 동일하게 정리).
    ray.shutdown()
    logger.info("실험 B(standalone) 종료 — ray.shutdown() 완료")


def _build_parser():
    parser = argparse.ArgumentParser(
        description="실험 B: standalone Ray 파이프라인 + 개선 플래그"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--duration", type=int, default=None)
    parser.add_argument("--cameras", type=int, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="smoke 설정(짧은 duration·1대 카메라)으로 실행",
    )
    parser.add_argument(
        "--all-on",
        dest="all_on",
        action="store_true",
        help="개선 플래그를 전부 on (B-all)",
    )
    for name in FLAG_NAMES:
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            dest=name,
            action="store_true",
            help=f"{name} 플래그를 on",
        )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    _config = load_config(args.config)

    _duration = args.duration
    _cameras = args.cameras
    if args.smoke:
        _duration = _duration or _config.smoke.duration_seconds
        _cameras = _cameras or _config.smoke.num_cameras

    _flags = resolve_flags(_config.standalone_flags, args)
    run(
        config_path=args.config,
        duration_seconds=_duration,
        num_cameras=_cameras,
        flags=_flags,
    )

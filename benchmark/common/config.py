"""Configuration loader for benchmark experiments."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import yaml


@dataclass
class InferenceMockConfig:
    """Mock inference server configuration."""

    host: str
    port: int
    latency_detect: float
    latency_track: float
    latency_pose: float
    latency_falldown: float
    latency_violence: float
    violence_num_calls: int
    jitter_ratio: float
    max_concurrency: int
    cpu_work_detect: float
    cpu_work_track: float
    cpu_work_pose: float


@dataclass
class LocalStagesConfig:
    """침입/배회 — 원본이 규칙 기반 로컬 연산이므로 gRPC가 아닌 Actor 내 busy-wait로 모의."""

    latency_intrusion: float
    latency_loitering: float


@dataclass
class NestedRayConfig:
    """Ray configuration for nested experiment."""

    num_cpus: int
    object_store_memory: int


@dataclass
class StandaloneRayConfig:
    """Ray configuration for standalone experiment."""

    num_cpus: int
    object_store_memory: int


@dataclass
class StandaloneFlagsConfig:
    """실험 B 개선 플래그 (plan.md §3.3). 전부 off = B0(nested와 동일 로직)."""

    use_deque_recorder: bool
    use_objectref_recorder: bool
    use_conditional_put: bool
    use_async_actor: bool
    explicit_object_store: bool
    set_cpu_affinity: bool


@dataclass
class MetricsConfig:
    """Metrics collection configuration."""

    csv_output: str
    prometheus_port: int
    sample_interval: float
    latency_buckets: list


@dataclass
class ParentConfig:
    """Parent process (Stub) configuration for nested experiment."""

    cpu_load_rate: float
    cpu_update_interval: float


@dataclass
class WorkloadConfig:
    """Experiment workload configuration."""

    num_cameras: int
    frame_width: int
    frame_height: int
    frame_channels: int
    fps: int
    recorder_max_frames: int
    batch_frame_count: int
    workload_mode: str
    event_trigger_interval: float


@dataclass
class SmokeTestConfig:
    """Smoke test configuration."""

    num_cameras: int
    duration_seconds: int


@dataclass
class ExperimentConfig:
    """Main configuration for benchmark experiments."""

    workload: WorkloadConfig
    ray_nested: NestedRayConfig
    ray_standalone: StandaloneRayConfig
    inference_mock: InferenceMockConfig
    local_stages: LocalStagesConfig
    metrics: MetricsConfig
    parent: ParentConfig
    standalone_flags: StandaloneFlagsConfig
    smoke: SmokeTestConfig
    nested_experiment: Dict[str, Any]
    standalone_experiment: Dict[str, Any]


def load_config(config_path: str = None) -> ExperimentConfig:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config YAML. If None, uses config/default.yaml.

    Returns:
        ExperimentConfig dataclass instance.
    """
    if config_path is None:
        repo_root = Path(__file__).parent.parent.parent
        config_path = repo_root / "config" / "default.yaml"

    with open(config_path) as f:
        data = yaml.safe_load(f)

    workload = WorkloadConfig(
        num_cameras=data["num_cameras"],
        frame_width=data["frame_width"],
        frame_height=data["frame_height"],
        frame_channels=data["frame_channels"],
        fps=data["fps"],
        recorder_max_frames=data["recorder_max_frames"],
        batch_frame_count=data["batch_frame_count"],
        workload_mode=data.get("workload_mode", "idle"),
        event_trigger_interval=data.get("event_trigger_interval", 5.0),
    )

    ray_nested = NestedRayConfig(
        num_cpus=data["nested"]["num_cpus"],
        object_store_memory=int(data["nested"]["object_store_memory"]),
    )

    ray_standalone = StandaloneRayConfig(
        num_cpus=data["standalone"]["num_cpus"],
        object_store_memory=int(data["standalone"]["object_store_memory"]),
    )

    inference_mock = InferenceMockConfig(
        host=data["inference_mock"]["host"],
        port=data["inference_mock"]["port"],
        latency_detect=data["inference_mock"]["latency_detect"],
        latency_track=data["inference_mock"]["latency_track"],
        latency_pose=data["inference_mock"]["latency_pose"],
        latency_falldown=data["inference_mock"]["latency_falldown"],
        latency_violence=data["inference_mock"]["latency_violence"],
        violence_num_calls=data["inference_mock"]["violence_num_calls"],
        jitter_ratio=data["inference_mock"]["jitter_ratio"],
        max_concurrency=data["inference_mock"]["max_concurrency"],
        cpu_work_detect=data["inference_mock"]["cpu_work_detect"],
        cpu_work_track=data["inference_mock"]["cpu_work_track"],
        cpu_work_pose=data["inference_mock"]["cpu_work_pose"],
    )

    local_stages = LocalStagesConfig(
        latency_intrusion=data["local_stages"]["latency_intrusion"],
        latency_loitering=data["local_stages"]["latency_loitering"],
    )

    metrics = MetricsConfig(
        csv_output=data["metrics"]["csv_output"],
        prometheus_port=data["metrics"]["prometheus_port"],
        sample_interval=data["metrics"]["sample_interval"],
        latency_buckets=data["metrics"]["latency_buckets"],
    )

    parent = ParentConfig(
        cpu_load_rate=data["parent"]["cpu_load_rate"],
        cpu_update_interval=data["parent"]["cpu_update_interval"],
    )

    flags_data = data.get("standalone_flags", {})
    standalone_flags = StandaloneFlagsConfig(
        use_deque_recorder=bool(flags_data.get("use_deque_recorder", False)),
        use_objectref_recorder=bool(flags_data.get("use_objectref_recorder", False)),
        use_conditional_put=bool(flags_data.get("use_conditional_put", False)),
        use_async_actor=bool(flags_data.get("use_async_actor", False)),
        explicit_object_store=bool(flags_data.get("explicit_object_store", False)),
        set_cpu_affinity=bool(flags_data.get("set_cpu_affinity", False)),
    )

    smoke = SmokeTestConfig(
        num_cameras=data["smoke"]["num_cameras"],
        duration_seconds=data["smoke"]["duration_seconds"],
    )

    return ExperimentConfig(
        workload=workload,
        ray_nested=ray_nested,
        ray_standalone=ray_standalone,
        inference_mock=inference_mock,
        local_stages=local_stages,
        metrics=metrics,
        parent=parent,
        standalone_flags=standalone_flags,
        smoke=smoke,
        nested_experiment=data.get("nested_experiment", {}),
        standalone_experiment=data.get("standalone_experiment", {}),
    )

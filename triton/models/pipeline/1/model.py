"""Triton python backend: 원본 Stub 역할 — 문제 1의 재현 대상 그 자체.

initialize()에서 nested Ray를 기동하고 카메라별 StreamActor/AnalysisActor를
fire-and-forget으로 띄운다. execute()는 헬스체크 이상의 역할이 없으며, 실제
파이프라인은 백그라운드 Ray Actor에서 이 요청/응답과 무관하게 상시 동작한다
(원본 Stub도 별도 요청 트리거 없이 상시 동작하는 구조였다 — research.md 문제 1).
"""

import logging

import numpy as np
import ray
import triton_python_backend_utils as pb_utils

from benchmark.common.config import load_config
from benchmark.common.metrics import MetricsActor
from benchmark.nested.actors import AnalysisActor, StreamActor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TritonPythonModel:
    def initialize(self, args):
        self.config = load_config()

        # 문제 1: Triton Stub(python backend) 프로세스 내부에 Ray를 nested 기동한다.
        # object_store_memory를 의도적으로 작게 설정해 Object Spilling을 유도한다
        # (plan.md §3.1 스케일다운 — 값은 config/default.yaml에 명시).
        ray.init(
            num_cpus=self.config.ray_nested.num_cpus,
            object_store_memory=self.config.ray_nested.object_store_memory,
            ignore_reinit_error=True,
        )
        logger.info(
            "nested ray.init 완료 (num_cpus=%s, object_store_memory=%s)",
            self.config.ray_nested.num_cpus,
            self.config.ray_nested.object_store_memory,
        )

        self.metrics_actor = MetricsActor.remote(
            self.config, experiment_name="nested_triton"
        )

        self.stream_actors = []
        self.analysis_actors = []
        for camera_id in range(self.config.workload.num_cameras):
            stream_actor = StreamActor.remote(camera_id, self.config.workload)
            analysis_actor = AnalysisActor.remote(
                camera_id,
                self.config,
                backend_kind="triton",
                triton_url="localhost:8001",
                metrics_actor=self.metrics_actor,
            )
            # fire-and-forget: 두 Actor 모두 반환하지 않는 무한 루프 메서드를 실행한다.
            stream_actor.start_streaming.remote()
            analysis_actor.start_analysis.remote(stream_actor)
            self.stream_actors.append(stream_actor)
            self.analysis_actors.append(analysis_actor)

        logger.info(f"{self.config.workload.num_cameras}대 카메라 파이프라인 기동 완료")

    def execute(self, requests):
        responses = []
        for _ in requests:
            healthy = pb_utils.Tensor("HEALTHY", np.array([True], dtype=np.bool_))
            responses.append(pb_utils.InferenceResponse(output_tensors=[healthy]))
        return responses

    def finalize(self):
        logger.info("pipeline 모델 종료 — ray.shutdown() 호출")
        ray.shutdown()

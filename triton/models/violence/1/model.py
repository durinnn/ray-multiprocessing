"""Triton python backend: 모의 violence(폭행) 모델.

클라이언트(AnalysisActor)가 이 모델을 violence_num_calls회 직렬 호출해 2모델
앙상블을 재현한다. 이 모델 자체는 배치 1회 통과(체크포인트 1개)만 모의한다.
"""

import numpy as np
import triton_python_backend_utils as pb_utils

from benchmark.common.config import load_config
from benchmark.common.mock_latency import simulate_inference


class TritonPythonModel:
    def initialize(self, args):
        self.config = load_config().inference_mock

    def execute(self, requests):
        responses = []
        for request in requests:
            batch_size = int(
                pb_utils.get_input_tensor_by_name(request, "BATCH_SIZE").as_numpy()[0]
            )

            # 원 시스템의 배치 추론과 동일하게 프레임 크기 비례 CPU 부하는 두지 않는다
            # (inference_mock/server.py의 batch_violence와 동일 취급).
            latency_ms = simulate_inference(
                self.config.latency_violence, self.config.jitter_ratio
            )

            out_count = pb_utils.Tensor(
                "RESULT_COUNT", np.array([batch_size], dtype=np.int32)
            )
            out_latency = pb_utils.Tensor(
                "LATENCY_MS", np.array([latency_ms], dtype=np.float32)
            )
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[out_count, out_latency])
            )
        return responses

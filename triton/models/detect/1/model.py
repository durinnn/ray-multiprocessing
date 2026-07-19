"""Triton python backend: 모의 detect 모델.

benchmark/common/mock_latency.py를 그대로 써서 inference_mock의 커스텀 gRPC 서버
(fallback 경로)와 동일한 지연 기준값을 재현한다 — 실험 A/B 비교에 "구조 차이 외
변수"가 섞이지 않도록 하기 위함이다.
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
            frame = pb_utils.get_input_tensor_by_name(request, "FRAME").as_numpy()
            frame_size = frame.shape[0]

            latency_ms = simulate_inference(
                self.config.latency_detect,
                self.config.jitter_ratio,
                self.config.cpu_work_detect,
                frame_size,
            )

            # 실제 좌표값은 이 벤치마크의 관심사가 아니므로 개수만 왕복한다
            num_boxes = int(np.random.randint(5, 11))

            out_count = pb_utils.Tensor(
                "OUTPUT_COUNT", np.array([num_boxes], dtype=np.int32)
            )
            out_latency = pb_utils.Tensor(
                "LATENCY_MS", np.array([latency_ms], dtype=np.float32)
            )
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[out_count, out_latency])
            )
        return responses

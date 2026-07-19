"""Triton python backend: 모의 pose 모델. detect/1/model.py와 지연 로직을 공유한다."""

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
            input_count = int(
                pb_utils.get_input_tensor_by_name(request, "INPUT_COUNT").as_numpy()[0]
            )
            frame_size = frame.shape[0]

            latency_ms = simulate_inference(
                self.config.latency_pose,
                self.config.jitter_ratio,
                self.config.cpu_work_pose,
                frame_size,
            )

            out_count = pb_utils.Tensor(
                "OUTPUT_COUNT", np.array([input_count], dtype=np.int32)
            )
            out_latency = pb_utils.Tensor(
                "LATENCY_MS", np.array([latency_ms], dtype=np.float32)
            )
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[out_count, out_latency])
            )
        return responses

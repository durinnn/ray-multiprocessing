"""Triton python backend: 모의 falldown(쓰러짐) 모델.

클라이언트가 프레임별로 이 모델을 최대 batch_frame_count회 직렬 호출하며 이전
응답의 STATE_BLOB을 다음 요청에 실어 보낸다 (상태 체이닝 재현).
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
            frame_id = int(
                pb_utils.get_input_tensor_by_name(request, "FRAME_ID").as_numpy()[0]
            )
            state_blob = pb_utils.get_input_tensor_by_name(
                request, "STATE_BLOB"
            ).as_numpy()
            frame_size = frame.shape[0]

            latency_ms = simulate_inference(
                self.config.latency_falldown,
                self.config.jitter_ratio,
                size_bytes=frame_size,
            )

            # 누적 상태를 흉내내기 위해 이전 state_blob에 이번 frame_id를 이어붙인다
            frame_id_bytes = np.array([frame_id], dtype=np.int32).view(np.uint8)
            updated_state = np.concatenate([state_blob, frame_id_bytes]).astype(
                np.uint8
            )

            out_state = pb_utils.Tensor("STATE_BLOB", updated_state)
            out_detected = pb_utils.Tensor(
                "DETECTED", np.array([False], dtype=np.bool_)
            )
            out_latency = pb_utils.Tensor(
                "LATENCY_MS", np.array([latency_ms], dtype=np.float32)
            )
            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[out_state, out_detected, out_latency]
                )
            )
        return responses

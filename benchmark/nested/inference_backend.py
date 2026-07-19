"""AnalysisActor가 사용하는 추론 백엔드 인터페이스.

실험 A는 두 가지 경로를 지원한다 (plan.md §3.2):
- TritonInferenceBackend: 실제 tritonserver에 올라간 모의 모델을 tritonclient.grpc로 호출.
  본 측정 경로. Stub 역할의 python backend 안에서 nested Ray로 기동된 Actor가 자기가
  사는 Triton으로 되돌아오는 gRPC 루프까지 재현한다.
- MockGrpcInferenceBackend: inference_mock의 커스텀 gRPC 서버를 호출. Docker 불가 환경을
  위한 fallback parent-process 경로 (본 측정에서는 제외, plan.md §3.2/§8).

두 백엔드 모두 동일한 config/default.yaml 지연 기준값을 쓰므로 실험 A/B 비교의
"구조 차이 외 변수"가 늘지 않는다.
"""

import logging
from typing import Any, Dict, List, Protocol, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class InferenceBackend(Protocol):
    """AnalysisActor가 호출하는 추론 인터페이스."""

    def detect(
        self, frame: np.ndarray, frame_id: int
    ) -> Tuple[List[Dict[str, Any]], float]: ...

    def track(
        self, detections: List[Dict[str, Any]], frame: np.ndarray, frame_id: int
    ) -> Tuple[List[Dict[str, Any]], float]: ...

    def pose(
        self, tracks: List[Dict[str, Any]], frame: np.ndarray, frame_id: int
    ) -> Tuple[List[Dict[str, Any]], float]: ...

    def violence(
        self, frames: List[np.ndarray], width: int, height: int
    ) -> Tuple[List[Dict[str, Any]], float]:
        """2모델 앙상블 재현. 내부에서 violence_num_calls회 직렬 호출한 총 지연을 반환한다."""
        ...

    def falldown(self, frames: List[np.ndarray]) -> Tuple[List[Dict[str, Any]], float]:
        """프레임별 상태 체이닝 재현. 내부에서 프레임당 1회씩 직렬 호출한 총 지연을 반환한다."""
        ...


class MockGrpcInferenceBackend:
    """inference_mock 커스텀 gRPC 서버 호출 (fallback parent-process 경로)."""

    def __init__(self, config):
        from benchmark.common.stages import InferenceClient

        self.config = config
        self.client = InferenceClient(
            config.inference_mock.host, config.inference_mock.port
        )

    def detect(self, frame, frame_id):
        return self.client.detect(frame, frame_id)

    def track(self, detections, frame, frame_id):
        return self.client.track(detections, frame, frame_id)

    def pose(self, tracks, frame, frame_id):
        return self.client.pose(tracks, frame, frame_id)

    def violence(self, frames, width, height):
        results: List[Dict[str, Any]] = []
        total_latency_ms = 0.0
        for _ in range(self.config.inference_mock.violence_num_calls):
            results, latency_ms = self.client.batch_violence(frames, width, height)
            total_latency_ms += latency_ms
        return results, total_latency_ms

    def falldown(self, frames):
        state_blob = b""
        results: List[Dict[str, Any]] = []
        total_latency_ms = 0.0
        for frame_id, frame in enumerate(frames):
            result, state_blob, latency_ms = self.client.falldown(
                frame, frame_id, state_blob
            )
            results.append(result)
            total_latency_ms += latency_ms
        return results, total_latency_ms


class TritonInferenceBackend:
    """같은 Triton 서버에 올라간 모의 모델을 tritonclient.grpc로 호출 (본 측정 경로).

    Triton 모델 IO는 구조적 병목(SerDes 크기, 왕복 횟수, busy-wait 시간) 재현에
    필요한 최소한만 담는다 — 실제 bbox/keypoint 좌표값은 이 벤치마크의 관심사가
    아니므로 모델별 결과는 개수(count)만 왕복한다 (triton/models 참고).
    """

    def __init__(
        self, config, url: str = "localhost:8001", ready_timeout_s: float = 120.0
    ):
        import time

        import tritonclient.grpc as grpcclient

        self.config = config
        self.url = url
        self.client = grpcclient.InferenceServerClient(url=url)
        self._grpcclient = grpcclient

        # pipeline 모델의 initialize() 안에서 기동된 Actor가 이 백엔드를 만드는데,
        # Triton gRPC 엔드포인트는 모든 모델 로드가 끝난 뒤에야 리슨을 시작한다.
        # 준비될 때까지 대기하지 않으면 첫 호출이 connection refused로 죽는다.
        deadline = time.monotonic() + ready_timeout_s
        while True:
            try:
                if self.client.is_server_ready():
                    break
            except Exception:
                pass
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Triton gRPC({url})가 {ready_timeout_s}초 내에 준비되지 않았습니다"
                )
            time.sleep(1.0)

    def _infer(self, model_name: str, inputs, outputs=None):
        response = self.client.infer(
            model_name=model_name, inputs=inputs, outputs=outputs
        )
        return response

    def _frame_input(self, name: str, frame: np.ndarray):
        grpcclient = self._grpcclient
        flat = np.ascontiguousarray(frame).reshape(-1)
        tensor = grpcclient.InferInput(name, flat.shape, "UINT8")
        tensor.set_data_from_numpy(flat)
        return tensor

    def _scalar_input(self, name: str, value: int, dtype: str = "INT32"):
        grpcclient = self._grpcclient
        arr = np.array([value], dtype=np.int32 if dtype == "INT32" else np.int64)
        tensor = grpcclient.InferInput(name, arr.shape, dtype)
        tensor.set_data_from_numpy(arr)
        return tensor

    def _bytes_input(self, name: str, data: bytes):
        grpcclient = self._grpcclient
        arr = (
            np.frombuffer(data, dtype=np.uint8)
            if data
            else np.zeros((0,), dtype=np.uint8)
        )
        tensor = grpcclient.InferInput(name, arr.shape, "UINT8")
        tensor.set_data_from_numpy(arr)
        return tensor

    def detect(self, frame, frame_id):
        inputs = [
            self._frame_input("FRAME", frame),
            self._scalar_input("FRAME_ID", frame_id),
        ]
        response = self._infer("detect", inputs)
        count = int(response.as_numpy("OUTPUT_COUNT")[0])
        latency_ms = float(response.as_numpy("LATENCY_MS")[0])
        detections = [{"index": i} for i in range(count)]
        return detections, latency_ms

    def track(self, detections, frame, frame_id):
        inputs = [
            self._frame_input("FRAME", frame),
            self._scalar_input("FRAME_ID", frame_id),
            self._scalar_input("INPUT_COUNT", len(detections)),
        ]
        response = self._infer("track", inputs)
        count = int(response.as_numpy("OUTPUT_COUNT")[0])
        latency_ms = float(response.as_numpy("LATENCY_MS")[0])
        tracks = [{"track_id": i} for i in range(count)]
        return tracks, latency_ms

    def pose(self, tracks, frame, frame_id):
        inputs = [
            self._frame_input("FRAME", frame),
            self._scalar_input("FRAME_ID", frame_id),
            self._scalar_input("INPUT_COUNT", len(tracks)),
        ]
        response = self._infer("pose", inputs)
        count = int(response.as_numpy("OUTPUT_COUNT")[0])
        latency_ms = float(response.as_numpy("LATENCY_MS")[0])
        poses = [{"track_id": i} for i in range(count)]
        return poses, latency_ms

    def violence(self, frames, width, height):
        flat = np.ascontiguousarray(np.stack(frames)).reshape(-1)
        grpcclient = self._grpcclient
        frames_tensor = grpcclient.InferInput("FRAMES", flat.shape, "UINT8")
        frames_tensor.set_data_from_numpy(flat)
        inputs = [frames_tensor, self._scalar_input("BATCH_SIZE", len(frames))]

        results: List[Dict[str, Any]] = []
        total_latency_ms = 0.0
        for _ in range(self.config.inference_mock.violence_num_calls):
            response = self._infer("violence", inputs)
            count = int(response.as_numpy("RESULT_COUNT")[0])
            total_latency_ms += float(response.as_numpy("LATENCY_MS")[0])
            results = [{"detected": False} for _ in range(count)]
        return results, total_latency_ms

    def falldown(self, frames):
        state_blob = b""
        results: List[Dict[str, Any]] = []
        total_latency_ms = 0.0
        for frame_id, frame in enumerate(frames):
            inputs = [
                self._frame_input("FRAME", frame),
                self._scalar_input("FRAME_ID", frame_id),
                self._bytes_input("STATE_BLOB", state_blob),
            ]
            response = self._infer("falldown", inputs)
            state_blob = response.as_numpy("STATE_BLOB").tobytes()
            detected = bool(response.as_numpy("DETECTED")[0])
            total_latency_ms += float(response.as_numpy("LATENCY_MS")[0])
            results.append({"detected": detected})
        return results, total_latency_ms

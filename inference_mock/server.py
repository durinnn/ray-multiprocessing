"""Mock inference server for benchmark experiments.

Simulates inference latency and CPU work without requiring actual models.
"""

import threading
import time
import logging
from concurrent import futures

import grpc
import numpy as np

from benchmark.common.mock_latency import simulate_inference

# Note: *_pb2 and *_pb2_grpc imports will be added after proto compilation
# from inference_pb2 import (
#     DetectResponse, BBox, TrackResponse, TrackObject,
#     PoseResponse, PoseResult, PoseKeypoint, BatchViolenceResponse, ViolenceResult,
#     FalldownResponse
# )
# from inference_pb2_grpc import InferenceServiceServicer, add_InferenceServiceServicer_to_server


logger = logging.getLogger(__name__)


class MockInferenceServicer:
    """Mock inference server implementing inference.InferenceServiceServicer."""

    def __init__(self, config):
        """Initialize with benchmark config.

        Args:
            config: ExperimentConfig with inference_mock settings.
        """
        self.config = config.inference_mock
        self.logger = logging.getLogger(__name__)
        # 실제 추론 서버의 GPU 직렬화가 만드는 큐잉 지연을 모사 (plan.md §3.1)
        self._semaphore = threading.Semaphore(self.config.max_concurrency)

    def Detect(self, request, context):
        """Simulate detection inference."""
        from inference_pb2 import DetectResponse, BBox

        frame_size = request.width * request.height * 3
        with self._semaphore:
            latency_ms = simulate_inference(
                self.config.latency_detect,
                self.config.jitter_ratio,
                self.config.cpu_work_detect,
                frame_size,
            )

        # Generate dummy detections (5-10 boxes per frame)
        num_boxes = np.random.randint(5, 11)
        bboxes = [
            BBox(
                x1=np.random.uniform(0, request.width),
                y1=np.random.uniform(0, request.height),
                x2=np.random.uniform(0, request.width),
                y2=np.random.uniform(0, request.height),
                confidence=np.random.uniform(0.5, 1.0),
                class_id=int(np.random.randint(0, 5)),
            )
            for _ in range(num_boxes)
        ]
        return DetectResponse(
            detections=bboxes, frame_id=request.frame_id, latency_ms=latency_ms
        )

    def Track(self, request, context):
        """Simulate tracking inference."""
        from inference_pb2 import TrackResponse, TrackObject

        frame_size = request.width * request.height * 3
        with self._semaphore:
            latency_ms = simulate_inference(
                self.config.latency_track,
                self.config.jitter_ratio,
                self.config.cpu_work_track,
                frame_size,
            )

        # Convert detections to tracks (dummy)
        tracks = [
            TrackObject(
                track_id=i,
                x=bbox.x1,
                y=bbox.y1,
                w=bbox.x2 - bbox.x1,
                h=bbox.y2 - bbox.y1,
                confidence=bbox.confidence,
            )
            for i, bbox in enumerate(request.detections)
        ]
        return TrackResponse(
            tracks=tracks, frame_id=request.frame_id, latency_ms=latency_ms
        )

    def Pose(self, request, context):
        """Simulate pose estimation inference."""
        from inference_pb2 import PoseResponse, PoseResult, PoseKeypoint

        frame_size = request.width * request.height * 3
        with self._semaphore:
            latency_ms = simulate_inference(
                self.config.latency_pose,
                self.config.jitter_ratio,
                self.config.cpu_work_pose,
                frame_size,
            )

        # Generate dummy keypoints (17 for COCO format)
        poses = []
        for track in request.tracks:
            keypoints = [
                PoseKeypoint(
                    x=np.random.uniform(0, request.width),
                    y=np.random.uniform(0, request.height),
                    confidence=np.random.uniform(0.5, 1.0),
                )
                for _ in range(17)
            ]
            poses.append(PoseResult(track_id=track.track_id, keypoints=keypoints))

        return PoseResponse(
            poses=poses, frame_id=request.frame_id, latency_ms=latency_ms
        )

    def BatchViolence(self, request, context):
        """Simulate one ensemble-checkpoint pass over an accumulated batch.

        폭행 = 클라이언트가 이 RPC를 violence_num_calls회(기본 2회) 직렬 호출해
        2모델 앙상블을 재현한다 (서버는 단일 호출당 한 체크포인트만 모의).
        """
        from inference_pb2 import BatchViolenceResponse, ViolenceResult

        with self._semaphore:
            latency_ms = simulate_inference(
                self.config.latency_violence, self.config.jitter_ratio
            )

        results = [
            ViolenceResult(detected=False, confidence=0.0)
            for _ in range(request.batch_size)
        ]
        return BatchViolenceResponse(results=results, latency_ms=latency_ms)

    def Falldown(self, request, context):
        """Simulate one frame's worth of falldown inference with state chaining.

        쓰러짐 = 클라이언트가 프레임별로 이 RPC를 최대 batch_frame_count회 직렬 호출하며
        이전 응답의 state_blob을 다음 요청에 실어 보낸다 (상태 체이닝 재현).
        """
        from inference_pb2 import FalldownResponse

        frame_size = request.width * request.height * 3
        with self._semaphore:
            latency_ms = simulate_inference(
                self.config.latency_falldown,
                self.config.jitter_ratio,
                size_bytes=frame_size,
            )

        # 누적 상태를 흉내내기 위해 이전 state_blob에 이번 frame_id를 이어붙인다
        updated_state = request.state_blob + request.frame_id.to_bytes(
            4, "little", signed=True
        )
        return FalldownResponse(
            detected=False,
            confidence=0.0,
            state_blob=updated_state,
            latency_ms=latency_ms,
        )


def serve(config, port=None):
    """Start mock inference server.

    Args:
        config: ExperimentConfig.
        port: Override port (optional).
    """
    if port is None:
        port = config.inference_mock.port

    # Must be called after proto compilation
    try:
        from inference_pb2_grpc import add_InferenceServiceServicer_to_server
    except ImportError as e:
        logger.error("Proto files not generated. Run: python -m grpc_tools.protoc ...")
        raise e

    servicer = MockInferenceServicer(config)

    # Create a gRPC server. 폭행 배치 요청(최대 batch_frame_count장의 원본 프레임)이
    # 기본 4MB 한도를 넘기므로 메시지 크기 한도를 상향한다.
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
        ],
    )
    add_InferenceServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")

    logger.info(f"Starting mock inference server on port {port}")
    server.start()

    try:
        # Keep server running
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        logger.info("Shutting down server")
        server.stop(0)


if __name__ == "__main__":
    import sys
    from benchmark.common.config import load_config

    logging.basicConfig(level=logging.INFO)
    config = load_config()

    if len(sys.argv) > 1:
        port = int(sys.argv[1])
        serve(config, port)
    else:
        serve(config)

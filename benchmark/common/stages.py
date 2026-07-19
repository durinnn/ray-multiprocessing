"""Pipeline stages for synthetic benchmark workload.

Stages: detect, track, pose, batch_inference, record.
Each calls inference_mock gRPC server.
"""

import logging
from typing import Any, List, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# 폭행 배치(최대 batch_frame_count장의 원본 프레임)가 gRPC 기본 메시지 한도(4MB)를
# 넘기므로 상향 조정한다. 1280x720x3 프레임 12장 ≈ 33MB 기준 여유를 둔 값.
_MAX_MESSAGE_LENGTH = 64 * 1024 * 1024


class InferenceClient:
    """Synchronous gRPC client for mock inference server."""

    def __init__(self, host: str, port: int):
        """Initialize inference client.

        Args:
            host: Server host.
            port: Server port.
        """
        self.host = host
        self.port = port
        # Import deferred until after proto compilation
        self.stub = None

    def _ensure_stub(self):
        """Lazy initialization of gRPC stub."""
        if self.stub is None:
            import grpc

            try:
                from inference_pb2_grpc import InferenceServiceStub
            except ImportError as e:
                logger.error(
                    f"Could not import proto stubs. "
                    f"Run: python -m grpc_tools.protoc ... "
                    f"Error: {e}"
                )
                raise

            channel = grpc.insecure_channel(
                f"{self.host}:{self.port}",
                options=[
                    ("grpc.max_send_message_length", _MAX_MESSAGE_LENGTH),
                    ("grpc.max_receive_message_length", _MAX_MESSAGE_LENGTH),
                ],
            )
            self.stub = InferenceServiceStub(channel)

    def detect(
        self, frame: np.ndarray, frame_id: int
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Call detection inference.

        Args:
            frame: Input frame (HxWx3 uint8).
            frame_id: Frame identifier.

        Returns:
            Tuple of (list of detection dicts, latency_ms).
        """
        self._ensure_stub()

        import inference_pb2

        # Serialize frame to bytes
        frame_bytes = frame.tobytes()

        request = inference_pb2.DetectRequest(
            frame_data=frame_bytes,
            width=frame.shape[1],
            height=frame.shape[0],
            frame_id=frame_id,
        )

        try:
            response = self.stub.Detect(request, timeout=10.0)
            detections = [
                {
                    "x1": bbox.x1,
                    "y1": bbox.y1,
                    "x2": bbox.x2,
                    "y2": bbox.y2,
                    "confidence": bbox.confidence,
                    "class_id": bbox.class_id,
                }
                for bbox in response.detections
            ]
            return detections, response.latency_ms
        except Exception as e:
            logger.error(f"Detection RPC failed: {e}")
            return [], 0.0

    def track(
        self, detections: List[Dict[str, Any]], frame: np.ndarray, frame_id: int
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Call tracking inference.

        Args:
            detections: List of detection dictionaries.
            frame: Input frame (HxWx3 uint8).
            frame_id: Frame identifier.

        Returns:
            Tuple of (list of track dicts, latency_ms).
        """
        self._ensure_stub()

        import inference_pb2

        # Convert detections to BBox protos
        bboxes = [
            inference_pb2.BBox(
                x1=det["x1"],
                y1=det["y1"],
                x2=det["x2"],
                y2=det["y2"],
                confidence=det["confidence"],
                class_id=det["class_id"],
            )
            for det in detections
        ]

        frame_bytes = frame.tobytes()

        request = inference_pb2.TrackRequest(
            detections=bboxes,
            frame_data=frame_bytes,
            width=frame.shape[1],
            height=frame.shape[0],
            frame_id=frame_id,
        )

        try:
            response = self.stub.Track(request, timeout=10.0)
            tracks = [
                {
                    "track_id": t.track_id,
                    "x": t.x,
                    "y": t.y,
                    "w": t.w,
                    "h": t.h,
                    "confidence": t.confidence,
                }
                for t in response.tracks
            ]
            return tracks, response.latency_ms
        except Exception as e:
            logger.error(f"Tracking RPC failed: {e}")
            return [], 0.0

    def pose(
        self, tracks: List[Dict[str, Any]], frame: np.ndarray, frame_id: int
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Call pose estimation inference.

        Args:
            tracks: List of track dictionaries.
            frame: Input frame (HxWx3 uint8).
            frame_id: Frame identifier.

        Returns:
            Tuple of (list of pose dicts, latency_ms).
        """
        self._ensure_stub()

        import inference_pb2

        # Convert tracks to TrackObject protos
        track_objs = [
            inference_pb2.TrackObject(
                track_id=t["track_id"],
                x=t["x"],
                y=t["y"],
                w=t["w"],
                h=t["h"],
                confidence=t["confidence"],
            )
            for t in tracks
        ]

        frame_bytes = frame.tobytes()

        request = inference_pb2.PoseRequest(
            frame_data=frame_bytes,
            tracks=track_objs,
            width=frame.shape[1],
            height=frame.shape[0],
            frame_id=frame_id,
        )

        try:
            response = self.stub.Pose(request, timeout=10.0)
            poses = [
                {
                    "track_id": p.track_id,
                    "keypoints": [
                        {"x": kp.x, "y": kp.y, "confidence": kp.confidence}
                        for kp in p.keypoints
                    ],
                }
                for p in response.poses
            ]
            return poses, response.latency_ms
        except Exception as e:
            logger.error(f"Pose RPC failed: {e}")
            return [], 0.0

    def batch_violence(
        self, frames: List[np.ndarray], width: int, height: int
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Call one violence-ensemble checkpoint pass over an accumulated batch.

        폭행 앙상블 재현을 위해 호출자가 이 메서드를 violence_num_calls회 직렬 호출한다.

        Args:
            frames: Accumulated frames for this batch.
            width: Frame width.
            height: Frame height.

        Returns:
            Tuple of (list of violence result dicts, latency_ms).
        """
        self._ensure_stub()

        import inference_pb2

        request = inference_pb2.BatchViolenceRequest(
            frame_data=[f.tobytes() for f in frames],
            width=width,
            height=height,
            batch_size=len(frames),
        )

        try:
            response = self.stub.BatchViolence(request, timeout=10.0)
            results = [
                {"detected": r.detected, "confidence": r.confidence}
                for r in response.results
            ]
            return results, response.latency_ms
        except Exception as e:
            logger.error(f"BatchViolence RPC failed: {e}")
            return [], 0.0

    def falldown(
        self, frame: np.ndarray, frame_id: int, state_blob: bytes
    ) -> Tuple[Dict[str, Any], bytes, float]:
        """Call falldown inference for a single frame with state chaining.

        쓰러짐 상태 체이닝 재현을 위해 호출자가 응답의 state_blob을 다음 호출의
        state_blob 인자로 전달해야 한다.

        Args:
            frame: Input frame (HxWx3 uint8).
            frame_id: Frame identifier.
            state_blob: Accumulated state from the previous call (empty on first call).

        Returns:
            Tuple of (result dict, updated state_blob, latency_ms).
        """
        self._ensure_stub()

        import inference_pb2

        request = inference_pb2.FalldownRequest(
            frame_data=frame.tobytes(),
            width=frame.shape[1],
            height=frame.shape[0],
            frame_id=frame_id,
            state_blob=state_blob,
        )

        try:
            response = self.stub.Falldown(request, timeout=10.0)
            result = {"detected": response.detected, "confidence": response.confidence}
            return result, response.state_blob, response.latency_ms
        except Exception as e:
            logger.error(f"Falldown RPC failed: {e}")
            return {"detected": False, "confidence": 0.0}, state_blob, 0.0


class FrameRecorder:
    """Accumulates frames (naive np.append implementation for nested experiment)."""

    def __init__(self, max_frames: int):
        """Initialize recorder.

        Args:
            max_frames: Maximum frames to accumulate.
        """
        self.max_frames = max_frames
        # Use np.object_ dtype to store heterogeneous data (mimics original implementation)
        self.frames = np.array([], dtype=np.object_).reshape(0, 3)
        self.lock = None  # Set externally if using threading

    def append(self, frame: np.ndarray, timestamp: float, track_result: Any):
        """Append frame to recorder (O(N) np.append operation).

        Args:
            frame: Frame array.
            timestamp: Timestamp of frame.
            track_result: Tracking result.
        """
        entry = np.array([[timestamp, frame.copy(), track_result]], dtype=np.object_)
        self.frames = np.append(self.frames, entry, axis=0)

        # Keep only last max_frames
        if len(self.frames) > self.max_frames:
            self.frames = self.frames[-self.max_frames :]

    def get_all(self) -> np.ndarray:
        """Get all accumulated frames (for serialization test).

        Returns:
            numpy array of (timestamp, frame, track_result) tuples.
        """
        return self.frames.copy()

    def clear(self):
        """Clear recorder."""
        self.frames = np.array([], dtype=np.object_).reshape(0, 3)


class FrameRecorderDeque:
    """Accumulates frames using deque (optimized for nested experiment fix)."""

    def __init__(self, max_frames: int):
        """Initialize recorder with deque.

        Args:
            max_frames: Maximum frames to accumulate.
        """
        from collections import deque

        self.max_frames = max_frames
        self.frames = deque(maxlen=max_frames)

    def append(self, frame: np.ndarray, timestamp: float, track_result: Any):
        """Append frame to recorder (O(1) deque operation).

        Args:
            frame: Frame array.
            timestamp: Timestamp of frame.
            track_result: Tracking result.
        """
        self.frames.append((timestamp, frame.copy(), track_result))

    def get_all(self) -> List[Tuple[float, np.ndarray, Any]]:
        """Get all accumulated frames.

        Returns:
            List of (timestamp, frame, track_result) tuples.
        """
        return list(self.frames)

    def clear(self):
        """Clear recorder."""
        self.frames.clear()

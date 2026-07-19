"""실험 A(nested) 파이프라인 Actor.

원 시스템 구조를 재현한다 (plan.md §3.2, research.md 문제 1~8):
- StreamActor: 내부 threading.Thread로 프레임을 생성해 무조건 ray.put() 하고 deque에 보관한다
  (문제 4 — 버려지는 프레임도 Plasma Store 쓰기 비용을 발생시킨다). 스트리밍 루프를 별도
  스레드로 돌리는 이유는 Actor의 메서드 디스패치 스레드를 get_frame() 응답용으로 비워두기
  위해서다 (원본과 동일 구조).
- AnalysisActor: sync Actor의 단일 메서드(start_analysis)가 while True로 무한 루프를 돈다
  (문제 8). 매 프레임 ray.get(stream.get_frame.remote())로 Pull한다 (문제 3, 2회 왕복).
  recorder는 np.object_ 배열에 np.append한다 (문제 6, 7). 이 Actor가 실행 중인 동안 다른
  메서드 호출(get_recorder)은 Ray의 단일 스레드 순차 처리 모델 때문에 영원히 실행 기회를
  얻지 못한다 — 이것이 그대로 재현 목표이므로 별도의 graceful stop을 만들지 않는다
  (ray.shutdown() 또는 컨테이너 종료로 정리한다).
"""

import logging
import os
import threading
import time
from collections import deque
from typing import Optional

import ray

from benchmark.common.frame_generator import FramePool, FrameSequence
from benchmark.common.stages import FrameRecorder
from benchmark.nested.inference_backend import (
    MockGrpcInferenceBackend,
    TritonInferenceBackend,
)

logger = logging.getLogger(__name__)
# Ray 워커 프로세스는 이 모듈을 프로세스마다 새로 import하므로, INFO 로그(예:
# utility_task의 hang 알림)가 기본 WARNING 임계값에 묻히지 않도록 여기서 설정한다.
logging.basicConfig(level=logging.INFO)

# 원 시스템의 FRAME_QUEUE_MAX_SIZE=15와 동일 (research.md 문제 4)
FRAME_QUEUE_MAX_SIZE = 15


def create_inference_backend(
    config, backend_kind: str, triton_url: Optional[str] = None
):
    """backend_kind에 따라 InferenceBackend 인스턴스를 새로 만든다.

    Ray 원격 프로세스 경계를 넘나드는 gRPC 채널 객체의 피클링 문제를 피하기 위해,
    Actor/Task 각자가 자기 프로세스 안에서 백엔드를 새로 생성한다.
    """
    if backend_kind == "triton":
        return TritonInferenceBackend(config, url=triton_url or "localhost:8001")
    if backend_kind == "mock_grpc":
        return MockGrpcInferenceBackend(config)
    raise ValueError(f"Unknown backend_kind: {backend_kind}")


@ray.remote
class StreamActor:
    """카메라 1대분 프레임 생성. 내부 스레드가 무조건 ray.put() 한다 (문제 4)."""

    def __init__(self, camera_id: int, workload_config):
        self.camera_id = camera_id
        self.workload_config = workload_config
        self.frame_pool = FramePool(
            width=workload_config.frame_width,
            height=workload_config.frame_height,
            channels=workload_config.frame_channels,
            seed=camera_id,
        )
        self.sequence = FrameSequence(self.frame_pool, fps=workload_config.fps)
        self.buffer = deque(maxlen=FRAME_QUEUE_MAX_SIZE)
        self.lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start_streaming(self):
        """스트리밍 루프를 백그라운드 스레드로 기동하고 즉시 반환한다.

        메서드 자체가 즉시 반환해야 Actor의 디스패치 스레드가 get_frame() 호출을
        계속 처리할 수 있다 (원본 구조와 동일).
        """
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._streaming_loop, daemon=True)
        self._thread.start()

    def _streaming_loop(self):
        interval = 1.0 / self.workload_config.fps
        while self._running:
            frame, frame_id, timestamp = self.sequence.next_frame()
            # 문제 4: 소비 여부와 무관하게 매 프레임 ray.put() — deque에서 밀려나 버려지는
            # 프레임도 이미 Plasma Store 쓰기 비용을 지불한 뒤다.
            frame_ref = ray.put(frame)
            with self.lock:
                self.buffer.append((frame_ref, frame_id, timestamp))
            time.sleep(interval)

    def get_frame(self):
        """가장 오래된 프레임을 꺼낸다. 없으면 None."""
        with self.lock:
            if not self.buffer:
                return None
            return self.buffer.popleft()

    def stop(self):
        self._running = False


@ray.remote
def utility_task(analysis_actor, metrics_actor, camera_id: int):
    """이벤트 발생 시 Recorder 스냅샷을 회수하려는 fire-and-forget 유틸리티 태스크.

    AnalysisActor가 단일 스레드로 무한 루프(start_analysis)를 돌고 있는 동안 이
    ray.get()은 영원히 응답을 받지 못한다 (문제 8). 호출자는 이 태스크를 ray.get()
    없이 fire-and-forget으로 기동하므로, 이벤트가 발생할 때마다 hang된 태스크가
    누적된다 — 원 시스템에서 관찰된 메모리 지속 상승의 재현 대상이다.
    """
    logger.info(f"[camera {camera_id}] utility_task: get_recorder 호출 (hang 예상)")
    recorder_snapshot = ray.get(analysis_actor.get_recorder.remote())
    if metrics_actor is not None:
        metrics_actor.record_event.remote()
    return len(recorder_snapshot)


@ray.remote
def violence_task(
    config, backend_kind: str, triton_url, frames, width, height, metrics_actor
):
    """폭행 = 2모델 앙상블 재현. violence_num_calls회 동기 gRPC 직렬 호출."""
    backend = create_inference_backend(config, backend_kind, triton_url)
    _, latency_ms = backend.violence(frames, width, height)
    if metrics_actor is not None:
        metrics_actor.record_frame_latency.remote("violence", latency_ms)
    return latency_ms


@ray.remote
def falldown_task(config, backend_kind: str, triton_url, frames, metrics_actor):
    """쓰러짐 = 프레임별 상태 blob 체이닝 재현. 최대 batch_frame_count회 직렬 호출."""
    backend = create_inference_backend(config, backend_kind, triton_url)
    _, latency_ms = backend.falldown(frames)
    if metrics_actor is not None:
        metrics_actor.record_frame_latency.remote("falldown", latency_ms)
    return latency_ms


@ray.remote
class AnalysisActor:
    """분석 오케스트레이션. start_analysis()가 while True로 무한 루프를 돈다 (문제 8)."""

    def __init__(
        self,
        camera_id: int,
        config,
        backend_kind: str,
        triton_url: Optional[str],
        metrics_actor=None,
    ):
        self.camera_id = camera_id
        self.config = config
        self.workload = config.workload
        self.backend_kind = backend_kind
        self.triton_url = triton_url
        self.metrics_actor = metrics_actor
        self.inference = create_inference_backend(config, backend_kind, triton_url)
        # np.object_ + np.append — 문제 6(O(N) 재할당), 7(SerDes 대용량 pickle)의 재현 대상
        self.recorder = FrameRecorder(max_frames=self.workload.recorder_max_frames)
        self.batch_buffer = []
        self.last_event_time = time.perf_counter()
        self.frame_count = 0
        self._running = True

    def start_analysis(self, stream_actor):
        """무한 루프. 이 메서드가 반환하지 않는 한 다른 메서드 호출은 큐에서 대기한다."""
        self_handle = ray.get_runtime_context().current_actor
        stress = self.workload.workload_mode == "stress"

        while self._running:
            # 문제 3: Pull 패턴 — 프레임 하나당 Raylet 왕복 2회 (get_frame.remote() 제출 + ray.get 회수)
            frame_info = ray.get(stream_actor.get_frame.remote())
            if frame_info is None:
                time.sleep(0.001)
                continue

            frame_ref, frame_id, timestamp = frame_info
            frame = ray.get(frame_ref)

            self._process_frame(frame, frame_id, timestamp, stress)
            self._maybe_report_self_metrics()
            self._maybe_trigger_event(self_handle)

            self.frame_count += 1

    def _process_frame(self, frame, frame_id, timestamp, stress: bool):
        cfg = self.config
        w, h = self.workload.frame_width, self.workload.frame_height

        detections, latency_detect = self.inference.detect(frame, frame_id)
        self._record_latency("detect", latency_detect, frame_id)

        tracks, latency_track = self.inference.track(detections, frame, frame_id)
        self._record_latency("track", latency_track, frame_id)

        if not stress:
            return

        # 문제 6/7의 살아있는 예: 프레임 원본 + 트래커 결과(이형 타입)를 np.object_로 보관
        self.recorder.append(frame, timestamp, tracks)

        poses, latency_pose = self.inference.pose(tracks, frame, frame_id)
        self._record_latency("pose", latency_pose, frame_id)

        # 침입/배회: 규칙 기반 로컬 연산이므로 gRPC가 아닌 Actor 내 busy-wait으로 모의
        from benchmark.common.mock_latency import busy_wait_ms

        busy_wait_ms(cfg.local_stages.latency_intrusion)
        busy_wait_ms(cfg.local_stages.latency_loitering)

        self.batch_buffer.append(frame)
        if len(self.batch_buffer) >= self.workload.batch_frame_count:
            frames_snapshot = list(self.batch_buffer)
            self.batch_buffer.clear()
            # fire-and-forget: 결과를 기다리지 않고 별도 태스크로 넘긴다
            violence_task.remote(
                cfg,
                self.backend_kind,
                self.triton_url,
                frames_snapshot,
                w,
                h,
                self.metrics_actor,
            )
            falldown_task.remote(
                cfg,
                self.backend_kind,
                self.triton_url,
                frames_snapshot,
                self.metrics_actor,
            )

    def _maybe_report_self_metrics(self):
        # 초당 1회 수준으로 자기 프로세스의 CPU/메모리를 보고한다 (분석 Actor 단독 CPU
        # 낙하 재현 여부를 확인하려면 이 계열을 별도로 분리 계측해야 하므로).
        if self.frame_count % self.workload.fps != 0:
            return
        if self.metrics_actor is not None:
            self.metrics_actor.record_process_metrics.remote(
                f"analysis_actor_{self.camera_id}", os.getpid()
            )
        if self.metrics_actor is not None:
            self.metrics_actor.record_frame_processed.remote()

    def _maybe_trigger_event(self, self_handle):
        now = time.perf_counter()
        if now - self.last_event_time < self.workload.event_trigger_interval:
            return
        self.last_event_time = now
        # fire-and-forget: 응답을 기다리지 않는다. 이 Actor가 while True로 점유된 동안
        # get_recorder()는 실행 기회를 얻지 못해 hang된다 (문제 8).
        utility_task.remote(self_handle, self.metrics_actor, self.camera_id)

    def _record_latency(self, stage: str, latency_ms: float, frame_id: int = -1):
        if self.metrics_actor is not None:
            self.metrics_actor.record_frame_latency.remote(stage, latency_ms, frame_id)

    def get_recorder(self):
        """Recorder 전체를 반환한다 (np.object_ 배열 전체 SerDes — 문제 7).

        start_analysis()가 점유하는 동안은 이 메서드가 호출될 기회조차 없다.
        """
        return self.recorder.get_all()

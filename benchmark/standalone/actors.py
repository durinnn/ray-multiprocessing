"""실험 B(standalone) 파이프라인 Actor.

nested(실험 A)와 동일한 파이프라인 구조를 독립 Ray 프로세스에서 실행하되,
plan.md §3.3의 개선 플래그를 개별로 켜고 끌 수 있게 분기한다. 플래그 전부 off =
B0 = nested와 동일 로직(위치만 standalone), 전부 on = B-all.

nested와 공유하는 부분(추론 백엔드, violence/falldown/utility 태스크, recorder의
컨테이너 축)은 그대로 재사용하고 nested 코드는 수정하지 않는다. standalone 고유
분기는 다음과 같다:
- StreamActor.use_conditional_put: 큐가 가득 차면 ray.put을 건너뛴다 (문제 4).
- AnalysisCore.build_recorder: deque/ObjectRef 저장 여부 (문제 6/7).
- SyncAnalysisActor vs AsyncAnalysisActor: sync 무한루프 vs async+await (문제 8).
- set_cpu_affinity: 프로세스별 코어 고정 (가설 [3], A/B 공용).
"""

import asyncio
import logging
import os
import threading
import time
from collections import deque
from typing import Optional

import ray

from benchmark.common.frame_generator import FramePool, FrameSequence
from benchmark.common.mock_latency import busy_wait_ms

# nested와 공유: 추론 백엔드 생성기 + fire-and-forget 태스크들. 재사용이므로 nested
# 코드는 건드리지 않는다 (plan.md §3.2.1 "Actor 코드는 실험 A/B가 공유").
from benchmark.nested.actors import (
    create_inference_backend,
    falldown_task,
    utility_task,
    violence_task,
)
from benchmark.standalone.recorder import build_recorder

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 원 시스템의 FRAME_QUEUE_MAX_SIZE=15와 동일 (research.md 문제 4)
FRAME_QUEUE_MAX_SIZE = 15


def maybe_set_affinity(enabled: bool, core_index: Optional[int]):
    """set_cpu_affinity가 켜져 있으면 현재 프로세스를 특정 코어에 고정한다.

    Actor/Task의 __init__은 각자의 워커 프로세스 안에서 실행되므로, 여기서
    psutil로 어피니티를 설정하면 그 워커 프로세스 전체가 해당 코어에 묶인다
    (가설 [3]: 어피니티 미설정 시 프로세스 전환마다 캐시 오염 → 캐시 미스).
    """
    if not enabled or core_index is None:
        return
    try:
        import psutil

        ncpu = os.cpu_count() or 1
        core = core_index % ncpu
        psutil.Process().cpu_affinity([core])
        logger.info("cpu affinity 설정: pid=%s core=%s", os.getpid(), core)
    except Exception as exc:  # noqa: BLE001 - 어피니티 실패는 치명적이지 않다
        logger.warning("cpu affinity 설정 실패: %s", exc)


@ray.remote
class StreamActor:
    """카메라 1대분 프레임 생성. 내부 스레드가 프레임을 deque에 채운다.

    use_conditional_put(문제 4 개선):
    - off: 소비 여부와 무관하게 매 프레임 무조건 ray.put (nested와 동일, B0).
    - on:  버퍼가 가득 차 있으면 ray.put 자체를 건너뛴다 (버려질 프레임의 Plasma
      Store 쓰기 비용을 아낀다).
    """

    def __init__(
        self,
        camera_id: int,
        workload_config,
        use_conditional_put: bool = False,
        set_cpu_affinity: bool = False,
        affinity_core: Optional[int] = None,
    ):
        maybe_set_affinity(set_cpu_affinity, affinity_core)
        self.camera_id = camera_id
        self.workload_config = workload_config
        self.use_conditional_put = use_conditional_put
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
        """스트리밍 루프를 백그라운드 스레드로 기동하고 즉시 반환한다."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._streaming_loop, daemon=True)
        self._thread.start()

    def _streaming_loop(self):
        interval = 1.0 / self.workload_config.fps
        while self._running:
            frame, frame_id, timestamp = self.sequence.next_frame()

            if self.use_conditional_put:
                # 문제 4 개선: 버퍼가 가득 차 있으면 어차피 밀려날 프레임이므로
                # ray.put(Plasma Store 쓰기)을 아예 건너뛴다.
                with self.lock:
                    full = len(self.buffer) >= FRAME_QUEUE_MAX_SIZE
                if full:
                    time.sleep(interval)
                    continue

            # 문제 4(off): 소비 여부와 무관하게 매 프레임 ray.put.
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


class AnalysisCore:
    """Sync/Async 두 Actor가 공유하는 프레임 처리 로직·상태.

    Actor 클래스는 프레임 Pull 루프(sync ray.get vs async await)만 다르고, 실제
    파이프라인·recorder·이벤트 트리거·메트릭 로직은 전부 이 코어가 담는다. 이로써
    두 Actor 간 중복을 없앤다.
    """

    def __init__(
        self,
        camera_id: int,
        config,
        backend_kind: str,
        triton_url: Optional[str],
        metrics_actor,
        flags,
    ):
        self.camera_id = camera_id
        self.config = config
        self.workload = config.workload
        self.backend_kind = backend_kind
        self.triton_url = triton_url
        self.metrics_actor = metrics_actor
        self.flags = flags
        self.inference = create_inference_backend(config, backend_kind, triton_url)
        self.recorder = build_recorder(
            self.workload.recorder_max_frames,
            flags.use_deque_recorder,
            flags.use_objectref_recorder,
        )
        self.batch_buffer = []
        self.frame_count = 0
        self.last_event_time = time.perf_counter()
        self.running = True

    @property
    def stress(self) -> bool:
        return self.workload.workload_mode == "stress"

    def process_frame(self, frame, frame_id, timestamp):
        cfg = self.config
        w, h = self.workload.frame_width, self.workload.frame_height

        detections, latency_detect = self.inference.detect(frame, frame_id)
        self.record_latency("detect", latency_detect, frame_id)

        tracks, latency_track = self.inference.track(detections, frame, frame_id)
        self.record_latency("track", latency_track, frame_id)

        if not self.stress:
            return

        # 문제 6/7 분기: objectref on이면 ray.put한 경량 ref만 저장, off면 프레임
        # 원본을 저장(recorder가 내부에서 copy). deque/np.append 축은 recorder가 처리.
        if self.flags.use_objectref_recorder:
            self.recorder.append(ray.put(frame), timestamp, tracks)
        else:
            self.recorder.append(frame, timestamp, tracks)

        poses, latency_pose = self.inference.pose(tracks, frame, frame_id)
        self.record_latency("pose", latency_pose, frame_id)

        # 침입/배회: 규칙 기반 로컬 연산이므로 gRPC가 아닌 Actor 내 busy-wait으로 모의
        busy_wait_ms(cfg.local_stages.latency_intrusion)
        busy_wait_ms(cfg.local_stages.latency_loitering)

        self.batch_buffer.append(frame)
        if len(self.batch_buffer) >= self.workload.batch_frame_count:
            frames_snapshot = list(self.batch_buffer)
            self.batch_buffer.clear()
            # nested와 공유하는 fire-and-forget 태스크. 결과를 기다리지 않는다.
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

    def on_frame_done(self, frame_id, timestamp, self_handle):
        # e2e = 프레임 생성(StreamActor의 perf_counter) → 동기 스테이지 완료.
        # nested와 동일 정의로 기록해 A/B 비교가 가능하게 한다.
        self.record_latency("e2e", (time.perf_counter() - timestamp) * 1000.0, frame_id)
        self._maybe_report_self_metrics()
        self._maybe_trigger_event(self_handle)
        self.frame_count += 1

    def record_latency(self, stage: str, latency_ms: float, frame_id: int = -1):
        if self.metrics_actor is not None:
            self.metrics_actor.record_frame_latency.remote(stage, latency_ms, frame_id)

    def _maybe_report_self_metrics(self):
        # 초당 1회 수준으로 자기 프로세스의 CPU/메모리를 보고 (분석 Actor 단독 계측).
        if self.frame_count % self.workload.fps != 0:
            return
        if self.metrics_actor is not None:
            self.metrics_actor.record_process_metrics.remote(
                f"analysis_actor_{self.camera_id}", os.getpid()
            )
            self.metrics_actor.record_frame_processed.remote()

    def _maybe_trigger_event(self, self_handle):
        now = time.perf_counter()
        if now - self.last_event_time < self.workload.event_trigger_interval:
            return
        self.last_event_time = now
        # nested와 공유하는 utility_task. get_recorder.remote()를 ray.get 한다:
        # sync Actor면 무한루프에 막혀 hang(문제 8), async Actor면 정상 완료.
        utility_task.remote(self_handle, self.metrics_actor, self.camera_id)

    def get_recorder_snapshot(self):
        return self.recorder.get_all()


@ray.remote
class SyncAnalysisActor:
    """sync Actor. start_analysis()가 while True로 무한 루프를 돈다 (문제 8, B0).

    이 Actor가 점유하는 동안 get_recorder() 호출은 큐에서 대기만 하다가 hang된다 —
    nested의 AnalysisActor와 동일한 재현 동작이다.
    """

    def __init__(
        self,
        camera_id: int,
        config,
        backend_kind: str,
        triton_url: Optional[str],
        flags,
        metrics_actor=None,
        affinity_core: Optional[int] = None,
    ):
        maybe_set_affinity(flags.set_cpu_affinity, affinity_core)
        self.core = AnalysisCore(
            camera_id, config, backend_kind, triton_url, metrics_actor, flags
        )

    def start_analysis(self, stream_actor):
        self_handle = ray.get_runtime_context().current_actor
        core = self.core
        while core.running:
            # 문제 3: Pull 패턴 — 프레임 하나당 Raylet 왕복 2회.
            frame_info = ray.get(stream_actor.get_frame.remote())
            if frame_info is None:
                time.sleep(0.001)
                continue
            frame_ref, frame_id, timestamp = frame_info
            frame = ray.get(frame_ref)
            core.process_frame(frame, frame_id, timestamp)
            core.on_frame_done(frame_id, timestamp, self_handle)

    def get_recorder(self):
        return self.core.get_recorder_snapshot()


@ray.remote
class AsyncAnalysisActor:
    """async Actor. start_analysis()가 await로 양보하므로 다른 메서드가 끼어든다.

    문제 8 개선: await 지점에서 이벤트 루프가 get_recorder() 같은 다른 메서드 호출을
    실행할 기회를 얻는다 → utility_task가 더는 hang되지 않는다. research.md §6에서
    "async Actor 병행 필수"라 한 그 경로다.
    """

    def __init__(
        self,
        camera_id: int,
        config,
        backend_kind: str,
        triton_url: Optional[str],
        flags,
        metrics_actor=None,
        affinity_core: Optional[int] = None,
    ):
        maybe_set_affinity(flags.set_cpu_affinity, affinity_core)
        self.core = AnalysisCore(
            camera_id, config, backend_kind, triton_url, metrics_actor, flags
        )

    async def start_analysis(self, stream_actor):
        self_handle = ray.get_runtime_context().current_actor
        core = self.core
        while core.running:
            # await로 프레임을 당겨온다 — 이 지점에서 다른 Actor 메서드가 실행될 수 있다.
            frame_info = await stream_actor.get_frame.remote()
            if frame_info is None:
                await asyncio.sleep(0.001)
                continue
            frame_ref, frame_id, timestamp = frame_info
            frame = await frame_ref
            core.process_frame(frame, frame_id, timestamp)
            core.on_frame_done(frame_id, timestamp, self_handle)

    async def get_recorder(self):
        return self.core.get_recorder_snapshot()


def make_analysis_actor(use_async: bool):
    """플래그에 따라 sync/async Analysis Actor 클래스를 고른다."""
    return AsyncAnalysisActor if use_async else SyncAnalysisActor

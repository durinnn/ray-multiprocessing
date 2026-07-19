"""실험 B recorder 선택 — 두 개의 독립 최적화 축을 플래그로 조합한다.

- 컨테이너 축(문제 6): np.append(O(N) 재할당) vs deque(maxlen)(O(1))
- 저장물 축(문제 7): 프레임 원본 복사(전체 pickle) vs ObjectRef(경량 SerDes)

컨테이너 축은 nested(실험 A)와 공유하는 benchmark.common.stages의
FrameRecorder / FrameRecorderDeque를 그대로 재사용한다. 저장물 축(ObjectRef)만
이 모듈의 ObjectRefRecorder로 추가한다.
"""

from collections import deque
from typing import Any

import numpy as np

from benchmark.common.stages import FrameRecorder, FrameRecorderDeque


class ObjectRefRecorder:
    """프레임 원본 대신 경량 ObjectRef를 저장한다 (문제 7 개선).

    저장물만 ObjectRef로 바꿀 뿐 컨테이너 축(np.append vs deque, 문제 6)은
    use_deque로 동일하게 반영한다. append()에 넘어오는 payload는 호출자가 이미
    ray.put(frame)으로 만든 ObjectRef이므로 여기서는 복사하지 않고 그대로 담는다.

    주의(research.md §6): deque가 유일한 참조자인 동안 Object Store 압박이 크면
    Ray가 참조 미검출로 판단해 eviction 할 수 있다. explicit_object_store 플래그로
    object_store_memory를 넉넉히 잡는 것과 병행하는 것을 전제로 한다.
    """

    def __init__(self, max_frames: int, use_deque: bool):
        self.max_frames = max_frames
        self.use_deque = use_deque
        if use_deque:
            self._deque = deque(maxlen=max_frames)
        else:
            self._arr = np.array([], dtype=np.object_).reshape(0, 3)

    def append(self, frame_ref: Any, timestamp: float, track_result: Any):
        if self.use_deque:
            self._deque.append((timestamp, frame_ref, track_result))
        else:
            entry = np.array([[timestamp, frame_ref, track_result]], dtype=np.object_)
            self._arr = np.append(self._arr, entry, axis=0)
            if len(self._arr) > self.max_frames:
                self._arr = self._arr[-self.max_frames :]

    def get_all(self):
        if self.use_deque:
            return list(self._deque)
        return self._arr.copy()

    def clear(self):
        if self.use_deque:
            self._deque.clear()
        else:
            self._arr = np.array([], dtype=np.object_).reshape(0, 3)


def build_recorder(max_frames: int, use_deque: bool, use_objectref: bool):
    """플래그 조합에 맞는 recorder를 만든다.

    use_objectref off 경로는 nested와 공유하는 공통 recorder를 그대로 쓴다:
    - (deque off, objectref off) = FrameRecorder      → B0, nested와 동일
    - (deque on,  objectref off) = FrameRecorderDeque
    use_objectref on 경로만 ObjectRefRecorder를 쓴다.
    """
    if use_objectref:
        return ObjectRefRecorder(max_frames, use_deque)
    if use_deque:
        return FrameRecorderDeque(max_frames)
    return FrameRecorder(max_frames)


__all__ = ["ObjectRefRecorder", "build_recorder"]

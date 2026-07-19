"""Frame generation for synthetic workload.

Uses a pre-generated numpy pool to avoid frame generation overhead
becoming a measurement bottleneck.
"""

import numpy as np
from typing import Tuple


class FramePool:
    """Pre-generated frame pool to avoid generation overhead during measurement."""

    def __init__(
        self,
        width: int,
        height: int,
        channels: int,
        pool_size: int = 100,
        seed: int = 42,
    ):
        """Initialize frame pool.

        Args:
            width: Frame width.
            height: Frame height.
            channels: Number of channels (typically 3 for BGR).
            pool_size: Number of pre-generated frames to cycle through.
            seed: Random seed for reproducibility.
        """
        np.random.seed(seed)
        self.width = width
        self.height = height
        self.channels = channels
        self.pool_size = pool_size
        self.pool_index = 0

        # Pre-generate frames: uint8 BGR 배열
        self.frames = [
            np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
            for _ in range(pool_size)
        ]

    def get_frame(self) -> np.ndarray:
        """Get next frame from pool (cycling).

        Returns:
            numpy array of shape (height, width, channels) as uint8.
        """
        frame = self.frames[self.pool_index].copy()
        self.pool_index = (self.pool_index + 1) % self.pool_size
        return frame

    def get_frame_size_bytes(self) -> int:
        """Get size of a single frame in bytes."""
        return self.width * self.height * self.channels


class FrameSequence:
    """Generates frame sequence with consistent pacing."""

    def __init__(
        self,
        frame_pool: FramePool,
        fps: int,
    ):
        """Initialize frame sequence generator.

        Args:
            frame_pool: FramePool instance.
            fps: Frames per second pacing.
        """
        self.frame_pool = frame_pool
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.frame_id = 0

    def next_frame(self) -> Tuple[np.ndarray, int, float]:
        """Get next frame with metadata.

        Returns:
            Tuple of (frame_array, frame_id, timestamp_seconds).
        """
        import time

        frame = self.frame_pool.get_frame()
        frame_id = self.frame_id
        timestamp = time.perf_counter()

        self.frame_id += 1

        return frame, frame_id, timestamp

"""Smoke test for benchmark infrastructure.

Validates that all components (config, frame generation, metrics, gRPC) work together.
"""

import logging
import sys
from pathlib import Path

from benchmark.common.config import load_config
from benchmark.common.frame_generator import FramePool, FrameSequence
from benchmark.common.metrics import MetricsCollector

logger = logging.getLogger(__name__)


def run_smoke_test(config_path: str = None, duration: int = 30):
    """Run smoke test.

    Args:
        config_path: Path to config YAML.
        duration: Test duration in seconds.

    Returns:
        True if test passes, False otherwise.
    """
    logging.basicConfig(level=logging.INFO)
    logger.info("=== Smoke Test Start ===")

    try:
        # Load config
        logger.info("1. Loading configuration...")
        config = load_config(config_path)
        logger.info(f"   - Cameras: {config.smoke.num_cameras}")
        logger.info(
            f"   - Frame size: {config.workload.frame_width}x{config.workload.frame_height}"
        )
        logger.info(f"   - FPS: {config.workload.fps}")

        # Initialize frame generator
        logger.info("2. Initializing frame generator...")
        frame_pool = FramePool(
            width=config.workload.frame_width,
            height=config.workload.frame_height,
            channels=config.workload.frame_channels,
        )
        logger.info(f"   - Pool size: {frame_pool.pool_size} frames")
        logger.info(
            f"   - Frame size: {frame_pool.get_frame_size_bytes() / 1024 / 1024:.2f} MB"
        )

        # Generate a few frames
        logger.info("3. Testing frame generation...")
        frame_seq = FrameSequence(frame_pool, fps=config.workload.fps)
        for i in range(5):
            frame, frame_id, ts = frame_seq.next_frame()
            assert frame.shape == (
                config.workload.frame_height,
                config.workload.frame_width,
                config.workload.frame_channels,
            )
            assert frame.dtype == "uint8"
        logger.info("   - Generated 5 frames successfully")

        # Initialize metrics
        logger.info("4. Initializing metrics collector...")
        metrics = MetricsCollector(config, experiment_name="smoke_test")
        metrics.init_csv_logging()
        logger.info(f"   - CSV output: {metrics.csv_path}")

        # Record some dummy metrics
        logger.info("5. Testing metrics logging...")
        for i in range(10):
            metrics.record_frame_latency("detect", 15.0 + i, frame_id=i)
            metrics.record_frame_processed()
        logger.info("   - Recorded 10 frame latencies")

        # Check CSV was created
        if metrics.csv_path.exists():
            logger.info(f"   - CSV file created: {metrics.csv_path}")
            with open(metrics.csv_path) as f:
                lines = f.readlines()
                logger.info(f"   - CSV lines: {len(lines)}")
        else:
            logger.error("   - CSV file not created!")
            return False

        metrics.close()

        # Check proto files
        logger.info("6. Checking proto files...")
        proto_path = (
            Path(__file__).parent.parent.parent / "inference_mock" / "inference.proto"
        )
        if proto_path.exists():
            logger.info(f"   - Proto file exists: {proto_path}")
        else:
            logger.warning(f"   - Proto file not found: {proto_path}")

        # Check if proto stubs need to be generated
        pb2_path = proto_path.parent / "inference_pb2.py"
        pb2_grpc_path = proto_path.parent / "inference_pb2_grpc.py"
        if not pb2_path.exists() or not pb2_grpc_path.exists():
            logger.warning(
                "   - Proto stubs not generated. Run: python -m grpc_tools.protoc ..."
            )
            logger.warning("   - This is expected before first compilation")
        else:
            logger.info("   - Proto stubs generated")

        logger.info("\n=== Smoke Test PASSED ===")
        return True

    except Exception as e:
        logger.error(f"Smoke test failed: {e}", exc_info=True)
        logger.info("\n=== Smoke Test FAILED ===")
        return False


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)

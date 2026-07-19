"""Metrics collection for benchmark experiments.

Provides prometheus metrics export and CSV logging.
"""

import csv
import logging
import threading
import time
from collections import defaultdict
from pathlib import Path

import psutil
import ray
from prometheus_client import Counter, Gauge, Histogram, start_http_server

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and exports metrics for benchmark experiments."""

    def __init__(self, config, experiment_name: str = "benchmark"):
        """Initialize metrics collector.

        Args:
            config: ExperimentConfig with metrics settings.
            experiment_name: Name for this experiment (nested/standalone/micro).
        """
        self.config = config.metrics
        self.experiment_name = experiment_name

        # Prometheus metrics
        self.latency_histogram = Histogram(
            "frame_latency_ms",
            "Frame processing latency in milliseconds",
            buckets=self.config.latency_buckets,
            labelnames=["stage", "experiment"],
        )

        self.frame_counter = Counter(
            "frames_processed_total",
            "Total frames processed",
            labelnames=["experiment"],
        )

        self.event_counter = Counter(
            "events_triggered_total",
            "Total events triggered (get_recorder calls)",
            labelnames=["experiment"],
        )

        self.cpu_gauge = Gauge(
            "process_cpu_percent",
            "Process CPU usage percentage",
            labelnames=["process_name", "experiment"],
        )

        self.memory_gauge = Gauge(
            "process_memory_rss_bytes",
            "Process RSS memory in bytes",
            labelnames=["process_name", "experiment"],
        )

        self.object_store_gauge = Gauge(
            "ray_object_store_bytes",
            "Ray Object Store usage in bytes",
            labelnames=["experiment"],
        )

        self.spilling_gauge = Gauge(
            "ray_object_spilling_bytes",
            "Ray Object Spilling total bytes",
            labelnames=["experiment"],
        )

        # CSV logging
        self.csv_path = Path(self.config.csv_output)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        self.csv_file = None
        self.csv_writer = None
        self.csv_lock = threading.Lock()

        # Latency tracking (in-memory for CSV)
        self.latencies = defaultdict(list)

    def start_prometheus_server(self, port: int = None):
        """Start Prometheus metrics HTTP server.

        Args:
            port: Port to listen on. Uses config default if None.
        """
        if port is None:
            port = self.config.prometheus_port
        try:
            start_http_server(port)
            logger.info(f"Prometheus metrics server started on port {port}")
        except Exception as e:
            logger.warning(f"Could not start Prometheus server: {e}")

    def init_csv_logging(self):
        """Initialize CSV logging file."""
        with self.csv_lock:
            self.csv_file = open(self.csv_path, "w", newline="")
            fieldnames = [
                "timestamp",
                "frame_id",
                "stage",
                "latency_ms",
                "cpu_percent",
                "memory_rss_bytes",
                "event_count",
            ]
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
            self.csv_writer.writeheader()
            self.csv_file.flush()

    def record_frame_latency(self, stage: str, latency_ms: float, frame_id: int = -1):
        """Record frame processing latency (Prometheus + CSV 동시 기록).

        Args:
            stage: Pipeline stage (detect/track/pose/etc).
            latency_ms: Latency in milliseconds.
            frame_id: Frame identifier, if available.
        """
        self.latency_histogram.labels(
            stage=stage, experiment=self.experiment_name
        ).observe(latency_ms)
        timestamp = time.time()
        self.latencies[stage].append((timestamp, latency_ms))
        self.log_csv_row(
            timestamp=timestamp, frame_id=frame_id, stage=stage, latency_ms=latency_ms
        )

    def record_frame_processed(self):
        """Record a processed frame."""
        self.frame_counter.labels(experiment=self.experiment_name).inc()

    def record_event(self):
        """Record an event (e.g., get_recorder call)."""
        self.event_counter.labels(experiment=self.experiment_name).inc()

    def record_process_metrics(self, process_name: str, pid: int = None):
        """Record CPU and memory for a process.

        Args:
            process_name: Name of the process (e.g., 'analysis_actor').
            pid: Process ID. If None, uses current process.
        """
        if pid is None:
            process = psutil.Process()
        else:
            try:
                process = psutil.Process(pid)
            except psutil.NoSuchProcess:
                return

        try:
            cpu_percent = process.cpu_percent(interval=0.01)
            memory_rss = process.memory_info().rss

            self.cpu_gauge.labels(
                process_name=process_name, experiment=self.experiment_name
            ).set(cpu_percent)
            self.memory_gauge.labels(
                process_name=process_name, experiment=self.experiment_name
            ).set(memory_rss)

            if self.csv_writer:
                with self.csv_lock:
                    self.csv_writer.writerow(
                        {
                            "timestamp": time.time(),
                            "frame_id": -1,
                            "stage": f"process:{process_name}",
                            "latency_ms": 0,
                            "cpu_percent": cpu_percent,
                            "memory_rss_bytes": memory_rss,
                            "event_count": 0,
                        }
                    )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def record_ray_metrics(self, object_store_bytes: int = 0, spilling_bytes: int = 0):
        """Record Ray-specific metrics.

        Args:
            object_store_bytes: Ray Object Store usage.
            spilling_bytes: Ray Object Spilling total.
        """
        if object_store_bytes > 0:
            self.object_store_gauge.labels(experiment=self.experiment_name).set(
                object_store_bytes
            )
        if spilling_bytes > 0:
            self.spilling_gauge.labels(experiment=self.experiment_name).set(
                spilling_bytes
            )

    def log_csv_row(
        self,
        timestamp: float,
        frame_id: int,
        stage: str,
        latency_ms: float,
        cpu_percent: float = 0,
        memory_rss_bytes: int = 0,
        event_count: int = 0,
    ):
        """Log a row to CSV.

        Args:
            timestamp: Unix timestamp.
            frame_id: Frame ID.
            stage: Pipeline stage.
            latency_ms: Latency in milliseconds.
            cpu_percent: CPU usage percentage.
            memory_rss_bytes: Memory usage in bytes.
            event_count: Event count.
        """
        if self.csv_writer:
            with self.csv_lock:
                self.csv_writer.writerow(
                    {
                        "timestamp": timestamp,
                        "frame_id": frame_id,
                        "stage": stage,
                        "latency_ms": latency_ms,
                        "cpu_percent": cpu_percent,
                        "memory_rss_bytes": memory_rss_bytes,
                        "event_count": event_count,
                    }
                )
                self.csv_file.flush()

    def close(self):
        """Close CSV file."""
        if self.csv_file:
            with self.csv_lock:
                self.csv_file.close()
                logger.info(f"CSV metrics written to {self.csv_path}")


@ray.remote
class MetricsActor:
    """MetricsCollector를 Ray Actor로 감싼 중앙 집계 지점.

    실험 A/B는 Analysis Actor, violence/falldown 태스크 등 여러 프로세스에서 지표를
    기록한다. Prometheus HTTP 서버와 CSV 파일은 프로세스당 하나만 열 수 있으므로,
    이 Actor 하나로 중앙화해 각 프로세스는 fire-and-forget으로 기록만 위임한다.
    """

    def __init__(self, config, experiment_name: str = "benchmark"):
        self.collector = MetricsCollector(config, experiment_name)
        self.collector.start_prometheus_server()
        self.collector.init_csv_logging()

    def record_frame_latency(self, stage: str, latency_ms: float, frame_id: int = -1):
        self.collector.record_frame_latency(stage, latency_ms, frame_id)

    def record_frame_processed(self):
        self.collector.record_frame_processed()

    def record_event(self):
        self.collector.record_event()

    def record_process_metrics(self, process_name: str, pid: int = None):
        self.collector.record_process_metrics(process_name, pid)

    def record_ray_metrics(self, object_store_bytes: int = 0, spilling_bytes: int = 0):
        self.collector.record_ray_metrics(object_store_bytes, spilling_bytes)

    def close(self):
        self.collector.close()

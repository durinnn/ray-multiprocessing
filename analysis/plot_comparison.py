"""Generate comparison graphs from benchmark measurement CSVs.

Reads the three measured runs under ``docs/data/`` and renders five comparison
PNGs into ``docs/img/``. Every number is computed from the CSVs -- nothing is
hard-coded.

Runs:
  - A       : nested stress, 256MB object store, oomkill off. Instrumentation
              collapses ~86s in (the collapse itself is the result).
  - B0      : standalone, all improvement flags OFF.
  - B-all   : standalone, all improvement flags ON.

CSV schema: timestamp,frame_id,stage,latency_ms,cpu_percent,memory_rss_bytes,
event_count
  - stage in {detect,track,pose,violence,falldown,e2e} -> latency_ms is valid.
  - stage == "process:analysis_actor_N" -> cpu_percent / memory_rss_bytes valid.

Reproduce with either::

    python -m analysis.plot_comparison
    .venv/bin/python analysis/plot_comparison.py
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (backend set above)
import numpy as np  # noqa: E402

# --------------------------------------------------------------------------- #
# Paths & run definitions
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "docs" / "data"
IMG_DIR = REPO_ROOT / "docs" / "img"

# Categorical palette (dataviz skill, validated light-surface hues).
# Fixed order per series -- never assigned by rank.
COLOR_A = "#e34948"  # red    -- nested (collapse case)
COLOR_B0 = "#eda100"  # amber -- standalone, improvements off
COLOR_BALL = "#2a78d6"  # blue -- standalone, improvements on

GRID_KW = dict(color="#d6d5d1", linewidth=0.8, alpha=0.9)
BUCKET_SECONDS = 30
ACTOR_STAGE_PREFIX = "process:analysis_actor"


@dataclass
class Run:
    """A single measured run and the rows loaded from its CSV."""

    key: str
    label: str
    color: str
    filename: str
    e2e_t: np.ndarray = field(default_factory=lambda: np.empty(0))
    e2e_latency: np.ndarray = field(default_factory=lambda: np.empty(0))
    actor_t: np.ndarray = field(default_factory=lambda: np.empty(0))
    actor_cpu: np.ndarray = field(default_factory=lambda: np.empty(0))
    actor_rss_mb: np.ndarray = field(default_factory=lambda: np.empty(0))


RUNS = [
    Run(
        "A",
        "A: nested (256MB, oomkill off)",
        COLOR_A,
        "nested-stress-256mb-oomkill-off.csv",
    ),
    Run("B0", "B0: standalone (all off)", COLOR_B0, "standalone-b0-stress-600s.csv"),
    Run(
        "Ball",
        "B-all: standalone (all on)",
        COLOR_BALL,
        "standalone-ball-stress-600s.csv",
    ),
]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def load_run(run: Run) -> Run:
    """Populate ``run`` arrays from its CSV. Time is made relative to the run's
    first timestamp so the three runs share a common x origin."""
    path = DATA_DIR / run.filename
    ts, e2e_lat = [], []
    a_ts, a_cpu, a_rss = [], [], []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            stage = row["stage"]
            timestamp = _to_float(row["timestamp"])
            if math.isnan(timestamp):
                continue
            if stage == "e2e":
                latency = _to_float(row["latency_ms"])
                if not math.isnan(latency):
                    ts.append(timestamp)
                    e2e_lat.append(latency)
            elif stage.startswith(ACTOR_STAGE_PREFIX):
                cpu = _to_float(row["cpu_percent"])
                rss = _to_float(row["memory_rss_bytes"])
                a_ts.append(timestamp)
                a_cpu.append(cpu)
                a_rss.append(rss)

    all_ts = ts + a_ts
    origin = min(all_ts) if all_ts else 0.0

    run.e2e_t = np.asarray(ts, dtype=float) - origin
    run.e2e_latency = np.asarray(e2e_lat, dtype=float)
    run.actor_t = np.asarray(a_ts, dtype=float) - origin
    run.actor_cpu = np.asarray(a_cpu, dtype=float)
    run.actor_rss_mb = np.asarray(a_rss, dtype=float) / (1024.0 * 1024.0)
    return run


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _bucketed_percentiles(
    times: np.ndarray, values: np.ndarray, bucket: int, pct: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return (bucket_center_seconds, percentile_per_bucket)."""
    if times.size == 0:
        return np.empty(0), np.empty(0)
    edges = np.arange(0, times.max() + bucket, bucket)
    centers, out = [], []
    for lo in edges:
        mask = (times >= lo) & (times < lo + bucket)
        if not np.any(mask):
            continue
        centers.append(lo + bucket / 2.0)
        out.append(np.percentile(values[mask], pct))
    return np.asarray(centers), np.asarray(out)


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(True, **GRID_KW)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#8a8a86")


def _save(fig: plt.Figure, name: str) -> Path:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    out = IMG_DIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #


def plot_e2e_timeseries(runs: Sequence[Run]) -> Path:
    """(1) e2e latency over time: P50 (solid) and P99 (dashed) per run in 30s
    buckets. A's line stops where its instrumentation collapses."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for run in runs:
        t50, p50 = _bucketed_percentiles(run.e2e_t, run.e2e_latency, BUCKET_SECONDS, 50)
        t99, p99 = _bucketed_percentiles(run.e2e_t, run.e2e_latency, BUCKET_SECONDS, 99)
        ax.plot(
            t50,
            p50,
            color=run.color,
            linewidth=2.0,
            marker="o",
            markersize=4,
            label=f"{run.label} - P50",
        )
        ax.plot(
            t99,
            p99,
            color=run.color,
            linewidth=2.0,
            linestyle="--",
            marker="^",
            markersize=4,
            alpha=0.85,
            label=f"{run.label} - P99",
        )
    ax.set_title(
        "End-to-end latency over time (30s buckets, P50 solid / " "P99 dashed)"
    )
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("e2e latency (ms)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, ncol=1, loc="upper left", framealpha=0.9)
    _style_axes(ax)
    return _save(fig, "01_e2e_latency_timeseries.png")


def plot_e2e_percentile_bars(runs: Sequence[Run]) -> Path:
    """(2) Overall e2e P50 / P99 grouped bars per run."""
    metrics = ["P50", "P99"]
    pct_values = {50: "P50", 99: "P99"}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    n = len(runs)
    width = 0.8 / n
    x = np.arange(len(metrics))
    for i, run in enumerate(runs):
        heights = [
            float(np.percentile(run.e2e_latency, pct)) if run.e2e_latency.size else 0.0
            for pct in pct_values
        ]
        offset = (i - (n - 1) / 2.0) * width
        bars = ax.bar(x + offset, heights, width, color=run.color, label=run.label)
        for rect, h in zip(bars, heights):
            ax.annotate(
                f"{h:.0f}",
                (rect.get_x() + rect.get_width() / 2, h),
                ha="center",
                va="bottom",
                fontsize=8,
                color="#52514e",
                xytext=(0, 2),
                textcoords="offset points",
            )
    ax.set_title("End-to-end latency percentiles by run")
    ax.set_xlabel("Percentile")
    ax.set_ylabel("e2e latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=8, framealpha=0.9)
    _style_axes(ax)
    return _save(fig, "02_e2e_percentile_bars.png")


def plot_frame_count_bars(runs: Sequence[Run]) -> Path:
    """(3) Processed frame count (e2e row count) per run -- throughput."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    labels = [run.label for run in runs]
    counts = [run.e2e_latency.size for run in runs]
    colors = [run.color for run in runs]
    x = np.arange(len(runs))
    bars = ax.bar(x, counts, width=0.6, color=colors)
    for rect, c in zip(bars, counts):
        ax.annotate(
            f"{c}",
            (rect.get_x() + rect.get_width() / 2, c),
            ha="center",
            va="bottom",
            fontsize=9,
            color="#52514e",
            xytext=(0, 2),
            textcoords="offset points",
        )
    ax.set_title("Processed frames per run (e2e completions)")
    ax.set_xlabel("Run")
    ax.set_ylabel("e2e rows (processed frames)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(bottom=0)
    _style_axes(ax)
    return _save(fig, "03_processed_frame_count_bars.png")


def plot_actor_cpu_timeseries(runs: Sequence[Run]) -> Path:
    """(4) AnalysisActor CPU% over time (all actors pooled). Shows A's drop to
    0% vs the standalone runs' sustained distribution."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for run in runs:
        if run.actor_t.size == 0:
            continue
        ax.scatter(
            run.actor_t,
            run.actor_cpu,
            s=22,
            color=run.color,
            alpha=0.65,
            edgecolors="white",
            linewidths=0.4,
            label=run.label,
        )
    ax.set_title("AnalysisActor CPU utilization over time (all actors pooled)")
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("CPU (%)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, framealpha=0.9)
    _style_axes(ax)
    return _save(fig, "04_actor_cpu_timeseries.png")


def plot_actor_rss_timeseries(runs: Sequence[Run]) -> Path:
    """(5) AnalysisActor RSS (MB) over time (all actors pooled)."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for run in runs:
        if run.actor_t.size == 0:
            continue
        order = np.argsort(run.actor_t)
        ax.plot(
            run.actor_t[order],
            run.actor_rss_mb[order],
            color=run.color,
            linewidth=1.6,
            marker="o",
            markersize=3.5,
            alpha=0.8,
            label=run.label,
        )
    ax.set_title("AnalysisActor resident memory over time (all actors pooled)")
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("RSS (MB)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, framealpha=0.9)
    _style_axes(ax)
    return _save(fig, "05_actor_rss_timeseries.png")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    runs = [load_run(run) for run in RUNS]
    outputs = [
        plot_e2e_timeseries(runs),
        plot_e2e_percentile_bars(runs),
        plot_frame_count_bars(runs),
        plot_actor_cpu_timeseries(runs),
        plot_actor_rss_timeseries(runs),
    ]
    for path in outputs:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

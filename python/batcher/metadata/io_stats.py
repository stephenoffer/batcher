"""Observed per-source I/O throughput — measured on read, captured for prediction.

Core measures the wall time and (decoded) byte volume of each source read and records a
smoothed throughput (MB/s) here, keyed by the source's stable `identity()`. A later read of
the same source can then *predict* its read cost — the signal that turns the documented
small-files scan pathology from an unpredictable stall into a sized fan-out. Best-effort
throughout: a write never breaks a read, and a cold store simply yields `None`.

This is metadata capture, not a scheduling decision — it records what the hardware did; the
optimizer/Carbonite consume it. The figure is decoded bytes per wall-second (compression
folds into it consistently per source), smoothed across runs so one noisy read doesn't jerk it.
"""

from __future__ import annotations

from batcher.config import active_config
from batcher.metadata.hub import MetadataHub

__all__ = ["load_source_throughput_mbps", "predicted_read_seconds", "record_source_io"]

_NAMESPACE = "io.throughput_mbps"


def record_source_io(
    hub: MetadataHub | None, identity: str, byte_count: int, elapsed_ms: float
) -> None:
    """Record a source read's observed throughput (MB/s), exp-smoothed across runs.

    Best-effort and non-blocking: a bad measurement (zero bytes/time) or any failure is
    dropped, never raised into the read path.
    """
    if hub is None or not identity or byte_count <= 0 or elapsed_ms <= 0:
        return
    try:
        mbps = (byte_count / (1024 * 1024)) / (elapsed_ms / 1000.0)
        alpha = active_config().optimizer.learning_smoothing_alpha
        prior = hub.get_keyed_param(_NAMESPACE, identity)
        smoothed = mbps if prior is None else alpha * mbps + (1.0 - alpha) * float(prior)
        hub.put_keyed_param(_NAMESPACE, identity, smoothed)
    except Exception:  # pragma: no cover - metadata capture must never break a read
        pass


def load_source_throughput_mbps(hub: MetadataHub | None, identity: str) -> float | None:
    """The learned read throughput (MB/s) for source `identity`, or `None` (cold/unavailable)."""
    if hub is None or not identity:
        return None
    try:
        value = hub.get_keyed_param(_NAMESPACE, identity)
    except Exception:  # pragma: no cover - a learned read must never break planning
        return None
    return float(value) if value is not None else None


def predicted_read_seconds(
    hub: MetadataHub | None, identity: str, byte_count: int
) -> float | None:
    """Predicted wall time to read `byte_count` bytes of source `identity`, from its learned
    throughput — the "predict" half: turn measured MB/s + a known byte size into an expected
    read cost the optimizer/`explain` can reason about *before* running. `None` when the
    source's throughput was never measured (cold) or the byte count is non-positive."""
    mbps = load_source_throughput_mbps(hub, identity)
    if mbps is None or mbps <= 0.0 or byte_count <= 0:
        return None
    return (byte_count / (1024 * 1024)) / mbps

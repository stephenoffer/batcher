"""Distributed streaming heterogeneous execution — overlapped, resource-class stages.

The distributed image of the single-node `ml/pipeline.py::run_pipeline`: a linear
`map_batches` chain that crosses a resource-class boundary (CPU preprocess → GPU /
load-once inference) is split into per-stage actor pools that stream partitions
stage→stage over Arrow Flight, so CPU and GPU overlap. The module grows here as the
pipeline gains morsel-level credit windows, N-stage generalization, and mid-stream
fault tolerance.
"""

from __future__ import annotations

from batcher.dist.streaming.pipeline import stream_distributed_pipeline

__all__ = ["stream_distributed_pipeline"]

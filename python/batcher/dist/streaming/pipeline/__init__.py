"""The distributed streaming heterogeneous inference pipeline (the GPU-feeding moat).

`driver` builds one actor pool per resource-class stage and runs a query through them;
`schedule` owns the overlap loop, the credit windows, and the preemption recovery. Read
`driver` first — it says what the pipeline is; `schedule` says how it is kept fed.
"""

from __future__ import annotations

from batcher.dist.streaming.pipeline.driver import stream_distributed_pipeline
from batcher.dist.streaming.pipeline.schedule import run_streamed

__all__ = ["run_streamed", "stream_distributed_pipeline"]

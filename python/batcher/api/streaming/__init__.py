"""The streaming-query surface: the public handle, and the launchers behind `ds.write`.

`api` is the only layer that may sequence Kyber → Core, so the streaming launchers live
here. Split by responsibility:

* `_query` — the `StreamingQuery` handle users hold, and the process-wide active-query
  registry exposed as `bt.streams`;
* `_launch` — the single-node launcher: optimize once, build the per-micro-batch
  processor and sink, start the engine;
* `_distributed` — the same engine with the micro-batch fanned across the cluster: a
  parallel drain for `available_now`/`once`, and a continuous stream whose every epoch is
  staged by workers and published by the driver as one transaction.

Batch and streaming share the one `ds.write(...)` surface — this package is reached only
when that terminal runs in streaming mode (a `Trigger` was set, or a source is unbounded).
"""

from __future__ import annotations

from batcher.api.streaming._distributed import (
    _DRAIN_TRIGGER_KINDS as _DRAIN_TRIGGER_KINDS,
)
from batcher.api.streaming._distributed import (
    start_distributed_stream,
    start_distributed_stream_drain,
)
from batcher.api.streaming._launch import (
    _build_run_batch as _build_run_batch,
)
from batcher.api.streaming._launch import (
    _is_stateless as _is_stateless,
)
from batcher.api.streaming._launch import (
    start_streaming_query,
)
from batcher.api.streaming._query import (
    StreamingQuery,
    active_streams,
    await_any_termination,
)
from batcher.api.streaming._query import (
    _warn_if_checkpoint_not_durable as _warn_if_checkpoint_not_durable,
)

__all__ = [
    "StreamingQuery",
    "active_streams",
    "await_any_termination",
    "start_distributed_stream",
    "start_distributed_stream_drain",
    "start_streaming_query",
]

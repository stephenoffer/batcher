"""Streaming training-data loader — Batcher feeding PyTorch DDP/FSDP/DeepSpeed.

Wraps the deterministic / balanced / elastic / resumable sample ordering from
`streaming_sampler` (which owns that contract, and which Ray Train's split iterator lacks) in a
``torch.utils.data.IterableDataset`` of ``{column: tensor}`` batches for one rank, so it drops
straight into a distributed training loop. Each rank reads its own index slice with no central
coordinator, so a slow or idle rank never blocks the others (the Ray ``#42008`` hang).

Three entry points, one per memory regime: `stream_loader` materializes the dataset once
(`collect()`) and indexes it; `shard_stream_loader` streams shards from disk with a bounded
cache, *computing* its indices instead of materializing them, so a corpus of any size loads
in constant driver memory; `iter_torch_batches` / `streaming_split` consume a lazy batch
stream for sources with no global length. All of them need `torch` (an `ImportError` says so).

The modules split on the seam between *deciding which rows* and *converting them*:

* `tensors` — Arrow → torch, and the device move. Knows nothing about ranks or epochs.
* `indexed` — `stream_loader` / `shard_stream_loader`: a deterministic global order over a
  bounded corpus.
* `lazy` — `iter_torch_batches` / `streaming_split`: an incremental stream with no global
  length.
"""

from __future__ import annotations

# `_rank_shard_stream` stays out of `__all__` — it is not public surface. It is re-exported
# anyway (the `as` form marks that as intentional, not a stray import) because its
# round-completion rule — the fix for the bug where low ranks reached the all-reduce barrier one
# batch ahead and hung the job — is pinned by tests that import it from this path.
from batcher.ml.loader.indexed import shard_stream_loader, stream_loader
from batcher.ml.loader.lazy import (
    _rank_shard_stream as _rank_shard_stream,
)
from batcher.ml.loader.lazy import (
    iter_torch_batches,
    streaming_split,
)
from batcher.ml.loader.tensors import column_to_tensor

__all__ = [
    "column_to_tensor",
    "iter_torch_batches",
    "shard_stream_loader",
    "stream_loader",
    "streaming_split",
]

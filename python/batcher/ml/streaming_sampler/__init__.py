"""Deterministic, resumable, elastic sample ordering for distributed training.

The hard part of a streaming training loader (MosaicML-Streaming's signature feature, and
where Ray Train's ``StreamSplitDataIterator`` struggles — rank hangs, no mid-epoch resume):
give every rank a sample sequence that is

* **deterministic** — same ``(seed, epoch)`` -> same global order (reproducible runs);
* **balanced** — every rank gets the *same* number of samples (``drop_last``), so no rank
  finishes early and stalls the others at the DDP all-reduce barrier;
* **elastic** — the global order is independent of ``world_size``, so a job can resume on
  a differently-sized cluster and still see each sample exactly once;
* **resumable** — checkpoint a global sample position and resume mid-epoch with no
  repeated and no skipped samples.

and, the constraint that decides the design,

* **O(1) memory** — the global order is *computed*, never materialized: a shuffled index
  list costs ~28 bytes per sample in CPython (280 GB of driver RAM for a 10-billion-sample
  corpus, 28 TB for a trillion), so `epoch_permutation` is a keyed pseudorandom bijection
  on ``[0, num_samples)`` instead — index in, shuffled index out, no state. A rank streams
  its slice of an exabyte corpus in constant memory and seeks to any position instantly,
  which is what makes mid-epoch resume O(1) too.

`ordering` holds the arithmetic and `resumable` the one stateful object built on it. Both are
pure index work — no engine, no framework, no I/O — so the ordering contract is exhaustively
unit-testable on its own. A loader layers shard reads, prefetch, and tensor collation on top.
"""

from __future__ import annotations

from batcher.ml.streaming_sampler.ordering import (
    elastic_shard,
    epoch_order,
    epoch_permutation,
    epoch_positions,
    num_rank_batches,
    rank_index_batches,
    rank_shard,
    usable_length,
)
from batcher.ml.streaming_sampler.resumable import ResumableSampler

__all__ = [
    "ResumableSampler",
    "elastic_shard",
    "epoch_order",
    "epoch_permutation",
    "epoch_positions",
    "num_rank_batches",
    "rank_index_batches",
    "rank_shard",
    "usable_length",
]

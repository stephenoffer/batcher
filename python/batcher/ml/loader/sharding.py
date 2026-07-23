"""Build one rank's shard of a corpus by streaming it, never materializing the whole corpus.

`stream_loader` used to call ``dataset.collect()``: with ``world_size=8`` that is eight full
copies of the corpus resident at once, one per rank, each holding the seven-eighths of the rows
it will never train on. It then materialized the epoch order as a Python list of every index —
~28 bytes per sample in CPython, which is 280 GB at ten billion samples and is exactly the
O(1)-memory promise `streaming_sampler` exists to keep.

This module replaces both. The rank's sample sequence is *computed* from the keyed permutation
(a NumPy `uint64` array of ``num_rows / world_size`` entries, not a Python list of ``num_rows``),
the corpus is read once as a batch stream, and only the rows this rank actually trains on are
retained. Peak driver memory drops by a factor of `world_size`, and the sample order is
bit-identical to the order `elastic_shard` produced from the materialized list.

The rows are gathered in corpus order (a forward-only pass over the stream) and reordered into
epoch order afterwards, which is why the permutation comes back alongside the table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher.ml.permutation import _FeistelPermutation, epoch_permutation
from batcher.ml.streaming_sampler import _rank_positions

if TYPE_CHECKING:
    import numpy as np
    import pyarrow as pa

    from batcher.api.dataset import Dataset

__all__ = ["gather_rank_shard"]


def gather_rank_shard(
    dataset: Dataset,
    *,
    num_rows: int,
    world_size: int,
    rank: int,
    epoch: int,
    seed: int,
    shuffle: bool,
    drop_last: bool,
    global_consumed: int,
) -> tuple[pa.Table, np.ndarray]:
    """Stream `dataset` and keep only `rank`'s rows, with the order to read them back in.

    Args:
        dataset: the corpus, read once as a batch stream (never collected).
        num_rows: its row count, as reported by `Dataset.count`.
        world_size: ranks in the data-parallel group.
        rank: this process's slot in that group.
        epoch: selects this epoch's global order (with `seed`).
        seed: keys the permutation.
        shuffle: use the identity order instead when false.
        drop_last: drop the epoch's tail remainder rather than padding it.
        global_consumed: samples already processed this epoch (the resume point).

    Returns:
        A ``(table, order)`` pair: `table` holds only this rank's rows, and ``order[k]`` is the
        row of `table` holding the rank's *k*-th sample this epoch.
    """
    import numpy as np
    import pyarrow as pa

    wanted = _rank_indices(
        num_rows,
        world_size=world_size,
        rank=rank,
        epoch=epoch,
        seed=seed,
        shuffle=shuffle,
        drop_last=drop_last,
        global_consumed=global_consumed,
    )
    schema = None
    chunks: list = []
    # Visit the corpus in increasing row order so one forward pass over the stream suffices.
    gather = np.argsort(wanted, kind="stable")
    sorted_wanted = wanted[gather]
    offset = 0
    for batch in dataset.iter_batches():
        schema = batch.schema
        lo = int(np.searchsorted(sorted_wanted, offset, side="left"))
        hi = int(np.searchsorted(sorted_wanted, offset + batch.num_rows, side="left"))
        if hi > lo:
            chunks.append(batch.take(pa.array(sorted_wanted[lo:hi] - offset)))
        offset += batch.num_rows
    table = pa.Table.from_batches(chunks) if chunks else pa.table({}, schema=schema)
    # `chunks` is in `sorted_wanted` order; invert the sort to get back to epoch order.
    order = np.empty(len(wanted), dtype=np.int64)
    order[gather] = np.arange(len(wanted), dtype=np.int64)
    return table, order


def _rank_indices(
    num_rows: int,
    *,
    world_size: int,
    rank: int,
    epoch: int,
    seed: int,
    shuffle: bool,
    drop_last: bool,
    global_consumed: int,
) -> np.ndarray:
    """This rank's corpus indices for the epoch, in epoch order — `elastic_shard` in NumPy.

    Holds ``num_rows / world_size`` `int64` entries rather than a Python list of every index,
    which is the difference between 8 bytes and ~28 bytes per *retained* sample and between
    O(``num_rows`` / `world_size`) and O(``num_rows``) driver memory.
    """
    import numpy as np

    positions = _rank_positions(num_rows, world_size, rank, global_consumed, drop_last)
    if num_rows <= 0 or len(positions) == 0:
        return np.empty(0, dtype=np.int64)
    pos = np.fromiter(positions, dtype=np.int64, count=len(positions))
    # Positions at or past the corpus exist only in the padded (`drop_last=False`) tail, where
    # the epoch repeats samples from the front — the same wrap `epoch_positions` applies.
    pos = np.where(pos < num_rows, pos, (pos - num_rows) % num_rows)
    permutation = epoch_permutation(num_rows, epoch=epoch, seed=seed, shuffle=shuffle)
    if isinstance(permutation, _FeistelPermutation):
        return permutation.take(pos.astype(np.uint64)).astype(np.int64)
    return pos

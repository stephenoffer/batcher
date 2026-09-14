#!/usr/bin/env python3
"""Draw `data_loader_shards.svg` - one global order, strided to ranks, shuffled in blocks.

Source of truth: `python/batcher/ml/streaming_sampler/ordering.py` (`_rank_positions`,
`usable_length`, `elastic_shard`), `python/batcher/ml/loader/lazy.py` (`_shuffled_blocks`,
`_rebatch`, `_SHUFFLE_BLOCK_MAX_BYTES`), `python/batcher/interop/arrays.py`
(`arrays_to_torch`) and `python/batcher/ml/loader/tensors.py` (`DeviceMover`).

Two facts are what the picture is for. A rank's shard is a *stride* over one global
order - `range(first, usable, world_size)` - which is why `global_consumed` is a position
in that global order and a run can resume on a differently sized cluster. And the local
shuffle is a block fill followed by one permutation, **not** a reservoir: a row never
crosses a block boundary, so a corpus written in label order stays clumped.
"""

from __future__ import annotations

from _authoring import arrow, band, card, label, note, svg, write

W, H = 980, 730

body = [
    band(20, 20, 940, 96, "ONE GLOBAL ORDER, INDEPENDENT OF THE CLUSTER SIZE", "grey"),
    card(
        180,
        44,
        620,
        56,
        "ds.ml.stream_loader(batch_size, world_size, rank, epoch)",
        "a seeded permutation of every row, computed the same way on every rank",
    ),
    arrow(400, 104, 220, 178, "blue"),
    label(292, 132, "rank 0's rows", anchor="middle", size=12),
    arrow(580, 104, 760, 178, "blue"),
    label(688, 132, "rank 1's rows", anchor="middle", size=12),
    # The shard is a stride, not a contiguous split.
    band(20, 140, 940, 152, "ONE SHARD PER RANK  -  A STRIDE, NOT A SPLIT", "blue"),
    card(60, 184, 300, 82, "rank 0", "rows 0, W, 2W, 3W, ..."),
    card(620, 184, 300, 82, "rank 1", "rows 1, W+1, 2W+1, ..."),
    note(490, 214, "trimmed or padded to a multiple", anchor="middle"),
    note(490, 232, "of world_size, so every rank", anchor="middle"),
    note(490, 250, "yields the same count.", anchor="middle"),
    # What one rank then does with its stride.
    band(20, 322, 940, 270, "WHAT ONE RANK THEN DOES", "amber"),
    label(329, 366, "batched", anchor="middle", size=12),
    label(645, 366, "non_blocking", anchor="middle", size=12),
    card(44, 378, 260, 86, "shuffle block", "fill, then permute once"),
    card(360, 378, 260, 86, "tensors", "numeric columns only"),
    card(676, 378, 260, 86, "device", "its own copy stream"),
    arrow(304, 421, 354, 421, "amber"),
    arrow(620, 421, 670, 421, "amber"),
    note(174, 492, "Fills to shuffle_block_size rows", anchor="middle"),
    note(174, 510, "or 256 MiB, then one permutation.", anchor="middle"),
    note(174, 534, "Not a reservoir: a row never", anchor="middle"),
    note(174, 552, "crosses a block boundary.", anchor="middle"),
    note(490, 492, "Non-numeric columns are dropped,", anchor="middle"),
    note(490, 510, "announced once. zero_copy views", anchor="middle"),
    note(490, 534, "the buffer through dlpack;", anchor="middle"),
    note(490, 552, "otherwise it copies.", anchor="middle"),
    note(806, 492, "The pinned staging tensor is held", anchor="middle"),
    note(806, 510, "for the next few batches, so an", anchor="middle"),
    note(806, 534, "in-flight copy cannot read", anchor="middle"),
    note(806, 552, "memory that was already freed.", anchor="middle"),
    # Resume.
    band(20, 622, 940, 88, "RESUME", "grey"),
    note(490, 664, "global_consumed counts positions in the global order, not in a rank's shard, so a run resumes on a differently sized cluster.", anchor="middle"),
    note(490, 686, "It must land on a multiple of world_size, a synchronized step boundary, or the ranks come back with unequal counts.", anchor="middle"),
]

write("data_loader_shards", svg(W, H, "".join(body)))
print("wrote data_loader_shards.svg")

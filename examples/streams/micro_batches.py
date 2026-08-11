"""Batch is the bounded case of streaming: the same operators over a batch at a time.

`iter_batches` is the streaming read, and an aggregate accumulated across batches is the
same computation the engine does internally. Doing it by hand once is worth it to see that
there is no second semantics hiding behind the streaming API.

    python examples/streams/micro_batches.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem").select("l_shipmode", "l_quantity", "l_extendedprice")

    # A mergeable accumulator, updated one micro-batch at a time.
    running: dict[str, list[float]] = {}
    batches = 0
    for batch in lineitem.iter_batches(batch_size=16_384):
        batches += 1
        modes = batch.column("l_shipmode").to_pylist()
        quantities = batch.column("l_quantity").to_pylist()
        for mode, quantity in zip(modes, quantities, strict=True):
            state = running.setdefault(mode, [0.0, 0.0])
            state[0] += 1
            state[1] += quantity

    print(f"{batches} micro-batches, {len(running)} keys")
    assert batches > 1

    # The engine's one-shot answer over the same data.
    reference = (
        lineitem.group_by("l_shipmode")
        .agg(lines=col("l_quantity").count(), qty=col("l_quantity").sum())
        .sort("l_shipmode")
        .to_pydict()
    )

    for index, mode in enumerate(reference["l_shipmode"]):
        assert running[mode][0] == reference["lines"][index], mode
        assert abs(running[mode][1] - reference["qty"][index]) < 1e-6, mode
    print("micro-batched accumulation matches the one-shot aggregate")

    # Which is the point: the same operator semantics, incrementally scheduled. Do the
    # accumulation in the engine, not in Python — this loop is an illustration, not advice.


if __name__ == "__main__":
    main()

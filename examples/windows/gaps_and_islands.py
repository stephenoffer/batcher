"""Finding consecutive runs: the gaps-and-islands pattern.

Subtract a row number from a sequential key and consecutive rows share the result. That
shared value is the island id, and the whole technique is that one subtraction — after
which finding run lengths is an ordinary group-by.

    python examples/windows/gaps_and_islands.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    # Order keys in TPC-H are sparse: they skip values, which makes them a natural
    # gaps-and-islands subject.
    keys = tpch("orders").select("o_orderkey").sort("o_orderkey").head(200)

    islands = keys.with_columns(
        position=bt.row_number().over(order_by=["o_orderkey"]),
    ).with_columns(island=col("o_orderkey") - col("position"))

    runs = (
        islands.group_by("island")
        .agg(
            length=bt.count(),
            start=col("o_orderkey").min(),
            end=col("o_orderkey").max(),
        )
        .sort("start")
        .to_pydict()
    )
    print(f"{len(runs['island'])} runs in 200 keys")
    print("longest run:", max(runs["length"]))

    # Every key belongs to exactly one run.
    assert sum(runs["length"]) == 200

    # A run really is consecutive: its span matches its length.
    assert all(
        end - start + 1 == length
        for start, end, length in zip(runs["start"], runs["end"], runs["length"], strict=True)
    )

    # And the runs are disjoint and ordered.
    assert all(
        previous_end < next_start
        for previous_end, next_start in zip(runs["end"], runs["start"][1:], strict=False)
    )


if __name__ == "__main__":
    main()

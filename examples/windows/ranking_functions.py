"""Ranking within a partition: row_number, rank, and dense_rank.

The three differ only in how they treat ties. `row_number` breaks them arbitrarily,
`rank` gives ties the same number and then skips, `dense_rank` gives ties the same number
and does not skip. Picking the wrong one shows up as a gap in a leaderboard.

    python examples/windows/ranking_functions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem").head(5_000)

    ranked = lineitem.select(
        "l_orderkey",
        "l_linenumber",
        "l_quantity",
        row=bt.row_number().over(partition_by=["l_orderkey"], order_by=[("l_quantity", True)]),
        rank=bt.rank().over(partition_by=["l_orderkey"], order_by=[("l_quantity", True)]),
        dense=bt.dense_rank().over(partition_by=["l_orderkey"], order_by=[("l_quantity", True)]),
    )

    sample_key = lineitem.head(1).to_pydict()["l_orderkey"][0]
    rows = ranked.filter(col("l_orderkey") == sample_key).sort("row").to_pydict()
    print(rows)

    # `row_number` is a dense 1..n with no repeats, whatever the ties.
    assert rows["row"] == list(range(1, len(rows["row"]) + 1))
    # `rank` can skip; `dense_rank` cannot, so dense is never larger.
    assert all(dense <= rank for rank, dense in zip(rows["rank"], rows["dense"], strict=True))
    assert max(rows["dense"]) <= max(rows["rank"])

    # Across the whole table: each partition restarts at 1.
    firsts = ranked.filter(col("row") == 1).count()
    assert firsts == lineitem.n_unique("l_orderkey")


if __name__ == "__main__":
    main()

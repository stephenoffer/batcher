"""Finding the row that holds an extreme, not just the extreme value.

`max` tells you the largest price. `arg_max` tells you which order it belongs to. Doing
that with a sort and a limit works, but costs a full ordering; `arg_max` is a single pass
and composes inside a group-by.

    python examples/aggregations/argmin_argmax.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem").select("l_orderkey", "l_extendedprice", "l_shipmode")

    extremes = lineitem.agg(
        dearest_price=col("l_extendedprice").max(),
        dearest_order=bt.arg_max(col("l_orderkey"), col("l_extendedprice")),
        cheapest_price=col("l_extendedprice").min(),
        cheapest_order=bt.arg_min(col("l_orderkey"), col("l_extendedprice")),
    ).to_pydict()
    print(extremes)

    # Cross-check against the sort-and-take-one version.
    by_sort = lineitem.sort("l_extendedprice", descending=True).head(1).to_pydict()
    assert extremes["dearest_price"][0] == by_sort["l_extendedprice"][0]
    assert extremes["dearest_order"][0] == by_sort["l_orderkey"][0]

    # Per group: the priciest line for each ship mode.
    per_mode = (
        lineitem.group_by("l_shipmode")
        .agg(
            top_price=col("l_extendedprice").max(),
            top_order=bt.arg_max(col("l_orderkey"), col("l_extendedprice")),
        )
        .sort("l_shipmode")
        .to_pydict()
    )
    print(per_mode)
    assert len(per_mode["l_shipmode"]) == lineitem.n_unique("l_shipmode")
    assert max(per_mode["top_price"]) == extremes["dearest_price"][0]


if __name__ == "__main__":
    main()

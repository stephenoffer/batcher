"""Several grouping levels in one pass: rollup, cube, and grouping sets.

Running the same aggregate at three levels of detail is three scans if you write three
queries. `rollup` computes the hierarchy, `cube` every combination, and `grouping_sets`
exactly the ones you name — all from one pass, with nulls marking the totalled columns.

    python examples/relational/grouping_sets_cube_rollup.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem").select("l_returnflag", "l_linestatus", "l_quantity")

    # Rollup: (flag, status), then (flag), then the grand total.
    rolled = (
        lineitem.rollup("l_returnflag", "l_linestatus")
        .agg(qty=col("l_quantity").sum(), lines=bt.count())
        .sort("l_returnflag", "l_linestatus")
        .to_pydict()
    )
    print("rollup rows:", len(rolled["qty"]))

    grand_total = [
        qty
        for qty, flag, status in zip(
            rolled["qty"], rolled["l_returnflag"], rolled["l_linestatus"], strict=True
        )
        if flag is None and status is None
    ]
    assert len(grand_total) == 1
    assert grand_total[0] == lineitem.agg(q=col("l_quantity").sum()).to_pydict()["q"][0]

    # Cube adds the (status) level that rollup skips.
    cubed = lineitem.cube("l_returnflag", "l_linestatus").agg(lines=bt.count()).to_pydict()
    assert len(cubed["lines"]) > len(rolled["qty"])

    # Grouping sets: name exactly the levels you want and nothing else.
    chosen = (
        lineitem.grouping_sets(["l_returnflag"], ["l_linestatus"]).agg(lines=bt.count()).to_pydict()
    )
    flags = lineitem.n_unique("l_returnflag")
    statuses = lineitem.n_unique("l_linestatus")
    assert len(chosen["lines"]) == flags + statuses


if __name__ == "__main__":
    main()

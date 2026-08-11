"""Frequency tables: value_counts for one column, crosstab for two.

Both are group-by shorthands. Knowing that is the point: when the shorthand stops fitting
— a different aggregate, a filter, more than two dimensions — drop to `group_by` rather
than looking for a longer shorthand.

    python examples/relational/crosstab_and_value_counts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem")

    counts = lineitem.value_counts("l_shipmode").sort("count", descending=True).to_pydict()
    print(counts)
    assert counts["count"] == sorted(counts["count"], reverse=True)
    assert sum(counts["count"]) == lineitem.count()

    # The same thing, spelled as the group-by it is.
    equivalent = (
        lineitem.group_by("l_shipmode")
        .agg(count=bt.count())
        .sort("count", descending=True)
        .to_pydict()
    )
    assert equivalent["count"] == counts["count"]

    # Two dimensions at once.
    table = lineitem.crosstab("l_returnflag", "l_linestatus").sort("l_returnflag").to_pydict()
    print(table)
    assert table["l_returnflag"] == sorted(table["l_returnflag"])
    # Every cell in the grid is counted exactly once.
    cells = [
        value
        for name, column in table.items()
        if name != "l_returnflag"
        for value in column
        if value is not None
    ]
    assert sum(cells) == lineitem.count()
    assert col is not None


if __name__ == "__main__":
    main()

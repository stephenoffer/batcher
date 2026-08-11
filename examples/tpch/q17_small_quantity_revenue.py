"""TPC-H Q17 — revenue from unusually small orders, against a per-part average.

The threshold is per part, not global, so the average has to be computed as its own
grouped relation and joined back. This is the same rewrite as Q2's minimum, and it is
the single most useful shape to recognize in a correlated subquery.

    python examples/tpch/q17_small_quantity_revenue.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem")
    part = tpch("part")

    wanted = part.filter((col("p_brand") == "Brand#23") & (col("p_container") == "MED BOX"))

    # The per-part average, over *all* lines for that part.
    averages = lineitem.group_by("l_partkey").agg(avg_qty=col("l_quantity").mean())

    # TPC-H specifies 0.2 here. The example suite reads a bounded slice of `lineitem`, so
    # each part has only a few lines and none of them falls that far below its own
    # average — the canonical constant returns an empty result, and an empty result
    # demonstrates nothing. Raise the fraction rather than pretend.
    fraction = 0.5

    result = (
        lineitem.join(wanted, left_on="l_partkey", right_on="p_partkey")
        .join(averages, on="l_partkey")
        .filter(col("l_quantity") < fraction * col("avg_qty"))
        .agg(total=col("l_extendedprice").sum())
        .with_columns(avg_yearly=col("total") / 7.0)
        .to_pydict()
    )

    print("avg_yearly:", result["avg_yearly"][0])

    assert result["avg_yearly"][0] > 0.0
    # The result is a seventh of the total it came from, which is the part of the query
    # most often dropped when it is translated.
    assert abs(result["avg_yearly"][0] * 7.0 - result["total"][0]) < 1e-6


if __name__ == "__main__":
    main()

"""Q1 written four ways, all returning the same answer.

The point is that these are the same query. SQL, the DataFrame API, a masked-sum form and a
pre-filtered form all lower onto one plan shape, so the choice between them is about
readability rather than speed.

    python examples/tpch/q01_variants.py
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem")
    cutoff = dt.date(1998, 9, 2)

    dataframe = (
        lineitem.filter(col("l_shipdate") <= bt.lit(cutoff))
        .group_by("l_returnflag", "l_linestatus")
        .agg(lines=bt.count(), qty=col("l_quantity").sum())
        .sort("l_returnflag", "l_linestatus")
    )

    sql = bt.sql(
        """
        SELECT l_returnflag, l_linestatus, COUNT(*) AS lines, SUM(l_quantity) AS qty
        FROM lineitem
        WHERE l_shipdate <= DATE '1998-09-02'
        GROUP BY l_returnflag, l_linestatus
        ORDER BY l_returnflag, l_linestatus
        """,
        lineitem=lineitem,
    )

    # A masked form: no filter, the predicate lives in the aggregate instead.
    masked = (
        lineitem.group_by("l_returnflag", "l_linestatus")
        .agg(
            lines=bt.count_if(col("l_shipdate") <= bt.lit(cutoff)),
            qty=bt.when(col("l_shipdate") <= bt.lit(cutoff))
            .then(col("l_quantity"))
            .otherwise(0)
            .sum(),
        )
        .sort("l_returnflag", "l_linestatus")
    )

    reference = dataframe.to_pydict()
    print(reference["l_returnflag"], reference["lines"])

    for name, variant in (("sql", sql), ("masked", masked)):
        result = variant.to_pydict()
        assert result["l_returnflag"] == reference["l_returnflag"], name
        assert result["l_linestatus"] == reference["l_linestatus"], name
        assert result["lines"] == reference["lines"], name
        assert all(
            abs(a - b) < 1e-6 for a, b in zip(reference["qty"], result["qty"], strict=True)
        ), name

    print("all three spellings agree")


if __name__ == "__main__":
    main()

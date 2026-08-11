"""Long to wide and back: pivot and unpivot.

A pivot turns distinct values of one column into columns of their own, so its output
schema depends on the *data*. That means the engine has to see the data before it can
know the schema, which is why a pivot is a pipeline breaker and a plain group-by is not.

    python examples/relational/pivot_and_unpivot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem")

    revenue = lineitem.with_columns(
        revenue=col("l_extendedprice") * (1 - col("l_discount"))
    ).select("l_returnflag", "l_linestatus", "revenue")

    wide = revenue.pivot(
        index=["l_returnflag"],
        on="l_linestatus",
        values="revenue",
        aggregate="sum",
    ).sort("l_returnflag")

    result = wide.to_pydict()
    print(result)

    # One row per return flag, one column per line status seen in the data.
    assert result["l_returnflag"] == sorted(result["l_returnflag"])
    assert set(result) - {"l_returnflag"} <= {"F", "O"}

    # Unpivot is the inverse: the generated columns become rows again.
    long_again = wide.unpivot(
        index=["l_returnflag"], variable_name="l_linestatus", value_name="revenue"
    ).filter(col("revenue").is_not_null())

    print(long_again.sort("l_returnflag", "l_linestatus").to_pydict())

    # Round trip: the same totals, however they are laid out.
    original_total = revenue.agg(total=col("revenue").sum()).to_pydict()["total"][0]
    round_trip_total = long_again.agg(total=col("revenue").sum()).to_pydict()["total"][0]
    assert abs(original_total - round_trip_total) < 1e-3


if __name__ == "__main__":
    main()

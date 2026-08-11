"""How nulls travel through each operator.

Every operator treats a null slightly differently, and the differences are all defensible
individually. Together they are the reason a null count changes halfway down a pipeline for
no visible reason, so this is the table worth knowing.

    python examples/relational/nulls_across_operators.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders").select("o_orderkey", "o_custkey")
    lineitem = tpch("lineitem").select("l_orderkey", "l_quantity")

    joined = orders.join(lineitem, left_on="o_orderkey", right_on="l_orderkey", how="left")
    nulls = joined.filter(col("l_quantity").is_null()).count()
    print("null quantities after the left join:", nulls)
    assert nulls > 0

    # A filter drops nulls: a comparison against null is null, which is not true.
    kept = joined.filter(col("l_quantity") > 0).count()
    dropped = joined.count() - kept
    print(f"filter dropped {dropped} rows, of which {nulls} were null")
    assert dropped >= nulls

    # An aggregate skips them: count is over non-nulls, sum ignores them.
    summary = joined.agg(
        rows=bt.count(),
        non_null=col("l_quantity").count(),
        total=col("l_quantity").sum(),
    ).to_pydict()
    assert summary["rows"][0] - summary["non_null"][0] == nulls

    # A group-by keeps them: null is a group.
    grouped = joined.group_by("l_quantity").agg(n=bt.count())
    has_null_group = grouped.filter(col("l_quantity").is_null()).count()
    print("null forms its own group:", has_null_group == 1)
    assert has_null_group == 1

    # `distinct` keeps one of them.
    distinct_values = joined.select("l_quantity").distinct()
    assert distinct_values.filter(col("l_quantity").is_null()).count() == 1

    # And a sort puts them somewhere definite rather than dropping them.
    sorted_rows = joined.select("l_quantity").sort("l_quantity")
    assert sorted_rows.count() == joined.count()


if __name__ == "__main__":
    main()

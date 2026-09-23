"""Measuring where a query spends its time.

`stats()` runs the query and reports what each operator measured: rows in and out, wall
time, and its share of the total. `explain(analyze=True)` renders the same measurements
beside the planner's estimates. Reading them is how you know whether the join or the scan
is the problem; guessing from the plan shape is how people end up optimizing the cheap half.

`Dataset.profile()` is a different tool: it summarizes the *data* (a count, null count, and
approximate distinct count per column), not the query's execution.

    python examples/operations/profiling_a_query.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem")
    orders = tpch("orders")

    query = (
        lineitem.join(orders, left_on="l_orderkey", right_on="o_orderkey")
        .group_by("o_orderpriority")
        .agg(revenue=(col("l_extendedprice") * (1 - col("l_discount"))).sum(), lines=bt.count())
        .sort("o_orderpriority")
    )

    # Per-operator measurements from an actual run.
    stats = query.stats()
    print(stats)
    by_kind = {op.kind: op for op in stats.ops}
    assert {"scan", "hash_join", "aggregate", "sort"} <= set(by_kind), sorted(by_kind)

    priorities = orders.count_distinct("o_orderpriority")
    line_count = lineitem.count()
    # The aggregate turned every joined line into one row per priority, and every lineitem
    # row has a matching order, so the join passed all of them through.
    assert by_kind["aggregate"].rows_out == priorities
    assert by_kind["hash_join"].rows_out == line_count
    assert any(op.kind == "scan" and op.rows_out == line_count for op in stats.ops)
    assert stats.rows_out == priorities
    assert stats.bottleneck is not None and stats.bottleneck.kind in by_kind

    # The same measurements as a machine-readable document, estimate beside actual.
    doc = json.loads(query.explain(analyze=True, format="json"))
    join = next(op for op in doc["ops"] if op["kind"] == "hash_join")
    print("join estimate", join["est_rows"], "actual", join["rows_out"])
    assert join["measured"] and join["rows_out"] == line_count

    # Measuring does not change the answer.
    result = query.to_pydict()
    print(result["o_orderpriority"], result["lines"])
    assert len(result["o_orderpriority"]) == priorities
    assert sum(result["lines"]) == line_count

    # `profile()` answers "what is in these columns", one row per column.
    data_profile = query.profile().to_pydict()
    print(data_profile)
    assert data_profile["column"] == ["o_orderpriority", "revenue", "lines"]
    assert data_profile["null_count"] == [0, 0, 0]


if __name__ == "__main__":
    main()

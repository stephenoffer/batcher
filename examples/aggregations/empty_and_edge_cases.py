"""What an aggregate returns when there is nothing to aggregate.

An empty sum is null, not zero, and an empty count is zero, not null. Both are right, and
the asymmetry is what catches people: a report that divides a sum by a count over an empty
filter gets a null rather than a division error.

    python examples/aggregations/empty_and_edge_cases.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem").select("l_quantity", "l_extendedprice")

    # A filter that matches nothing.
    empty = lineitem.filter(col("l_quantity") > 10_000)
    assert empty.count() == 0
    assert empty.is_empty()

    result = empty.agg(
        rows=bt.count(),
        total=col("l_quantity").sum(),
        average=col("l_quantity").mean(),
        smallest=col("l_quantity").min(),
        distinct=bt.n_unique(col("l_quantity")),
    ).to_pydict()
    print(result)

    # Count is zero; the value aggregates are null.
    assert result["rows"][0] == 0
    assert result["total"][0] is None
    assert result["average"][0] is None
    assert result["smallest"][0] is None
    assert result["distinct"][0] == 0

    # A single row is the other edge: variance needs two.
    single = lineitem.head(1)
    edge = single.agg(
        sample_var=bt.var(col("l_quantity")),
        population_var=bt.var_pop(col("l_quantity")),
        mean=col("l_quantity").mean(),
    ).to_pydict()
    print(edge)
    assert edge["population_var"][0] == 0.0
    assert edge["sample_var"][0] is None or edge["sample_var"][0] != edge["sample_var"][0]

    # A grouped aggregate over an empty input produces no groups at all.
    grouped = empty.group_by("l_quantity").agg(n=bt.count())
    assert grouped.count() == 0


if __name__ == "__main__":
    main()

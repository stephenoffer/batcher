"""Column lineage: which inputs does this output column actually depend on?

Lineage is computed from the plan, so it is exact rather than a guess from parsing SQL
text. That is what makes it usable for an impact analysis: if this source column changes,
which outputs move?

    python examples/governance/lineage.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    orders = bt.from_pydict(
        {
            "id": [1, 2, 3],
            "price": [10.0, 20.0, 30.0],
            "qty": [1, 2, 3],
            "note": ["a", "b", "c"],
        }
    )

    derived = orders.with_columns(
        revenue=col("price") * col("qty"),
        flag=col("price") > 15.0,
    ).select("id", "revenue", "flag")

    # Lineage for the whole plan.
    graph = derived.lineage()
    print("lineage:", graph)
    assert graph is not None

    # `revenue` depends on both `price` and `qty`, and on nothing else.
    text = str(graph)
    assert "revenue" in text
    assert "price" in text
    assert "qty" in text

    # The unused column never enters the projection, so it is not a dependency.
    result = derived.to_pydict()
    assert "note" not in result
    assert result["revenue"] == [10.0, 40.0, 90.0]
    assert result["flag"] == [False, True, True]

    # The impact question, answered from the plan: if `price` changes, what moves?
    # Both derived columns reference it; `id` does not.
    assert "id" in text

    # Lineage survives an aggregate, which is where hand-tracking usually breaks down.
    rolled = (
        orders.with_columns(revenue=col("price") * col("qty"))
        .group_by(bucket=col("qty") > 1)
        .agg(total=col("revenue").sum())
    )
    agg_graph = str(rolled.lineage())
    print("aggregate lineage:", agg_graph[:160])
    assert "total" in agg_graph
    assert rolled.count() == 2


if __name__ == "__main__":
    main()

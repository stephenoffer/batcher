"""Reading a plan, timing a query, and checking what the engine actually ran.

``explain()`` shows the optimized plan, which is where you confirm a predicate really was
pushed into the scan. Reading the plan is faster than guessing, and it is the only way to
tell a fused pipeline from three separate passes.

    python examples/operations/inspecting_a_query.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    ds = bt.from_pydict(
        {
            "region": ["us", "eu", "us", "eu"],
            "amount": [10, 20, 30, 40],
            "note": ["a", "b", "c", "d"],
        }
    )

    query = (
        ds.filter(col("amount") > 15)
        .select("region", "amount")
        .group_by("region")
        .agg(total=col("amount").sum())
    )

    # The optimized plan, as text: one line per operator with its row estimate, so you
    # can confirm the filter really sits below the aggregate.
    plan = query.explain()
    print(plan)
    assert isinstance(plan, str)
    assert "aggregate" in plan
    assert "filter" in plan
    assert "scan" in plan
    # The filter is nested under the aggregate, i.e. it runs first.
    assert plan.index("aggregate") < plan.index("filter") < plan.index("scan")

    # The plan as structured data, for asserting on shape in a test.
    structured = ds.meta.explain()
    assert isinstance(structured, dict)

    # Version and build information, for a bug report.
    versions = bt.versions()
    print("versions:", {k: versions[k] for k in sorted(versions)[:4]})
    assert "batcher" in versions or len(versions) > 0

    # Quick descriptive summaries, without writing the aggregates yourself.
    described = ds.describe().to_pydict()
    print("describe columns:", sorted(described)[:5])
    assert len(described) > 0

    counts = ds.value_counts("region").to_pydict()
    print("value_counts:", counts)
    assert sorted(counts["region"]) == ["eu", "us"]

    nulls = ds.null_count().to_pydict()
    assert isinstance(nulls, dict)

    # `glimpse`/`info` print a compact overview; `show` prints rows.
    ds.glimpse()
    ds.show(2)

    # Caching a reused intermediate, so it is computed once.
    shared = ds.filter(col("amount") > 15).cache()
    first = shared.select(t=col("amount").sum()).to_pydict()
    second = shared.count()
    assert first["t"] == [90]
    assert second == 3

    # The result is identical with and without the cache -- it is purely an execution hint.
    uncached = ds.filter(col("amount") > 15).select(t=col("amount").sum()).to_pydict()
    assert uncached == first


if __name__ == "__main__":
    main()

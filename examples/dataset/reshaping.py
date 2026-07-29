"""Reshaping: pivot, unpivot, explode, unnest, and set operations.

Long-to-wide and back is the most common reshape in reporting. Pivot needs to know the
value columns it will produce, which means it materializes; unpivot is the cheap direction
and is usually what a downstream model actually wants.

    python examples/dataset/reshaping.py
"""

from __future__ import annotations

import batcher as bt


def main() -> None:
    long = bt.from_pydict(
        {
            "region": ["us", "us", "eu", "eu"],
            "quarter": ["q1", "q2", "q1", "q2"],
            "revenue": [10, 20, 30, 40],
        }
    )

    # Long -> wide.
    wide = long.pivot(index=["region"], on="quarter", values="revenue").sort("region")
    w = wide.to_pydict()
    print("pivoted:", w)
    assert sorted(w) == ["q1", "q2", "region"]
    assert w["region"] == ["eu", "us"]
    assert w["q1"] == [30, 10]

    # Wide -> long, the inverse.
    back = wide.unpivot(
        index=["region"], on=["q1", "q2"], variable_name="quarter", value_name="revenue"
    ).sort("region", "quarter")
    b = back.to_pydict()
    print("unpivoted:", b)
    assert len(b["revenue"]) == 4
    assert b["quarter"] == ["q1", "q2", "q1", "q2"]

    # `melt` is the pandas spelling of unpivot.
    melted = wide.melt(id_vars=["region"], value_vars=["q1", "q2"]).to_pydict()
    assert len(melted[next(iter(melted))]) == 4

    # Explode a list column into one row per element.
    nested = bt.from_pydict({"id": [1, 2], "tags": [["a", "b"], ["c"]]})
    exploded = nested.explode("tags").to_pydict()
    print("exploded:", exploded)
    assert exploded["id"] == [1, 1, 2]

    # Unnest a struct column into one column per field.
    structs = bt.from_pydict({"who": [{"first": "ada", "last": "l"}]})
    flat = structs.unnest("who").to_pydict()
    print("unnested:", sorted(flat))
    assert "first" in flat and "last" in flat

    # Set operations over two datasets with the same schema.
    a = bt.from_pydict({"v": [1, 2, 3]})
    b2 = bt.from_pydict({"v": [3, 4]})
    assert sorted(a.union(b2).to_pydict()["v"]) == [1, 2, 3, 3, 4]
    assert sorted(a.union(b2, distinct=True).to_pydict()["v"]) == [1, 2, 3, 4]
    assert a.intersect(b2).to_pydict()["v"] == [3]
    assert sorted(a.except_(b2).to_pydict()["v"]) == [1, 2]

    # Row numbering and counting, for a stable key or a rank.
    numbered = a.with_row_index("idx").to_pydict()
    print("numbered:", numbered)
    assert numbered["idx"] == [0, 1, 2]

    # Crosstab: a frequency pivot in one call.
    pairs = bt.from_pydict({"x": ["a", "a", "b"], "y": ["p", "q", "p"]})
    ct = pairs.crosstab("x", "y").to_pydict()
    print("crosstab:", ct)
    assert len(ct["x"]) == 2


if __name__ == "__main__":
    main()

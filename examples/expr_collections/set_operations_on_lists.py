"""Treating list columns as sets, per row.

Set operations on a list column answer per-row questions a join cannot: which tags does this
document share with that one, and how much do two rows overlap.

Watch `list.jaccard`: it is *positional agreement*, the fraction of indices where the two
lists hold the same value. That is the right estimator over two `str.minhash` signatures
and is not the set Jaccard index. For sets, divide the intersection by the union.

    python examples/expr_collections/set_operations_on_lists.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from batcher import col


def main() -> None:
    rows = bt.from_pydict(
        {
            "id": [1, 2, 3],
            "left": [["a", "b", "c"], ["a", "b"], ["x"]],
            "right": [["b", "c", "d"], ["a", "b"], ["y", "z"]],
        }
    )

    compared = rows.select(
        "id",
        shared=col("left").list.intersect(col("right")),
        combined=col("left").list.union(col("right")),
        only_left=col("left").list.difference(col("right")),
        positional=col("left").list.jaccard(col("right")),
        overlap=col("left").list.multiset_overlap(col("right")),
        has_all=col("left").list.has_all(col("right")),
        has_any=col("left").list.has_any(col("right")),
    )
    result = compared.to_pydict()
    print(result)

    # Row 1: {a,b,c} vs {b,c,d} — two shared of four combined.
    assert sorted(result["shared"][0]) == ["b", "c"]
    assert sorted(result["combined"][0]) == ["a", "b", "c", "d"]
    assert result["only_left"][0] == ["a"]

    # Row 2 is identical on both sides; row 3 is disjoint.
    assert result["has_all"][1] is True
    assert result["shared"][2] == []
    assert result["has_any"][2] is False

    # The set Jaccard index: intersection over union, computed from the two columns.
    set_jaccard = compared.with_columns(
        jaccard=col("shared").list.len() / col("combined").list.len()
    ).to_pydict()["jaccard"]
    print("set jaccard:", set_jaccard)
    assert abs(set_jaccard[0] - 0.5) < 1e-9
    assert abs(set_jaccard[1] - 1.0) < 1e-9
    assert abs(set_jaccard[2]) < 1e-9

    # `list.jaccard` answers the other question — positional agreement — and gives a
    # different number for row 1, where the shared values sit at different indices.
    print("positional agreement:", result["positional"])
    assert result["positional"][0] != set_jaccard[0]
    assert abs(result["positional"][1] - 1.0) < 1e-9

    # A similarity filter, which is what all of this is for.
    similar = [
        identifier
        for identifier, score in zip(result["id"], set_jaccard, strict=True)
        if score >= 0.5
    ]
    assert set(similar) == {1, 2}


if __name__ == "__main__":
    main()

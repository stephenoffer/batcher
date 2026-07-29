"""Treating two list columns as sets: union, intersection, difference, overlap.

This is the shape behind "which tags do these two documents share" and "what did the user
add to the cart since last time". Everything is per row and columnar, so a set operation
over a million rows never builds a million Python sets.

    python examples/expressions/lists_set_operations.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    docs = bt.from_pydict(
        {
            "before": [["a", "b", "c"], ["x"], []],
            "after": [["b", "c", "d"], ["x", "y"], ["z"]],
        }
    )

    diffed = docs.with_columns(
        both=col("before").list.intersect(col("after")),
        either=col("before").list.union(col("after")),
        removed=col("before").list.difference(col("after")),
        added=col("after").list.difference(col("before")),
        # Predicates rather than lists, when you only need a yes/no.
        covers_all=col("after").list.has_all(col("before")),
        overlaps=col("before").list.has_any(col("after")),
        # Concatenation keeps duplicates; union does not.
        appended=col("before").list.concat(col("after")),
        # Set similarity: |intersection| / |union|. Build it from the two set operators
        # above. (`.list.jaccard` is a different thing -- a position-by-position
        # agreement rate for `str.minhash` signatures, not a set overlap.)
        shared=col("before").list.intersect(col("after")).list.len(),
        combined=col("before").list.union(col("after")).list.len(),
    )

    result = diffed.to_pydict()
    print(result)

    assert sorted(result["both"][0]) == ["b", "c"]
    assert sorted(result["either"][0]) == ["a", "b", "c", "d"]
    assert result["removed"][0] == ["a"]
    assert result["added"][0] == ["d"]
    assert result["overlaps"] == [True, True, False]
    # Row 2 gained "y" while keeping "x", so `after` covers everything `before` had.
    assert result["covers_all"][1] is True
    assert result["appended"][0] == ["a", "b", "c", "b", "c", "d"]
    # |{b,c}| / |{a,b,c,d}| = 0.5
    assert result["shared"][0] == 2
    assert result["combined"][0] == 4
    jaccard = [
        s / c if c else None for s, c in zip(result["shared"], result["combined"], strict=True)
    ]
    print(jaccard)
    assert jaccard[0] == 0.5


if __name__ == "__main__":
    main()

"""Reshaping a list column: slicing, flattening, joining, and set operations.

These keep the list *as a list*, which is the difference from `explode`. Reach for them
when the per-row grouping is meaningful and you do not want one row per element.

    python examples/expr_collections/list_transforms.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from batcher import col


def main() -> None:
    data = bt.from_pydict(
        {
            "id": [1, 2, 3],
            "tags": [["red", "green", "blue", "red"], ["green"], []],
            "other": [["blue", "black"], ["green", "grey"], ["white"]],
        }
    )

    shaped = data.select(
        "id",
        size=col("tags").list.len(),
        head_two=col("tags").list.head(2),
        window=col("tags").list.slice(1, 2),
        joined=col("tags").list.join(","),
        shared=col("tags").list.intersect(col("other")),
        combined=col("tags").list.union(col("other")),
        only_mine=col("tags").list.difference(col("other")),
    )

    result = shaped.to_pydict()
    print(result)

    assert result["size"] == [4, 1, 0]
    assert result["head_two"][0] == ["red", "green"]
    assert result["window"][0] == ["green", "blue"]
    assert result["joined"][0] == "red,green,blue,red"
    # The empty list stays empty rather than becoming null.
    assert result["joined"][2] == ""

    # Set operations deduplicate, which is what makes them set operations.
    assert result["shared"][0] == ["blue"]
    assert set(result["combined"][1]) == {"green", "grey"}
    assert set(result["only_mine"][0]) == {"red", "green"}

    # Flattening a list of lists.
    nested = bt.from_pydict({"n": [[[1, 2], [3]], [[4]]]})
    flat = nested.select(f=col("n").list.flatten()).to_pydict()
    print(flat)
    assert flat["f"] == [[1, 2, 3], [4]]


if __name__ == "__main__":
    main()

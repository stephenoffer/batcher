"""List columns: indexing, slicing, joining, and flattening.

A list column holds a variable-length array per row. Indexing and slicing stay columnar,
so ``.list.get(0)`` over a million rows is one operator rather than a million Python
subscripts. ``explode`` is the escape hatch when you want one row per element instead.

    python examples/expressions/lists_basics.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    carts = bt.from_pydict(
        {
            "user": ["a", "b", "c"],
            "items": [["apple", "pear", "fig"], ["milk"], []],
        }
    )

    shaped = carts.with_columns(
        n=col("items").list.len(),
        first=col("items").list.first(),
        last=col("items").list.last(),
        second=col("items").list.get(1),
        # `slice`/`head` take a window; both are safe on short or empty lists.
        top2=col("items").list.head(2),
        tail_window=col("items").list.slice(1),
        # Collapse a list to one string.
        joined=col("items").list.join(", "),
        # Membership and position (1-based; 0 when absent).
        has_fig=col("items").list.contains("fig"),
        fig_at=col("items").list.position("fig"),
    )

    result = shaped.to_pydict()
    print(result)

    assert result["n"] == [3, 1, 0]
    assert result["first"] == ["apple", "milk", None]
    assert result["last"] == ["fig", "milk", None]
    assert result["second"] == ["pear", None, None]
    assert result["top2"] == [["apple", "pear"], ["milk"], []]
    assert result["tail_window"] == [["pear", "fig"], [], []]
    assert result["joined"][0] == "apple, pear, fig"
    assert result["has_fig"] == [True, False, False]
    assert result["fig_at"][0] == 3

    # One row per element, which is how you build a fact table from a list column.
    # An empty list contributes no rows, so user "c" drops out entirely -- keep a copy of
    # the key side if you need those rows back.
    exploded = carts.explode("items").to_pydict()
    print(exploded)
    assert exploded["user"] == ["a", "a", "a", "b"]
    assert exploded["items"] == ["apple", "pear", "fig", "milk"]


if __name__ == "__main__":
    main()

"""JSON documents that hold arrays, and getting them into rows.

A JSON array inside a string column is two conversions away from being useful: extract it as
a list, then explode. Doing both in the engine keeps the parsing vectorized instead of
turning into a Python loop over documents.

    python examples/expr_collections/json_arrays.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from batcher import col


def main() -> None:
    orders = bt.from_pydict(
        {
            "id": [1, 2, 3],
            "payload": [
                '{"customer": "ada", "items": [{"sku": "a1", "qty": 2}, {"sku": "b2", "qty": 1}]}',
                '{"customer": "bob", "items": [{"sku": "c3", "qty": 5}]}',
                '{"customer": "cy", "items": []}',
            ],
        }
    )

    described = orders.select(
        "id",
        customer=col("payload").json.extract_string("$.customer"),
        item_count=col("payload").json.array_length("$.items"),
        first_sku=col("payload").json.extract_string("$.items[0].sku"),
        first_qty=col("payload").json.extract_int("$.items[0].qty"),
    )
    result = described.to_pydict()
    print(result)

    assert result["customer"] == ["ada", "bob", "cy"]
    assert result["item_count"] == [2, 1, 0]

    # The empty array has no element zero: null rather than an error.
    assert result["first_sku"][:2] == ["a1", "c3"]
    assert result["first_sku"][2] is None
    assert result["first_qty"][2] is None

    # The item count reconciles with what a full extraction would produce.
    assert sum(result["item_count"]) == 3

    # Documents with no items are the ones an inner-join-shaped explode would drop, which
    # is why the count is worth keeping before you flatten.
    with_items = described.filter(col("item_count") > 0)
    assert with_items.count() == 2


if __name__ == "__main__":
    main()

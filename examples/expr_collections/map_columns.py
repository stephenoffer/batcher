"""Map columns: key-value pairs in one column.

A map is the right shape for sparse attributes — a hundred possible keys of which each row
carries three. A struct will not do: it needs the key set fixed in the schema, and every row
then carries every field. Building one takes an explicit Arrow map type, because a dict in
`from_pydict` infers a struct.

    python examples/expr_collections/map_columns.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyarrow as pa

import batcher as bt
from batcher import col


def main() -> None:
    # An explicit map type: a dict literal would infer a struct instead.
    table = pa.table(
        {
            "id": pa.array([1, 2, 3]),
            "attributes": pa.array(
                [
                    [("source", "web"), ("region", "eu")],
                    [("source", "mobile")],
                    [("region", "us"), ("campaign", "spring")],
                ],
                type=pa.map_(pa.string(), pa.string()),
            ),
        }
    )
    events = bt.from_arrow(table)
    print(events.schema)

    described = events.select(
        "id",
        keys=col("attributes").map.keys(),
        values=col("attributes").map.values(),
        source=col("attributes").map.get("source"),
        has_region=col("attributes").map.contains("region"),
    )
    result = described.to_pydict()
    print(result)

    # A missing key gives null, not an error.
    assert result["source"] == ["web", "mobile", None]
    assert result["has_region"] == [True, False, True]

    # Keys and values line up per row.
    assert all(
        len(keys) == len(values)
        for keys, values in zip(result["keys"], result["values"], strict=True)
    )
    assert sorted(result["keys"][0]) == ["region", "source"]

    # Filtering on a map entry works like any other expression.
    european = events.filter(col("attributes").map.get("region") == "eu")
    assert european.count() == 1
    assert european.to_pydict()["id"] == [1]


if __name__ == "__main__":
    main()

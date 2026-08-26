"""Map columns: key-value pairs in one column.

A map is the right shape for sparse attributes — a hundred possible keys of which each row
carries three. A struct will not do: it needs the key set fixed in the schema, and every row
then carries every field.

There are two ways to get one. Reading it from Arrow needs an explicit map type, because a
dict in `from_pydict` infers a struct. Building one *inside a query* is `map_from_arrays`,
which pairs a column of key lists with a column of value lists — the constructor SQL spells
`map(keys, values)`.

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

    # Building a map inside the query, rather than reading one from Arrow. The two lists
    # are paired positionally, so they must be the same length in every row.
    built = bt.from_pydict(
        {
            "id": [1, 2],
            "k": [["source", "region"], ["source"]],
            "v": [["web", "eu"], ["mobile"]],
        }
    ).select("id", attributes=bt.map_from_arrays(col("k"), col("v")))
    # It is an ordinary map column: the read accessors work on it unchanged.
    round_tripped = built.select("id", source=col("attributes").map.get("source")).to_pydict()
    assert round_tripped["source"] == ["web", "mobile"]

    # Three inputs raise rather than guessing, matching DuckDB: a null key (Arrow map keys
    # are non-nullable), a duplicate key (keeping first or last would be a guess), and key
    # and value lists of different lengths (truncating would silently drop data).
    for bad_k, bad_v in (
        [[["a", "a"]], [[1, 2]]],  # duplicate key
        [[["a", "b"]], [[1]]],  # length mismatch
    ):
        frame = bt.from_pydict({"k": bad_k, "v": bad_v})
        try:
            frame.select(m=bt.map_from_arrays(col("k"), col("v"))).collect()
        except RuntimeError as err:
            print(f"refused, as DuckDB does: {err}")
        else:
            raise AssertionError("expected the malformed map to be refused")

    # Filtering on a map entry works like any other expression.
    european = events.filter(col("attributes").map.get("region") == "eu")
    assert european.count() == 1
    assert european.to_pydict()["id"] == [1]


if __name__ == "__main__":
    main()

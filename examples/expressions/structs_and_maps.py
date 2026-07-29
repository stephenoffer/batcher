"""Struct and map columns: nested records without flattening the table.

A struct column holds a fixed set of named fields per row; a map column holds
variable key/value pairs. Both are read with an accessor rather than by exploding the
table, so a nested field stays one projection away.

    python examples/expressions/structs_and_maps.py
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col


def main() -> None:
    # A struct column arrives as a dict per row with a consistent set of fields.
    people = bt.from_pydict(
        {
            "who": [
                {"first": "ada", "last": "lovelace"},
                {"first": "alan", "last": "turing"},
            ],
        }
    )

    flat = people.with_columns(
        first=col("who").struct.field("first"),
        last=col("who").struct.field("last"),
        # `get` is the same lookup under another name.
        first_again=col("who").struct.get("first"),
    )

    result = flat.to_pydict()
    print(result)

    assert result["first"] == ["ada", "alan"]
    assert result["last"] == ["lovelace", "turing"]
    assert result["first_again"] == result["first"]

    # Compose: a struct field feeds any other expression.
    initials = people.select(
        initials=col("who").struct.field("first").str.head(1).str.to_uppercase()
    ).to_pydict()
    assert initials["initials"] == ["A", "A"]

    # A map column: variable keys per row. Note that `from_pydict` on a column of dicts
    # infers a *struct* (one fixed field set), so build a real map through Arrow.
    settings = bt.from_arrow(
        pa.table(
            {
                "prefs": pa.array(
                    [[("theme", "dark"), ("lang", "en")], [("theme", "light")]],
                    type=pa.map_(pa.string(), pa.string()),
                )
            }
        )
    )
    read = settings.with_columns(
        keys=col("prefs").map.keys(),
        values=col("prefs").map.values(),
        n=col("prefs").map.len(),
        theme=col("prefs").map.get("theme"),
        has_lang=col("prefs").map.contains("lang"),
    ).to_pydict()
    print(read)

    assert sorted(read["keys"][0]) == ["lang", "theme"]
    assert read["n"] == [2, 1]
    assert read["theme"] == ["dark", "light"]
    assert read["has_lang"] == [True, False]


if __name__ == "__main__":
    main()

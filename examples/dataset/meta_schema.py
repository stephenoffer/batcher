"""Asking about a dataset's shape without executing it.

``ds.meta`` is the introspection accessor. Schema questions are answered from the plan, so
they cost nothing: you can branch on whether a column is numeric before deciding what
pipeline to build, without touching a row.

    python examples/dataset/meta_schema.py
"""

from __future__ import annotations

from datetime import datetime

import batcher as bt


def main() -> None:
    ds = bt.from_pydict(
        {
            "id": [1, 2, 3],
            "price": [1.5, 2.5, 3.5],
            "name": ["a", "b", "c"],
            "active": [True, False, True],
            "seen": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "tags": [["x"], ["y"], []],
        }
    )
    schema = ds.meta.schema

    # Presence and position.
    assert schema.has("price")
    assert not schema.has("missing")
    assert schema.num_columns() == 6
    assert schema.index("id") == 0
    print("dtype of price:", schema.dtype("price"))

    # Type predicates, one per family.
    assert schema.is_integer("id")
    assert schema.is_float("price")
    assert schema.is_numeric("price") and schema.is_numeric("id")
    assert schema.is_string("name")
    assert schema.is_boolean("active")
    assert schema.is_temporal("seen")
    assert schema.is_nested("tags")

    # The columns of each family, as lists.
    print("numeric:", schema.numeric())
    assert schema.numeric() == ["id", "price"]
    assert schema.strings() == ["name"]
    assert schema.booleans() == ["active"]
    assert schema.temporal() == ["seen"]
    assert schema.nested() == ["tags"]

    # `select(family)` returns a Dataset narrowed to that family -- useful when you want
    # to scale "every numeric column" without naming them.
    numeric_only = schema.select("numeric").to_pydict()
    assert sorted(numeric_only) == ["id", "price"]

    # Row/column shape, which does need to count rows.
    assert ds.meta.shape() == (3, 6)

    # The plan itself, as a dict you can assert on in a test.
    plan = ds.meta.explain()
    print("plan keys:", sorted(plan)[:5])
    assert isinstance(plan, dict)


if __name__ == "__main__":
    main()

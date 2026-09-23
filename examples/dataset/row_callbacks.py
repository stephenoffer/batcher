"""Row callbacks: map, flat_map, and filter with a Python function.

An expression runs in the engine, so reach for one first. When the logic is genuinely
Python (a parser, a lookup in a Python object, a model), these verbs call your function
inside the worker: ``map`` once per row, ``flat_map`` once per row returning any number of
rows, and ``filter`` once per *batch* returning a boolean mask. A class passed with
``fn_constructor_args`` is built once per worker instead of once per call.

    python examples/dataset/row_callbacks.py
"""

from __future__ import annotations

import pyarrow.compute as pc

import batcher as bt


class Tagger:
    """Built once per worker: the place for a model, a client, or a lookup table."""

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    def __call__(self, row: dict) -> dict:
        return {**row, "tag": f"{self.prefix}{row['sku']}"}


def main() -> None:
    orders = bt.from_pydict(
        {
            "order": [1, 2, 3, 4],
            "sku": ["a", "b", "a", "c"],
            "items": ["x,y", "z", "", "x"],
            "qty": [2, 1, 5, 3],
        }
    )

    # map: one dict in, one dict out. Return `{**row, ...}` to keep the input columns.
    doubled = orders.map(lambda r: {**r, "qty2": r["qty"] * 2})
    print("map:", doubled.to_pydict()["qty2"])
    assert doubled.to_pydict()["qty2"] == [4, 2, 10, 6]

    # When rows return different keys, every key becomes a column and a row without it
    # is null there, the rule map_batches applies across batches.
    ragged = orders.map(
        lambda r: (
            {"order": r["order"], "big": True}
            if r["qty"] > 2
            else {"order": r["order"], "small": True}
        )
    ).to_pydict()
    print("ragged:", ragged)
    assert ragged == {
        "order": [1, 2, 3, 4],
        "small": [True, True, None, None],
        "big": [None, None, True, True],
    }

    # flat_map: one row in, any number out. An empty list drops the row.
    exploded = orders.flat_map(
        lambda r: [{"order": r["order"], "item": i} for i in r["items"].split(",") if i]
    )
    print("flat_map:", exploded.to_pydict())
    assert exploded.to_pydict() == {"order": [1, 1, 2, 4], "item": ["x", "y", "z", "x"]}

    # filter(fn): the function sees a whole batch and returns one boolean per row, so
    # no Python runs per row. Every column keeps its exact type.
    big = orders.filter(lambda batch: pc.greater(batch["qty"], 2))
    assert big.to_pydict()["order"] == [3, 4]

    # A class is loaded once per worker; fn_constructor_args reach its __init__.
    tagged = orders.map(Tagger, fn_constructor_args=("sku-",))
    print("tagged:", tagged.to_pydict()["tag"])
    assert tagged.to_pydict()["tag"] == ["sku-a", "sku-b", "sku-a", "sku-c"]

    # Options are checked when the stage is defined, not in a worker halfway through.
    try:
        orders.map(Tagger, fn_constructor_args="sku-")
    except bt.PlanError as err:
        assert "fn_constructor_args must be a tuple" in str(err)
    else:
        raise AssertionError("a string is not a tuple of arguments")

    # A column a callback adds is unknown to the plan until you declare it, and the error
    # says how. `map` declares it through output_columns, as map_batches does.
    def add(row: dict) -> dict:
        return {**row, "qty2": row["qty"] * 2}

    try:
        orders.map(add).select(bt.col("qty2").sum())
    except bt.ColumnNotFoundError as err:
        assert "output_columns" in str(err)
    else:
        raise AssertionError("an undeclared output column must be reported")
    total = orders.map(add, output_columns=["order", "sku", "items", "qty", "qty2"])
    assert total.select(bt.col("qty2").sum()).to_pydict() == {"qty2": [22]}


if __name__ == "__main__":
    main()

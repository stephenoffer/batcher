"""Asserting the schema, not just the values.

A column that changes type upstream breaks everything downstream, and it does so quietly
whenever the new type still holds the old values. Checking names and types at the boundary
is a cheap assertion that catches an entire class of incident.

    python examples/quality/schema_contracts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch


def main() -> None:
    orders = tpch("orders")

    expected = {
        "o_orderkey": "int64",
        "o_custkey": "int64",
        "o_orderstatus": "string",
        "o_totalprice": "double",
        "o_orderdate": "date32[day]",
    }

    actual = dict(zip(orders.columns, [str(dtype) for dtype in orders.dtypes], strict=True))
    print({name: actual[name] for name in expected})

    # Every expected column is present, with the expected type.
    for name, dtype in expected.items():
        assert name in actual, f"missing column {name}"
        assert actual[name] == dtype, f"{name}: expected {dtype}, got {actual[name]}"

    # A widened type still holds the values but breaks the contract, which is the case a
    # value-only check misses entirely.
    widened = orders.astype({"o_orderkey": "float64"})
    widened_types = dict(
        zip(widened.columns, [str(dtype) for dtype in widened.dtypes], strict=True)
    )
    assert widened_types["o_orderkey"] != expected["o_orderkey"]
    assert widened.count() == orders.count()
    # The values compare equal, which is exactly why the type check is needed.
    assert widened.head(3).to_pydict()["o_orderkey"] == [
        float(value) for value in orders.head(3).to_pydict()["o_orderkey"]
    ]

    # A column added upstream is a different kind of change, and often a safe one.
    extended = orders.with_columns(o_channel=bt.lit("web"))
    assert set(expected) <= set(extended.columns)
    assert extended.count() == orders.count()

    # A column removed is never safe, and the check names it.
    reduced = orders.drop("o_orderstatus")
    missing = set(expected) - set(reduced.columns)
    print("missing after the drop:", missing)
    assert missing == {"o_orderstatus"}


if __name__ == "__main__":
    main()

"""Changing types: cast on an expression, astype on a frame.

Casts are where silent data loss lives. A float to int cast truncates, and a cast that
cannot represent its input produces null rather than raising. Both are useful and both
are worth asserting on rather than assuming.

    python examples/relational/casting_and_types.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem").select("l_orderkey", "l_quantity", "l_extendedprice", "l_shipdate")

    # Expression-level cast.
    as_float = lineitem.select(qty=col("l_quantity").cast("float64")).head(3).to_pydict()
    print(as_float)
    assert all(isinstance(value, float) for value in as_float["qty"])

    # Frame-level cast of several columns at once.
    retyped = lineitem.astype({"l_orderkey": "float64", "l_quantity": "float64"})
    types = dict(zip(retyped.columns, [str(dtype) for dtype in retyped.dtypes], strict=True))
    assert types["l_orderkey"] == "double"

    # Float to integer rounds to nearest — it does not truncate. That is the opposite
    # of C-style casting and of `int()` in Python, so it is worth pinning down: a
    # price of 13309.6 becomes 13310, not 13309.
    truncated = (
        lineitem.select(
            price=col("l_extendedprice"),
            whole=col("l_extendedprice").cast("int64"),
        )
        .head(5)
        .to_pydict()
    )
    print(truncated)
    assert all(
        whole == round(price)
        for price, whole in zip(truncated["price"], truncated["whole"], strict=True)
    )
    # To truncate instead, say so explicitly before the cast.
    floored = (
        lineitem.select(whole=col("l_extendedprice").floor().cast("int64")).head(5).to_pydict()
    )
    assert all(
        floor_value <= round_value
        for floor_value, round_value in zip(floored["whole"], truncated["whole"], strict=True)
    )

    # Dates cast to strings and back.
    text = lineitem.select(day=col("l_shipdate").cast("string")).head(3).to_pydict()
    print(text)
    assert all(len(value) == 10 for value in text["day"])


if __name__ == "__main__":
    main()

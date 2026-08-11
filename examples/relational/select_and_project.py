"""Choosing columns: `select` replaces the projection, `with_columns` extends it.

These are the two spellings people mix up. `select` decides the entire output shape, so
anything it does not name is gone. `with_columns` keeps everything and adds or replaces.
Reaching for `select` when you meant `with_columns` is how a column quietly disappears
three steps later.

    python examples/relational/select_and_project.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders")

    # `select` names the whole output. Nine columns in, three out.
    narrow = orders.select("o_orderkey", "o_custkey", "o_totalprice")
    print("select ->", narrow.columns)
    assert narrow.columns == ["o_orderkey", "o_custkey", "o_totalprice"]

    # Expressions can be derived inside `select`, with a keyword giving the name.
    derived = orders.select(
        "o_orderkey",
        price_in_thousands=col("o_totalprice") / 1000.0,
    )
    assert derived.columns == ["o_orderkey", "price_in_thousands"]

    # `with_columns` keeps every existing column and adds to it.
    widened = orders.with_columns(price_in_thousands=col("o_totalprice") / 1000.0)
    assert widened.columns == [*orders.columns, "price_in_thousands"]

    # Naming an existing column replaces it in place, keeping its position.
    replaced = orders.with_columns(o_totalprice=col("o_totalprice").round(0))
    assert replaced.columns == orders.columns

    # `alias` is the expression-level spelling of the same rename.
    aliased = orders.select(col("o_orderkey").alias("id")).head(3).to_pydict()
    print("aliased:", aliased)
    assert list(aliased) == ["id"]


if __name__ == "__main__":
    main()

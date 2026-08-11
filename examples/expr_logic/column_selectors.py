"""Choosing columns by type or by name pattern instead of listing them.

A selector resolves against the schema at plan time, so it works on a table whose column
list you do not know when you write the code. That is what makes a generic cleanup step
possible without reflection in Python.

    python examples/expr_logic/column_selectors.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch


def main() -> None:
    lineitem = tpch("lineitem")

    # By type.
    numbers = lineitem.select(bt.numeric())
    strings = lineitem.select(bt.string())
    dates = lineitem.select(bt.temporal())
    print("numeric:", numbers.columns)
    print("string:", strings.columns)
    print("temporal:", dates.columns)

    # The three families are disjoint and together cover the table.
    assert set(numbers.columns).isdisjoint(strings.columns)
    assert len(numbers.columns) + len(strings.columns) + len(dates.columns) == lineitem.width

    # By name.
    prefixed = lineitem.select(bt.matches(r"^l_ship"))
    print("l_ship*:", prefixed.columns)
    assert set(prefixed.columns) == {"l_shipdate", "l_shipinstruct", "l_shipmode"}

    # Everything except a few.
    trimmed = lineitem.select(bt.exclude("l_comment", "l_shipinstruct"))
    assert "l_comment" not in trimmed.columns
    assert trimmed.width == lineitem.width - 2

    # A selector composes with an expression, so "round every float" is one line.
    rounded = lineitem.select(bt.by_dtype("float64"))
    assert set(rounded.columns) == {"l_extendedprice", "l_discount", "l_tax"}


if __name__ == "__main__":
    main()

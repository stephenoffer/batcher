"""Bitwise and boolean folds over a column.

`bit_and` across a column tells you which bits every value shares — a fast way to check
an invariant like "every id is even". `bool_and`/`bool_or` are the same idea for
predicates, and they short-circuit the "does any row violate this" question into one pass.

    python examples/aggregations/bitwise_and_boolean.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem")

    folded = lineitem.agg(
        common_bits=bt.bit_and(col("l_linenumber")),
        any_bits=bt.bit_or(col("l_linenumber")),
        parity=bt.bit_xor(col("l_linenumber")),
        all_positive=bt.bool_and(col("l_quantity") > 0),
        any_free=bt.bool_or(col("l_extendedprice") == 0),
        all_discounted=bt.bool_and(col("l_discount") > 0),
    ).to_pydict()
    print(folded)

    # Line numbers start at 1 and go up to 7, so no bit is common to all of them, and
    # the union of their bits covers 1..7.
    assert folded["common_bits"][0] == 0
    assert folded["any_bits"][0] == 7

    # Data-quality checks as one-pass folds.
    assert folded["all_positive"][0] is True
    assert folded["any_free"][0] is False
    # Not every line is discounted, so the universal claim is false.
    assert folded["all_discounted"][0] is False

    # `bool_and` over a predicate is the same as "no row fails it".
    violations = lineitem.filter(col("l_quantity") <= 0).count()
    assert (violations == 0) == folded["all_positive"][0]


if __name__ == "__main__":
    main()

"""Which predicate shapes can be pushed into a scan, and which cannot.

A predicate on a raw column can be evaluated by the reader; one on a derived column cannot,
because the derivation happens after the read. Rewriting the predicate to reference the raw
column is often the whole optimization.

    python examples/expr_logic/predicate_pushdown_shapes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch_path
from batcher import col


def main() -> None:
    path = tpch_path("lineitem")

    # Pushable: a comparison on a stored column.
    pushable = bt.read.parquet(path).filter(col("l_quantity") > 40)

    # Not pushable as written: the predicate is on a derived column, so the derivation has
    # to happen first.
    derived = (
        bt.read.parquet(path)
        .with_columns(net=col("l_extendedprice") * (1 - col("l_discount")))
        .filter(col("net") > 50_000)
    )

    # The same question, rewritten to reference stored columns only. Not always possible,
    # but when it is, it is the difference between reading a column and reading all of it.
    equivalent = bt.read.parquet(path).filter(
        col("l_extendedprice") * (1 - col("l_discount")) > 50_000
    )

    print("pushable:", pushable.count())
    print("derived:", derived.count())
    print("rewritten:", equivalent.count())

    # The two spellings of the derived predicate agree.
    assert derived.count() == equivalent.count()

    # And both are selective.
    total = bt.read.parquet(path).count()
    assert 0 < derived.count() < total
    assert 0 < pushable.count() < total

    for name, query in (
        ("pushable", pushable),
        ("derived", derived),
        ("rewritten", equivalent),
    ):
        plan = query.explain()
        assert "filter" in plan.lower(), name
        assert "scan" in plan.lower(), name

    print(pushable.explain())

    # A predicate the reader can never help with: one that has to look at the string.
    textual = bt.read.parquet(path).filter(col("l_comment").str.contains("final"))
    print("substring match:", textual.count())
    assert 0 < textual.count() < total


if __name__ == "__main__":
    main()

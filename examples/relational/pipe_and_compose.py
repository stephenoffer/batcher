"""Composing pipelines: `pipe` for reuse, and why laziness makes it free.

A Dataset is a plan, not data, so factoring a pipeline into functions costs nothing at
runtime — the pieces fuse into one plan before anything executes. That is what makes
`pipe` a real structuring tool here rather than a stylistic flourish.

    python examples/relational/pipe_and_compose.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def only_shipped_by(dataset: bt.Dataset, mode: str) -> bt.Dataset:
    """One reusable step, written against Dataset in and Dataset out."""
    return dataset.filter(col("l_shipmode") == mode)


def with_revenue(dataset: bt.Dataset) -> bt.Dataset:
    """Another, deriving the column every downstream step wants."""
    return dataset.with_columns(revenue=col("l_extendedprice") * (1 - col("l_discount")))


def main() -> None:
    lineitem = tpch("lineitem")

    piped = (
        lineitem.pipe(only_shipped_by, "AIR")
        .pipe(with_revenue)
        .group_by("l_returnflag")
        .agg(revenue=col("revenue").sum())
        .sort("l_returnflag")
    )

    # Composition is not execution: nothing has run yet.
    plan = piped.explain()
    print(plan)
    assert "aggregate" in plan.lower()

    result = piped.to_pydict()
    print(result)

    # The hand-written equivalent gives the same answer.
    direct = (
        lineitem.filter(col("l_shipmode") == "AIR")
        .with_columns(revenue=col("l_extendedprice") * (1 - col("l_discount")))
        .group_by("l_returnflag")
        .agg(revenue=col("revenue").sum())
        .sort("l_returnflag")
        .to_pydict()
    )
    assert result == direct


if __name__ == "__main__":
    main()

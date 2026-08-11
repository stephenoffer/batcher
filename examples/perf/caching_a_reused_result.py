"""Caching an intermediate that several branches read.

A Dataset is a plan, so referencing it twice computes it twice. `cache` breaks that: the
result is materialized once and both branches read the materialized copy. It is only worth
it when the intermediate is genuinely reused and genuinely expensive.

    python examples/perf/caching_a_reused_result.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem")

    # An expensive intermediate: a grouped aggregate over the whole fact table.
    per_order = lineitem.group_by("l_orderkey").agg(
        revenue=(col("l_extendedprice") * (1 - col("l_discount"))).sum(),
        lines=bt.count(),
    )

    cached = per_order.cache()

    # Two branches over the same intermediate.
    started = time.perf_counter()
    big = cached.filter(col("revenue") > 50_000).count()
    many = cached.filter(col("lines") >= 5).count()
    elapsed = time.perf_counter() - started

    print(f"{big} large orders, {many} many-line orders in {elapsed:.2f}s")
    assert big > 0
    assert many > 0

    # Caching changes cost, never the answer.
    assert big == per_order.filter(col("revenue") > 50_000).count()
    assert many == per_order.filter(col("lines") >= 5).count()

    # `persist` is the explicit form for keeping a result across a longer stretch of work.
    kept = per_order.persist()
    assert kept.count() == per_order.count()


if __name__ == "__main__":
    main()

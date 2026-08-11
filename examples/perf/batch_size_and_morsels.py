"""Morsel size: the scheduling knob that must never change an answer.

Work is divided into morsels so scheduling is granular and cache-friendly. The size is a
performance choice. Sweeping it and asserting the result is unchanged is the cheapest test
that an operator is not accidentally order- or batch-dependent.

    python examples/perf/batch_size_and_morsels.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col
from batcher.config import get_option, option_context


def main() -> None:
    lineitem = tpch("lineitem")

    query = (
        lineitem.group_by("l_returnflag", "l_linestatus")
        .agg(
            lines=bt.count(),
            qty=col("l_quantity").sum(),
            biggest=col("l_extendedprice").max(),
        )
        .sort("l_returnflag", "l_linestatus")
    )

    default = get_option("execution.morsel_rows")
    baseline = query.to_pydict()
    print(f"default morsel_rows={default}: {baseline['lines']}")

    for size in (1024, 4096, 65_536):
        with option_context("execution.morsel_rows", size):
            result = query.to_pydict()
        assert result["l_returnflag"] == baseline["l_returnflag"], size
        assert result["lines"] == baseline["lines"], size
        assert result["qty"] == baseline["qty"], size
        assert result["biggest"] == baseline["biggest"], size
    print("identical at 1,024, 4,096 and 65,536 rows per morsel")

    # A sorted top-N is the order-sensitive case, so check it position by position.
    top = lineitem.sort("l_extendedprice", descending=True).head(10)
    reference = top.to_pydict()["l_extendedprice"]
    with option_context("execution.morsel_rows", 1024):
        assert top.to_pydict()["l_extendedprice"] == reference


if __name__ == "__main__":
    main()

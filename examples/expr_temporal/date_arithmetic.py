"""Shifting a date by an interval.

`offset_by` takes a duration string, so "3 months from now" stays correct across
month lengths — adding 90 days does not. `date_add` and `date_sub` are the plain
day-count forms for when that is genuinely what you mean.

    python examples/expr_temporal/date_arithmetic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders").select("o_orderkey", "o_orderdate").head(500)

    shifted = orders.select(
        "o_orderdate",
        next_month=col("o_orderdate").dt.offset_by("1mo"),
        next_quarter=col("o_orderdate").dt.offset_by("3mo"),
        last_week=col("o_orderdate").dt.offset_by("-1w"),
        plus_ninety=bt.date_add(col("o_orderdate"), 90),
        minus_thirty=bt.date_sub(col("o_orderdate"), 30),
    )

    result = shifted.head(3).to_pydict()
    print(result)

    full = shifted.to_pydict()

    # Direction is respected.
    assert all(
        later > original
        for original, later in zip(full["o_orderdate"], full["next_month"], strict=True)
    )
    assert all(
        earlier < original
        for original, earlier in zip(full["o_orderdate"], full["last_week"], strict=True)
    )

    # A one-week shift really is seven days.
    assert all(
        (original - earlier).days == 7
        for original, earlier in zip(full["o_orderdate"], full["last_week"], strict=True)
    )

    # A calendar month is not a fixed number of days, which is the whole point.
    month_gaps = {
        (later - original).days
        for original, later in zip(full["o_orderdate"], full["next_month"], strict=True)
    }
    print("distinct day-gaps for +1 month:", sorted(month_gaps))
    assert len(month_gaps) > 1

    # Whereas date_add is exactly the number of days you asked for.
    assert all(
        (later - original).days == 90
        for original, later in zip(full["o_orderdate"], full["plus_ninety"], strict=True)
    )


if __name__ == "__main__":
    main()

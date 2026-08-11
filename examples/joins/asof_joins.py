"""As-of joins: matching the most recent row at or before a timestamp.

This is the join for "what was the rate when this transaction happened". An equality join
cannot express it, and a range join plus a max is the slow way round. Both sides have to be
sorted on the join column, which is the constraint that makes it linear.

    python examples/joins/asof_joins.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    # Real order dates as the event stream.
    events = (
        tpch("orders")
        .select("o_orderkey", "o_orderdate", "o_totalprice")
        .sort("o_orderdate")
        .head(4_000)
    )

    # A rate table with far fewer rows than the event stream.
    dates = events.select("o_orderdate").distinct().sort("o_orderdate").to_pydict()["o_orderdate"]
    rate_days = dates[::10]
    rates = bt.from_pydict(
        {
            "as_of": rate_days,
            # Keep a second copy of the effective date: `right_on` is consumed by the
            # join, so without this there is no way to check which rate was matched.
            "rate_effective": rate_days,
            "rate": [1.0 + index * 0.01 for index in range(len(rate_days))],
        }
    ).sort("as_of")
    print(f"{events.count()} events, {rates.count()} rate changes")

    joined = events.join_asof(rates, left_on="o_orderdate", right_on="as_of").sort("o_orderdate")
    result = joined.to_pydict()
    print(result["o_orderdate"][:3], result["rate"][:3])

    # Every event keeps its row: an as-of join is a lookup, not a filter.
    assert joined.count() == events.count()

    # The matched rate is never from the future.
    matched = joined.filter(col("rate").is_not_null()).to_pydict()
    assert all(
        effective <= day
        for day, effective in zip(matched["o_orderdate"], matched["rate_effective"], strict=True)
    )

    # Events before the first rate change have no match, which is correct rather than an
    # error — and is why the null check above is not decoration.
    unmatched = joined.filter(col("rate").is_null()).count()
    print("events before the first rate:", unmatched)
    assert unmatched + len(matched["o_orderdate"]) == events.count()


if __name__ == "__main__":
    main()

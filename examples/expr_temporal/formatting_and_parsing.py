"""Dates to text and back.

`strftime` formats and `str.to_date` parses. Both are the boundary with systems that have
no date type, and both are where a format mismatch turns into a null rather than an error
— so check the null count after parsing, every time.

    python examples/expr_temporal/formatting_and_parsing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders").select("o_orderdate").head(500)

    formatted = orders.select(
        "o_orderdate",
        iso=col("o_orderdate").dt.strftime("%Y-%m-%d"),
        pretty=col("o_orderdate").dt.strftime("%d %B %Y"),
        compact=col("o_orderdate").dt.strftime("%Y%m"),
        as_text=col("o_orderdate").dt.to_string(),
    )
    sample = formatted.head(3).to_pydict()
    print(sample)

    assert all(len(value) == 10 for value in sample["iso"])
    assert all(len(value) == 6 for value in sample["compact"])

    # Parsing the ISO form back gives the original date.
    round_trip = formatted.select(
        "o_orderdate", parsed=col("iso").str.to_date("%Y-%m-%d")
    ).to_pydict()
    assert round_trip["parsed"] == round_trip["o_orderdate"]

    # A format that does not match the text produces nulls rather than raising, so the
    # only way to notice is to count them.
    mismatched = formatted.select(parsed=col("iso").str.to_date("%d/%m/%Y"))
    # Count the nulls with `count_if`: `sum` has no meaning on a boolean column and
    # the engine says so rather than coercing it.
    nulls = mismatched.agg(bad=bt.count_if(col("parsed").is_null())).to_pydict()["bad"][0]
    print("rows that failed to parse:", nulls)
    assert nulls == formatted.count()

    # The compact form groups by month with no date type involved, which is how this ends
    # up in a warehouse partition key.
    months = formatted.n_unique("compact")
    print("distinct months:", months)
    assert 0 < months <= 12 * 8


if __name__ == "__main__":
    main()

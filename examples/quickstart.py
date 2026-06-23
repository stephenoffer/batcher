"""Quickstart: build a lazy pipeline and run it.

The mirror of the README's headline example, but over in-memory data so it runs
anywhere with no cloud bucket. A Dataset is lazy and immutable: each step returns a
new Dataset and nothing executes until a terminal op (here ``to_pydict``).

Run it directly::

    python examples/quickstart.py
"""

from __future__ import annotations

import batcher as bt


def main() -> None:
    events = bt.from_pydict(
        {
            "region": ["us", "eu", "us", "eu", "us"],
            "status": ["active", "active", "churned", "active", "active"],
            "amount": [10.0, 5.0, 99.0, 7.0, 3.0],
        }
    )

    # Nothing runs while the pipeline is built — only the final to_pydict() executes.
    revenue = (
        events.filter(bt.col("status") == "active")
        .group_by("region")
        .agg(total=bt.col("amount").sum())
        .sort("total", descending=True)
    )

    result = revenue.to_pydict()
    print(result)

    # eu: 5 + 7 = 12, us: 10 + 3 = 13 (the churned 99 is filtered out).
    assert result["region"] == ["us", "eu"]
    assert result["total"] == [13.0, 12.0]


if __name__ == "__main__":
    main()

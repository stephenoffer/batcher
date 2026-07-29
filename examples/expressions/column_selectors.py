"""Selectors: naming columns by type or pattern instead of one at a time.

A selector is an ``Expr`` leaf standing for *every* matching column, so "round every float"
is one expression that keeps working when a column is added. Spelling out names is how a
pipeline silently stops covering a new column. Combine selectors with ``|``, ``&``, ``-``,
and ``~``.

    python examples/expressions/column_selectors.py
"""

from __future__ import annotations

from datetime import datetime

import batcher as bt


def main() -> None:
    wide = bt.from_pydict(
        {
            "id": [1, 2],
            "amount_usd": [10.5, 20.5],
            "amount_eur": [9.25, 18.75],
            "name": ["a", "b"],
            "active": [True, False],
            "seen": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
        }
    )

    # By type family.
    numeric = wide.select(bt.numeric()).to_pydict()
    print("numeric:", sorted(numeric))
    assert sorted(numeric) == ["amount_eur", "amount_usd", "id"]

    assert sorted(wide.select(bt.string()).to_pydict()) == ["name"]
    assert sorted(wide.select(bt.boolean()).to_pydict()) == ["active"]
    assert sorted(wide.select(bt.temporal()).to_pydict()) == ["seen"]
    assert sorted(wide.select(bt.integer()).to_pydict()) == ["id"]
    assert sorted(wide.select(bt.floating()).to_pydict()) == ["amount_eur", "amount_usd"]

    # By name.
    assert sorted(wide.select(bt.starts_with("amount_")).to_pydict()) == [
        "amount_eur",
        "amount_usd",
    ]
    assert sorted(wide.select(bt.ends_with("_usd")).to_pydict()) == ["amount_usd"]
    assert sorted(wide.select(bt.contains("mount")).to_pydict()) == [
        "amount_eur",
        "amount_usd",
    ]
    assert sorted(wide.select(bt.matches(r"^amount_[a-z]{3}$")).to_pydict()) == [
        "amount_eur",
        "amount_usd",
    ]

    # Everything, or everything except.
    assert len(wide.select(bt.all()).to_pydict()) == 6
    kept = wide.select(bt.exclude("name", "seen")).to_pydict()
    print("excluded:", sorted(kept))
    assert "name" not in kept and "seen" not in kept

    # Set algebra over selectors: floats that are not the EUR column.
    combined = wide.select(bt.floating() - bt.ends_with("_eur")).to_pydict()
    print("floating minus eur:", sorted(combined))
    assert sorted(combined) == ["amount_usd"]

    union = wide.select(bt.integer() | bt.boolean()).to_pydict()
    assert sorted(union) == ["active", "id"]

    # The payoff: compute over every matched column at once, in place.
    rounded = wide.with_columns(bt.floating().round(0)).to_pydict()
    print("rounded:", rounded["amount_usd"], rounded["amount_eur"])
    # `.round` breaks ties away from zero, so 10.5 -> 11 rather than 10.
    assert rounded["amount_usd"] == [11.0, 21.0]
    assert rounded["amount_eur"] == [9.0, 19.0]
    # Non-matching columns are untouched.
    assert rounded["name"] == ["a", "b"]


if __name__ == "__main__":
    main()

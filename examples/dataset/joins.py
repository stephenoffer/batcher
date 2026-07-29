"""Join types, key spellings, and the as-of join for time series.

The join type decides what happens to rows with no match, which is where most join bugs
live. An inner join silently drops them; a left join keeps them with nulls. Decide which
you meant before you write it, then assert the row count.

    python examples/dataset/joins.py
"""

from __future__ import annotations

import batcher as bt


def main() -> None:
    orders = bt.from_pydict({"id": [1, 2, 3], "cid": [10, 20, 99], "total": [5, 15, 25]})
    customers = bt.from_pydict({"cid": [10, 20, 30], "name": ["ada", "bob", "cy"]})

    # Inner: only matching rows survive. Order 3 (cid 99) disappears.
    inner = orders.join(customers, on="cid").sort("id").to_pydict()
    print("inner:", inner)
    assert inner["id"] == [1, 2]

    # Left: every left row survives; unmatched right columns are null.
    left = orders.join(customers, on="cid", how="left").sort("id").to_pydict()
    print("left:", left)
    assert left["id"] == [1, 2, 3]
    assert left["name"][2] is None

    # Right and full outer.
    right = orders.join(customers, on="cid", how="right").to_pydict()
    assert len(right["cid"]) == 3
    outer = orders.join(customers, on="cid", how="outer").to_pydict()
    assert len(outer["cid"]) == 4  # 10, 20, 30, 99

    # Semi keeps left rows that have a match, without adding right columns.
    semi = orders.join(customers, on="cid", how="semi").sort("id").to_pydict()
    print("semi:", semi)
    assert semi["id"] == [1, 2]
    assert "name" not in semi

    # Anti is the complement: left rows with no match. This is the orphan check.
    anti = orders.join(customers, on="cid", how="anti").to_pydict()
    print("orphans:", anti)
    assert anti["id"] == [3]

    # Differently named keys.
    renamed = bt.from_pydict({"customer": [10, 20], "tier": ["gold", "silver"]})
    keyed = orders.join(renamed, left_on="cid", right_on="customer").sort("id").to_pydict()
    assert keyed["tier"] == ["gold", "silver"]

    # Cross join: every pair. Guard the size before you run one.
    small = bt.from_pydict({"a": [1, 2]})
    other = bt.from_pydict({"b": ["x", "y"]})
    cross = small.cross_join(other).to_pydict()
    assert len(cross["a"]) == 4

    # As-of join: match the most recent row at or before each timestamp. This is the
    # right tool for "what was the price when this trade happened".
    trades = bt.from_pydict({"t": [10, 20, 30], "sym": ["a", "a", "a"], "qty": [1, 2, 3]})
    quotes = bt.from_pydict({"t": [5, 15, 25], "sym": ["a", "a", "a"], "px": [100, 110, 120]})
    asof = trades.join_asof(quotes, on="t", by="sym").sort("t").to_pydict()
    print("asof:", asof)
    assert asof["px"] == [100, 110, 120]

    # The guard worth writing: an inner join that drops rows is usually a bug.
    assert orders.count() == 3
    assert orders.join(customers, on="cid").count() == 2
    dropped = orders.count() - orders.join(customers, on="cid").count()
    print("rows dropped by the inner join:", dropped)
    assert dropped == 1


if __name__ == "__main__":
    main()

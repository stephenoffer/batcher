"""Cheap yes/no questions about the data, and the column-check shorthands.

These short-circuit. ``any_match`` stops at the first matching row rather than counting
them all, which makes "does this table contain any bad rows?" much cheaper than
"how many bad rows does it contain?".

    python examples/dataset/meta_predicates.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    ds = bt.from_pydict(
        {
            "id": [1, 2, 3, 4, 5],
            "amount": [10.0, 20.0, 30.0, 40.0, 50.0],
            "status": ["ok", "ok", "ok", "ok", "bad"],
        }
    )

    # Existence questions, answered without a full count.
    assert ds.meta.any_match(col("status") == "bad")
    assert ds.meta.none_match(col("amount") < 0)
    assert ds.meta.all_match(col("amount") > 0)
    assert not ds.meta.is_empty_where(col("status") == "bad")
    assert ds.meta.is_empty_where(col("status") == "missing")

    # When you do want the number.
    assert ds.meta.count_where(col("status") == "ok") == 4

    # The `check` namespace is the same idea, scoped to one column and read as a
    # sentence: "all amounts are between 1 and 100".
    check = ds.meta.col("amount").check
    assert check.all_positive()
    assert check.all_greater_than(5.0)
    assert check.all_greater_equal(10.0)
    assert check.all_less_than(100.0)
    assert check.all_less_equal(50.0)
    assert check.all_between(10.0, 50.0)
    assert not check.all_greater_than(10.0)  # 10.0 is not > 10.0

    # Sort knowledge is tracked on the plan, so a second sort can be skipped.
    print("sorted by:", ds.meta.sorted_by())
    ordered = ds.sort("amount")
    print("after sort:", ordered.meta.sorted_by())
    assert ordered.meta.is_known_sorted_by("amount")

    # The gate this exists for: refuse to proceed when a contract is violated.
    def assert_clean(dataset: bt.Dataset) -> None:
        if dataset.meta.any_match(col("amount") <= 0):
            raise ValueError("non-positive amount found")

    assert_clean(ds)
    try:
        assert_clean(bt.from_pydict({"amount": [1.0, -5.0]}))
    except ValueError as exc:
        print("caught:", exc)
    else:
        raise AssertionError("expected the guard to fire")


if __name__ == "__main__":
    main()

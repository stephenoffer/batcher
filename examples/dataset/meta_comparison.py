"""Asking about a join before running it, and reading approximate statistics.

``ds.meta.against(other)`` answers the question that saves the most time in practice: will
this join produce anything at all? A join that silently returns zero rows because the keys
never overlap is one of the most common quiet failures in a pipeline.

    python examples/dataset/meta_comparison.py
"""

from __future__ import annotations

import batcher as bt


def main() -> None:
    orders = bt.from_pydict({"customer_id": [1, 2, 3, 4], "total": [10, 20, 30, 40]})
    customers = bt.from_pydict({"id": [3, 4, 5, 6], "name": ["c", "d", "e", "f"]})
    strangers = bt.from_pydict({"id": [90, 91], "name": ["x", "y"]})

    # Do the key ranges overlap at all?
    pair = orders.meta.against(customers)
    assert pair.overlaps("customer_id", "id")
    assert not pair.join_is_empty("customer_id", "id")
    print("key overlap:", pair.key_overlap("customer_id", "id"))
    print("estimated rows:", pair.estimated_rows("customer_id", "id"))

    # A join with no shared keys is detectable before you pay for it.
    empty = orders.meta.against(strangers)
    assert empty.join_is_empty("customer_id", "id")
    assert not empty.overlaps("customer_id", "id")

    # The guard this exists for.
    def join_or_explain(left: bt.Dataset, right: bt.Dataset) -> bt.Dataset:
        if left.meta.against(right).join_is_empty("customer_id", "id"):
            raise ValueError("join keys do not overlap -- check the id space")
        return left.join(right, left_on="customer_id", right_on="id")

    joined = join_or_explain(orders, customers).to_pydict()
    print(joined)
    assert sorted(joined["name"]) == ["c", "d"]

    try:
        join_or_explain(orders, strangers)
    except ValueError as exc:
        print("caught:", exc)
    else:
        raise AssertionError("expected the guard to fire")

    # Approximate statistics, backed by sketches rather than exact passes. They are
    # available only where the source carries them, so every accessor returns `None`
    # rather than guessing -- check `is_measured` before relying on one.
    big = bt.from_pydict({"user": [f"u{i % 50}" for i in range(1000)]})
    approx = big.meta.approx
    print("rows:", approx.rows(), "measured:", approx.is_measured("user"))
    assert approx.rows() == 1000.0

    est = approx.n_unique("user")
    print("approx distinct users:", est)
    if approx.is_measured("user"):
        assert est is not None and 40 <= est <= 60
    else:
        # An in-memory dataset carries no column sketch, so this is `None` by design.
        assert est is None

    # The exact answer is always available, it just costs a pass.
    exact = big.meta.col("user").n_unique()
    print("exact distinct users:", exact)
    assert exact == 50

    # Storage facts: how many files, how big, and is it partitioned?
    storage = big.meta.storage
    print("sources:", storage.num_sources(), "partitioned:", storage.is_partitioned())
    assert storage.num_sources() >= 1
    assert storage.is_partitioned() in (True, False)


if __name__ == "__main__":
    main()

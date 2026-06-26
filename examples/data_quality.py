"""Data-quality checks: validate, quarantine, drop, and enforce a contract.

Checks live on the ``ds.dq`` accessor — a chain of expectations (each a boolean
expression that is TRUE for a valid row) followed by a terminal action:
``validate`` reports counts, ``drop`` keeps only clean rows, ``quarantine`` splits
clean from rejected, and ``fail`` raises a `DataQualityError` at a contract boundary.

Run it directly::

    python examples/data_quality.py
"""

from __future__ import annotations

import batcher as bt
from batcher._internal.errors import DataQualityError


def main() -> None:
    people = bt.from_pydict(
        {
            "id": [1, 2, 3, 4, 5],
            "email": ["a@x.io", "b@x.io", None, "d@x.io", "e@x.io"],
            "age": [34, 28, 51, 200, 40],
            "country": ["US", "CA", "US", "ZZ", "CA"],
        }
    )

    # A report of per-constraint violation counts — no raise.
    report = (
        people.dq.not_null("email")
        .in_range("age", 0, 120)
        .accepted_values("country", ["US", "CA", "MX"])
        .validate()
    )
    print(report)
    assert not report.ok
    assert report.total_violations == 3  # one null email, one bad age, one bad country

    # Split clean rows from rejected ones for a dead-letter sink.
    clean, rejected = people.dq.in_range("age", 0, 120).quarantine()
    assert clean.count() == 4
    assert rejected.to_pydict()["id"] == [4]

    # Drop keeps only rows that satisfy every constraint.
    kept = people.dq.in_range("age", 0, 120).not_null("email").drop()
    assert kept.sort("id").to_pydict()["id"] == [1, 2, 5]

    # fail() is the contract gate: it raises when a constraint is violated.
    raised = False
    try:
        people.dq.in_range("age", 0, 120).fail()
    except DataQualityError:
        raised = True
    assert raised

    # Referential integrity: orphan rows whose key is absent from the reference.
    orders = bt.from_pydict({"order_id": [1, 2, 3], "customer_id": [10, 20, 99]})
    customers = bt.from_pydict({"customer_id": [10, 20, 30]})
    orphans = orders.dq.foreign_key("customer_id", references=customers)
    assert orphans.to_pydict()["customer_id"] == [99]

    # Deduplicate to the latest record per key.
    events = bt.from_pydict(
        {"user": ["a", "a", "b", "b"], "ts": [1, 2, 1, 2], "val": [10, 11, 20, 21]}
    )
    latest = events.distinct(subset=["user"], keep="last", order_by="ts").sort("user")
    assert latest.to_pydict()["val"] == [11, 21]


if __name__ == "__main__":
    main()

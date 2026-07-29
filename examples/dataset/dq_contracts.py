"""Data-quality contracts: validate, fail, drop, or quarantine.

The four terminal calls are the whole design. ``validate()`` reports without changing the
data, ``fail()`` raises, ``drop()`` silently removes bad rows, and ``quarantine()`` splits
them out so you can inspect them. Choosing between them is a decision about who is
responsible for the bad rows.

    python examples/dataset/dq_contracts.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    orders = bt.from_pydict(
        {
            "id": [1, 2, 3, 4, 5],
            "email": ["a@x.com", "b@x.com", "nope", "d@x.com", "e@x.com"],
            "amount": [10, 20, -5, 40, 50],
            "status": ["new", "paid", "paid", "weird", "new"],
        }
    )

    def contract(ds: bt.Dataset) -> bt.DatasetDQ:
        return (
            ds.dq.not_null("id", "email")
            .unique("id")
            .in_range("amount", 0, 1000)
            .matches("email", r"^[^@]+@[^@]+\.[^@]+$")
            .accepted_values("status", ["new", "paid", "refunded"])
            .check(col("amount") != 0, name="amount_nonzero")
        )

    # 1. Report, without changing anything.
    report = contract(orders).validate()
    print("ok:", report.ok, "violations:", report.total_violations)
    assert not report.ok
    assert report.total_violations >= 2  # the bad email and the negative amount

    # 2. Drop the offending rows.
    clean = contract(orders).drop().to_pydict()
    print("kept:", clean["id"])
    assert clean["id"] == [1, 2, 5]

    # 3. Split them out, so nothing is thrown away silently.
    good, bad = contract(orders).quarantine()
    g, b = good.to_pydict(), bad.to_pydict()
    print("good:", g["id"], "quarantined:", b["id"])
    assert g["id"] == [1, 2, 5]
    assert sorted(b["id"]) == [3, 4]

    # 4. Refuse to proceed at all.
    try:
        contract(orders).fail().to_pydict()
    except Exception as exc:
        print("fail() raised:", type(exc).__name__)
    else:
        raise AssertionError("expected fail() to raise")

    # A clean dataset passes every one of them.
    spotless = bt.from_pydict({"id": [1], "email": ["a@x.com"], "amount": [10], "status": ["paid"]})
    assert contract(spotless).validate().ok
    assert contract(spotless).drop().count() == 1

    # Referential integrity against another dataset.
    customers = bt.from_pydict({"cid": [1, 2]})
    facts = bt.from_pydict({"cid": [1, 2, 99]})
    checked = facts.dq.foreign_key("cid", references=customers, ref_columns="cid")
    print("fk-checked rows:", checked.count())
    assert checked.count() >= 1


if __name__ == "__main__":
    main()

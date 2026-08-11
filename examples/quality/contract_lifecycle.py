"""The life of a data contract: watch a rule, tolerate it, then enforce it.

A rule that fails the run the first time it is written gets deleted. The way one survives is
to arrive as a warning, earn a tolerance, and only then become a gate — and to say, per row,
which rule rejected it. This runs the whole arc against one small in-memory table.

    python examples/quality/contract_lifecycle.py
"""

from __future__ import annotations

import datetime as dt

import batcher as bt
from batcher import col
from batcher._internal.errors import DataQualityError


def orders() -> bt.Dataset:
    """Five orders, two of which are wrong in different ways."""
    now = dt.datetime.now()
    return bt.from_pydict(
        {
            "order_id": [1, 2, 3, 4, 5],
            "customer_id": [10, 20, 10, 30, None],
            "amount": [12.5, 40.0, -7.25, 99.0, 15.0],
            "email": ["a@x.io", "b@x.io", "not-an-address", "d@x.io", "e@x.io"],
            "placed_at": [now - dt.timedelta(minutes=m) for m in (1, 2, 3, 4, 5)],
        }
    )


def main() -> None:
    ds = orders()

    # 1. The schema contract comes first: a missing column would break every value check
    #    written against it, and the report would then name the symptoms, not the cause.
    schema_gate = ds.dq.has_columns("order_id", "amount", "email").column_types(
        {"order_id": "int64", "amount": "float64"}
    )
    assert schema_gate.validate().ok

    # 2. A new rule arrives as a warning. It is measured on every run and fails none.
    watched = ds.dq.positive("amount", severity="warn").matches_format("email", "email")
    report = watched.validate()
    print("warnings:", [r.name for r in report.warnings])
    assert report.violations["positive(amount)"] == 1
    assert not report.ok  # the email rule is enforced, and one address is malformed
    assert [r.name for r in report.failed] == ["is_email(email)"]

    # A warning removes no row.
    assert watched.drop().count() == ds.count() - 1  # only the email rule filtered

    # 3. Tolerance: 80% of amounts are positive, and 80% is the agreed bar for now.
    tolerated = ds.dq.positive("amount", mostly=0.8)
    assert tolerated.validate().ok
    assert tolerated.validate().violations["positive(amount)"] == 1  # still counted
    assert tolerated.drop().count() == 4  # and still dropped
    print("tolerated pass rate:", tolerated.validate().result("positive(amount)").pass_rate)

    # 4. Enforcement: the same rule with no tolerance is a gate.
    try:
        ds.dq.positive("amount").fail()
        raise AssertionError("the gate should have raised")
    except DataQualityError as err:
        print("gate raised:", str(err).split(".")[0])

    # 5. The rejected rows travel with their reason, which is what a dead-letter sink needs.
    labelled = ds.dq.positive("amount").matches_format("email", "email").annotate()
    reasons = labelled.filter(col("dq_failed") != "").select("order_id", "dq_failed").to_pydict()
    print("rejected:", dict(zip(reasons["order_id"], reasons["dq_failed"], strict=True)))
    assert set(reasons["order_id"]) == {3}
    assert reasons["dq_failed"] == ["positive(amount),is_email(email)"]

    # 6. Relation-level checks catch what no row is responsible for: volume, distribution,
    #    missingness, and staleness. They cannot drop a row, so they are validated, not split.
    table_contract = (
        ds.dq.row_count_between(1, 1_000_000)
        .mean_between("amount", 1.0, 500.0)
        .null_rate_below("customer_id", 0.25)
        .fresh_within("placed_at", "1h")
    )
    assert table_contract.validate().ok, table_contract.validate().violations
    stale = bt.from_pydict({"placed_at": [dt.datetime(2020, 1, 1)]})
    assert not stale.dq.fresh_within("placed_at", "1h").validate().ok

    # 7. Referential integrity, as a constraint the chain can act on.
    customers = bt.from_pydict({"customer_id": [10, 20, 30]})
    with_ref = ds.dq.references("customer_id", to=customers)
    assert with_ref.validate().ok  # the NULL key is "no reference", not a broken one
    orphaned = ds.with_columns(customer_id=col("customer_id").fill_null(999))
    assert orphaned.dq.references("customer_id", to=customers).drop().count() == 4

    # 8. One contract, many tables: `on` rebinds the whole chain.
    contract = ds.dq.not_null("order_id").positive("amount", mostly=0.8)
    tomorrow = bt.from_pydict({"order_id": [6, 7], "amount": [1.0, 2.0]})
    assert contract.on(tomorrow).validate().ok
    print("contract reused on", tomorrow.count(), "rows")


if __name__ == "__main__":
    main()

"""Checking that foreign keys point at rows that exist.

Every foreign key is a claim, and an anti join is the whole test. Running it at ingest turns
a class of downstream join bugs — rows silently vanishing at the next inner join — into a
report you can act on.

    python examples/quality/referential_integrity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    customer = tpch("customer")
    nation = tpch("nation")
    orders = tpch("orders")

    def orphans(child, child_key: str, parent, parent_key: str) -> int:
        return child.join(
            parent.select(parent_key), left_on=child_key, right_on=parent_key, how="anti"
        ).count()

    # `c_nationkey` -> `nation`: complete, because the whole dimension is present.
    bad_nations = orphans(customer, "c_nationkey", nation, "n_nationkey")
    print("customers with an unknown nation:", bad_nations)
    assert bad_nations == 0

    # `o_custkey` -> `customer`: the customer slice is smaller than the order slice, so
    # this legitimately finds orphans. That is the check working, not the data being bad.
    bad_customers = orphans(orders, "o_custkey", customer, "c_custkey")
    print("orders with an unknown customer:", bad_customers)
    assert bad_customers > 0

    # The orphans are exactly the rows an inner join would have dropped.
    inner = orders.join(customer.select("c_custkey"), left_on="o_custkey", right_on="c_custkey")
    assert inner.count() + bad_customers == orders.count()

    # Which is why the anti join belongs *before* the inner one: without it, the drop is
    # invisible.
    print(f"an inner join would silently drop {bad_customers} of {orders.count()} orders")

    # As a gate: the keys that must be complete are asserted, the ones that need not are
    # reported.
    report = customer.dq.not_null("c_nationkey").validate()
    assert report.ok
    assert bt is not None
    assert col is not None


if __name__ == "__main__":
    main()

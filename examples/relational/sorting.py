"""Sorting: direction per key, and where nulls land.

A multi-key sort takes a direction per key, so `descending=[True, False]` is a different
query from `descending=True`. Assert the order you asked for with an order-*dependent*
comparison; comparing sorted output as a set is how a sort bug survives a test suite.

    python examples/relational/sorting.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    customer = tpch("customer").select("c_custkey", "c_nationkey", "c_acctbal")

    ascending = customer.sort("c_acctbal").limit(5).to_pydict()
    print("poorest:", [round(value, 2) for value in ascending["c_acctbal"]])
    assert ascending["c_acctbal"] == sorted(ascending["c_acctbal"])

    descending = customer.sort("c_acctbal", descending=True).limit(5).to_pydict()
    print("richest:", [round(value, 2) for value in descending["c_acctbal"]])
    assert descending["c_acctbal"] == sorted(descending["c_acctbal"], reverse=True)

    # Two keys, opposite directions: nation ascending, balance descending within it.
    mixed = (
        customer.sort("c_nationkey", "c_acctbal", descending=[False, True]).limit(50).to_pydict()
    )
    pairs = list(zip(mixed["c_nationkey"], mixed["c_acctbal"], strict=True))
    assert pairs == sorted(pairs, key=lambda pair: (pair[0], -pair[1]))

    # Sorting by a derived expression needs the expression to exist as a column first.
    by_magnitude = (
        customer.with_columns(magnitude=col("c_acctbal").abs())
        .sort("magnitude", descending=True)
        .limit(5)
        .to_pydict()
    )
    assert by_magnitude["magnitude"] == sorted(by_magnitude["magnitude"], reverse=True)

    # `reverse` flips the current order rather than sorting again.
    reversed_head = customer.sort("c_custkey").limit(5).reverse().to_pydict()
    assert reversed_head["c_custkey"] == sorted(reversed_head["c_custkey"], reverse=True)


if __name__ == "__main__":
    main()

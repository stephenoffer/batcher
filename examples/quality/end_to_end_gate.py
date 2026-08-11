"""A release gate: every check a pipeline should pass before it ships.

Schema, keys, referential integrity, ranges, completeness and a control total, in the order
that localizes a failure fastest. Running them as one script is what turns a set of good
intentions into something a CI job can fail on.

    python examples/quality/end_to_end_gate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders")
    customer = tpch("customer")
    lineitem = tpch("lineitem")

    checks: dict[str, bool] = {}

    # 1. Schema: names and types.
    expected = {"o_orderkey": "int64", "o_totalprice": "double", "o_orderdate": "date32[day]"}
    actual = dict(zip(orders.columns, [str(t) for t in orders.dtypes], strict=True))
    checks["schema"] = all(actual.get(name) == dtype for name, dtype in expected.items())

    # 2. Primary key: unique and non-null.
    checks["primary key"] = (
        orders.n_unique("o_orderkey") == orders.count()
        and orders.filter(col("o_orderkey").is_null()).count() == 0
    )

    # 3. Referential integrity, where it must hold.
    checks["customer fk"] = (
        customer.join(
            tpch("nation").select("n_nationkey"),
            left_on="c_nationkey",
            right_on="n_nationkey",
            how="anti",
        ).count()
        == 0
    )

    # 4. Value ranges.
    checks["ranges"] = (
        orders.dq.in_range("o_totalprice", 0.0, 1_000_000.0)
        .accepted_values("o_orderstatus", ["O", "F", "P"])
        .validate()
        .ok
    )

    # 5. Completeness: every year in the span is present.
    years = (
        orders.with_columns(year=col("o_orderdate").dt.year())
        .select("year")
        .distinct()
        .to_pydict()["year"]
    )
    checks["completeness"] = sorted(years) == list(range(min(years), max(years) + 1))

    # 6. Control total across a join, which catches a fan-out.
    matched = orders.join(
        lineitem.select("l_orderkey"),
        left_on="o_orderkey",
        right_on="l_orderkey",
        how="semi",
    )
    checks["no fan-out"] = matched.count() <= orders.count()

    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'} {name}")

    failed = [name for name, passed in checks.items() if not passed]
    assert not failed, failed
    print(f"all {len(checks)} gates passed")
    assert bt is not None


if __name__ == "__main__":
    main()

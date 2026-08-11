"""TPC-H Q22 — customers with a healthy balance who have never ordered.

Two things worth copying: the country code is a substring of the phone number, taken
with `.str.slice`, and "never ordered" is an anti join rather than a `NOT IN`. The
average is over the qualifying population only, so it is computed after the balance
filter and before the anti join.

    python examples/tpch/q22_global_sales_opportunity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    customer = tpch("customer")
    orders = tpch("orders")

    codes = ["13", "31", "23", "29", "30", "18", "17"]

    with_code = customer.with_columns(cntrycode=col("c_phone").str.slice(0, 2)).filter(
        col("cntrycode").is_in(codes)
    )

    # The average is over customers with a positive balance, in those countries only.
    average = with_code.filter(col("c_acctbal") > 0.0).agg(avg=col("c_acctbal").mean())
    threshold = average.to_pydict()["avg"][0]
    print(f"threshold balance {threshold:,.2f}")

    result = (
        with_code.filter(col("c_acctbal") > threshold)
        .join(orders, left_on="c_custkey", right_on="o_custkey", how="anti")
        .group_by("cntrycode")
        .agg(numcust=col("c_custkey").count(), totacctbal=col("c_acctbal").sum())
        .sort("cntrycode")
        .to_pydict()
    )

    print(result)

    assert set(result["cntrycode"]) <= set(codes)
    assert result["cntrycode"] == sorted(result["cntrycode"])
    # Everyone counted is above the threshold, so the mean of each bucket must be too.
    assert all(
        total / count > threshold
        for total, count in zip(result["totacctbal"], result["numcust"], strict=True)
    )


if __name__ == "__main__":
    main()

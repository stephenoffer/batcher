"""TPC-H Q19 — three unrelated product filters OR'd into one scan.

Written as three queries this is three passes over `lineitem`. Written as one disjunction
it is one pass, and the engine can still push the parts common to all three branches
(the shipmode and instruction predicates) down ahead of the OR.

    python examples/tpch/q19_discounted_revenue.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem")
    part = tpch("part")

    joined = lineitem.join(part, left_on="l_partkey", right_on="p_partkey")

    shipped_by_hand = col("l_shipmode").is_in(["AIR", "AIR REG"]) & (
        col("l_shipinstruct") == "DELIVER IN PERSON"
    )

    small = (
        (col("p_brand") == "Brand#12")
        & col("p_container").is_in(["SM CASE", "SM BOX", "SM PACK", "SM PKG"])
        & (col("l_quantity") >= 1)
        & (col("l_quantity") <= 11)
        & (col("p_size") >= 1)
        & (col("p_size") <= 5)
    )
    medium = (
        (col("p_brand") == "Brand#23")
        & col("p_container").is_in(["MED BAG", "MED BOX", "MED PKG", "MED PACK"])
        & (col("l_quantity") >= 10)
        & (col("l_quantity") <= 20)
        & (col("p_size") >= 1)
        & (col("p_size") <= 10)
    )
    large = (
        (col("p_brand") == "Brand#34")
        & col("p_container").is_in(["LG CASE", "LG BOX", "LG PACK", "LG PKG"])
        & (col("l_quantity") >= 20)
        & (col("l_quantity") <= 30)
        & (col("p_size") >= 1)
        & (col("p_size") <= 15)
    )

    selected = joined.filter(shipped_by_hand & (small | medium | large))
    result = selected.agg(
        revenue=(col("l_extendedprice") * (1 - col("l_discount"))).sum()
    ).to_pydict()

    print("revenue:", result["revenue"])

    # A sum over an empty selection is null, not zero — worth being explicit about,
    # because a narrow disjunction over a data slice legitimately matches nothing.
    revenue = result["revenue"][0]
    assert revenue is None or revenue >= 0.0
    # Whatever survived the OR must satisfy the predicate every branch shares.
    kept = selected.select("l_shipmode", "l_shipinstruct").to_pydict()
    assert set(kept["l_shipmode"]) <= {"AIR", "AIR REG"}
    assert set(kept["l_shipinstruct"]) <= {"DELIVER IN PERSON"}


if __name__ == "__main__":
    main()

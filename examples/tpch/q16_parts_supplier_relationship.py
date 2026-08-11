"""TPC-H Q16 — how many suppliers can supply each part variant, after an exclusion.

The exclusion is an anti join against suppliers whose comment marks them as a complaint
case. Doing it with `NOT IN` over a subquery is the version that goes quadratic; the
anti join is the same answer in one hash pass.

    python examples/tpch/q16_parts_supplier_relationship.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    partsupp = tpch("partsupp")
    part = tpch("part")
    supplier = tpch("supplier")

    complaining = supplier.filter(
        col("s_comment").str.contains("Customer") & col("s_comment").str.contains("Complaints")
    ).select("s_suppkey")

    wanted_parts = part.filter(
        (col("p_brand") != "Brand#45")
        & ~col("p_type").str.starts_with("MEDIUM POLISHED")
        & col("p_size").is_in([49, 14, 23, 45, 19, 3, 36, 9])
    )

    result = (
        partsupp.join(complaining, left_on="ps_suppkey", right_on="s_suppkey", how="anti")
        .join(wanted_parts, left_on="ps_partkey", right_on="p_partkey")
        .group_by("p_brand", "p_type", "p_size")
        .agg(supplier_cnt=col("ps_suppkey").n_unique())
        .sort("supplier_cnt", "p_brand", "p_type", "p_size", descending=[True, False, False, False])
        .limit(20)
        .to_pydict()
    )

    print(f"{len(result['p_brand'])} variants; top count {result['supplier_cnt'][0]}")

    assert result["supplier_cnt"] == sorted(result["supplier_cnt"], reverse=True)
    assert "Brand#45" not in result["p_brand"]
    assert set(result["p_size"]) <= {49, 14, 23, 45, 19, 3, 36, 9}


if __name__ == "__main__":
    main()

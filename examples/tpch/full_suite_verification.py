"""Verifying every cached TPC-H table against the source it came from.

The examples suite is only as trustworthy as its fixture, so this checks the fixture: row
counts, schemas, key uniqueness and referential integrity across all eight tables. Run it
first when something in the suite looks wrong.

    python examples/tpch/full_suite_verification.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import TPCH_COLUMNS, is_offline, tpch


def main() -> None:
    tables = {name: tpch(name) for name in TPCH_COLUMNS}

    print("source:", "synthesized (offline)" if is_offline() else "s3://ray-benchmark-data")

    for name, dataset in tables.items():
        assert dataset.columns == list(TPCH_COLUMNS[name]), name
        assert dataset.count() > 0, name
        nulls = dataset.null_count().to_pydict()
        print(f"  {name:<10} {dataset.count():>7} rows  {dataset.width:>2} columns")
        assert all(value == 0 for column in nulls.values() for value in column), f"{name} has nulls"

    # The dimension tables are whole.
    assert tables["region"].count() == 5
    assert tables["nation"].count() == 25
    assert tables["supplier"].count() == 10_000

    # Primary keys are unique.
    for name, key in (
        ("region", "r_regionkey"),
        ("nation", "n_nationkey"),
        ("supplier", "s_suppkey"),
        ("customer", "c_custkey"),
        ("part", "p_partkey"),
        ("orders", "o_orderkey"),
    ):
        assert tables[name].n_unique(key) == tables[name].count(), name

    # Composite key on partsupp.
    assert (
        tables["partsupp"].select("ps_partkey", "ps_suppkey").distinct().count()
        == tables["partsupp"].count()
    )
    assert (
        tables["lineitem"].select("l_orderkey", "l_linenumber").distinct().count()
        == tables["lineitem"].count()
    )

    # Referential integrity where both sides are whole.
    orphan_nations = tables["nation"].join(
        tables["region"].select("r_regionkey"),
        left_on="n_regionkey",
        right_on="r_regionkey",
        how="anti",
    )
    assert orphan_nations.count() == 0

    orphan_suppliers = tables["supplier"].join(
        tables["nation"].select("n_nationkey"),
        left_on="s_nationkey",
        right_on="n_nationkey",
        how="anti",
    )
    assert orphan_suppliers.count() == 0

    print(f"all {len(tables)} tables verified")


if __name__ == "__main__":
    main()

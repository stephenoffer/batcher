"""What each TPC-H table costs to scan, and how much a projection saves.

The numbers here are the input to every join-order decision: which table is the fact table,
which are dimensions, and how much of each a query actually reads. Measuring them once is
worth more than guessing at them repeatedly.

    python examples/tpch/scan_and_project_costs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import TPCH_COLUMNS, tpch, tpch_path


def main() -> None:
    sizes: dict[str, int] = {}
    for table in TPCH_COLUMNS:
        dataset = tpch(table)
        sizes[table] = dataset.count()
        on_disk = Path(tpch_path(table)).stat().st_size
        print(
            f"{table:<10} {sizes[table]:>8} rows  {dataset.width:>2} cols  "
            f"{on_disk / 1024:>8.0f} KiB"
        )
        assert dataset.width == len(TPCH_COLUMNS[table])

    # The shape of the schema: one fact table, several dimensions.
    assert sizes["lineitem"] > sizes["orders"] > sizes["customer"]
    assert sizes["region"] == 5
    assert sizes["nation"] == 25

    # A projection narrows what the reader touches, and the row count is unchanged.
    full = tpch("lineitem")
    narrow = full.select("l_orderkey", "l_quantity")
    assert narrow.count() == full.count()
    assert narrow.width < full.width
    print(f"lineitem: {full.width} columns -> {narrow.width}")

    # The plan records the projection, which is what makes it pushable into the scan.
    plan = narrow.explain()
    print(plan)
    assert "scan" in plan.lower()
    assert bt is not None


if __name__ == "__main__":
    main()

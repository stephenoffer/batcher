"""Writing a report partitioned by a business key, and reading one partition back.

The payoff of partitioning is on the read side, and it only arrives if the reader can
recover the partition column — which means `read.parquet_dataset`, not `read.parquet`.
Writing without checking the read path is how a partitioned table ends up scanned in full.

    python examples/io/writing_partitioned_reports.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    report = (
        tpch("orders")
        .with_columns(year=col("o_orderdate").dt.year())
        .select("year", "o_orderkey", "o_totalprice", "o_orderpriority")
    )

    with tempfile.TemporaryDirectory() as directory:
        root = str(Path(directory) / "orders_by_year")
        report.write.parquet(root, partition_by=["year"])

        years = sorted(p.name for p in Path(root).iterdir() if p.is_dir())
        print(years[:4], "...")
        assert all(name.startswith("year=") for name in years)
        assert len(years) == report.n_unique("year")

        # The dataset reader recovers the partition column.
        back = bt.read.parquet_dataset(root)
        assert "year" in back.columns
        assert back.count() == report.count()

        # One partition, by filtering on the recovered column.
        one_year = int(years[0].split("=")[1])
        slice_of = back.filter(col("year") == one_year)
        direct = report.filter(col("year") == one_year)
        print(f"year {one_year}: {slice_of.count()} rows")
        assert slice_of.count() == direct.count()

        totals = slice_of.agg(t=col("o_totalprice").sum()).to_pydict()["t"][0]
        expected = direct.agg(t=col("o_totalprice").sum()).to_pydict()["t"][0]
        assert abs(totals - expected) < 1e-3


if __name__ == "__main__":
    main()

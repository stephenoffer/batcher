"""Reading an earlier version of a Delta table.

Time travel is not a backup feature — it is a consequence of the log. Each commit adds
files rather than replacing them, so an older version is still fully described and can be
read as it stood. That is also what makes `vacuum` destructive: it removes the files older
versions still point at.

    python examples/io/delta_time_travel.py
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
    customer = tpch("customer").select("c_custkey", "c_name", "c_acctbal")

    with tempfile.TemporaryDirectory() as directory:
        table = str(Path(directory) / "customers")

        customer.head(500).write.delta(table)  # version 0
        customer.slice(500, 300).write.delta(table, mode="append")  # version 1
        customer.slice(800, 200).write.delta(table, mode="append")  # version 2

        latest = bt.read.delta(table)
        print("latest:", latest.count())
        assert latest.count() == 1_000

        # Each earlier version is still readable, exactly as it stood.
        v0 = bt.read.delta(table, version=0)
        v1 = bt.read.delta(table, version=1)
        print("v0:", v0.count(), "v1:", v1.count())
        assert v0.count() == 500
        assert v1.count() == 800

        # An append is additive, so an old version's rows are a subset of the new one's.
        old_keys = set(v0.to_pydict()["c_custkey"])
        new_keys = set(latest.to_pydict()["c_custkey"])
        assert old_keys < new_keys

        # And the aggregate over an old version matches what it was at the time.
        original = customer.head(500).agg(total=col("c_acctbal").sum()).to_pydict()["total"][0]
        travelled = v0.agg(total=col("c_acctbal").sum()).to_pydict()["total"][0]
        assert abs(original - travelled) < 1e-6


if __name__ == "__main__":
    main()

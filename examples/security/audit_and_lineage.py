"""Proving where a governed column went.

An audit asks two things: who could read this, and where did it flow. The first is the
catalog; the second is lineage over the plan. Both are answerable without running anything,
which is what makes them usable in review rather than after an incident.

    python examples/security/audit_and_lineage.py
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
    with tempfile.TemporaryDirectory() as directory:
        table = str(Path(directory) / "customers")
        tpch("customer").select("c_custkey", "c_name", "c_phone", "c_nationkey").head(
            1_000
        ).write.parquet(table)

        analyst = bt.Principal("analyst", roles={"analyst"})
        catalog = (
            bt.SecurityCatalog()
            .grant("analyst", on=table, select=["c_custkey", "c_nationkey"])
            .tag(table, "c_phone", "pii")
        )

        # The grant is a whitelist: columns outside it are not readable.
        with bt.security(catalog, analyst):
            allowed = bt.read.parquet(table).select("c_custkey", "c_nationkey")
            print("allowed columns:", allowed.columns)
            assert allowed.count() == 1_000

            try:
                bt.read.parquet(table).select("c_phone").count()
            except Exception as error:
                print("denied:", type(error).__name__, str(error)[:60])
            else:
                raise AssertionError("selecting an ungranted column must be denied")

        # Lineage over a pipeline that reads the governed table.
        pipeline = bt.read.parquet(table).group_by("c_nationkey").agg(customers=bt.count())
        trace = pipeline.lineage()
        print("lineage:", str(trace)[:120])
        assert trace is not None
        assert pipeline.count() > 0
        assert col is not None


if __name__ == "__main__":
    main()

"""The full quarantine loop: split, write both sides, and reconcile.

A rejected row is data, not an error. Writing it somewhere with the reason it failed is what
makes a pipeline debuggable — and reconciling the two counts against the input is what
proves nothing was lost on the way.

    python examples/quality/quarantine_workflow.py
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
    lineitem = tpch("lineitem").select("l_orderkey", "l_linenumber", "l_quantity", "l_discount")
    total = lineitem.count()

    contract = lineitem.dq.in_range("l_discount", 0.0, 0.04).in_range("l_quantity", 1, 40)
    clean, rejected = contract.quarantine()

    print(f"{clean.count()} clean, {rejected.count()} quarantined of {total}")

    # Nothing is lost and nothing is duplicated.
    assert clean.count() + rejected.count() == total
    assert clean.count() > 0
    assert rejected.count() > 0

    # The clean side genuinely satisfies the contract.
    assert clean.dq.in_range("l_discount", 0.0, 0.04).in_range("l_quantity", 1, 40).validate().ok

    # Every quarantined row violates at least one rule.
    violations = rejected.filter(
        (col("l_discount") > 0.04) | (col("l_quantity") > 40) | (col("l_quantity") < 1)
    )
    assert violations.count() == rejected.count()

    with tempfile.TemporaryDirectory() as directory:
        good = str(Path(directory) / "clean.parquet")
        bad = str(Path(directory) / "quarantine.parquet")
        clean.write.parquet(good)
        rejected.write.parquet(bad)

        # Reconcile what landed on disk against what went in.
        written_clean = bt.read.parquet(good).count()
        written_bad = bt.read.parquet(bad).count()
        print(f"on disk: {written_clean} clean, {written_bad} quarantined")
        assert written_clean + written_bad == total


if __name__ == "__main__":
    main()

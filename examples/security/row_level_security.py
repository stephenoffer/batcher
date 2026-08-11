"""Restricting which rows a principal can see.

Row-level security is a filter injected into the plan, not a check applied to the result.
That distinction matters: an aggregate computed by a restricted principal is computed over
the restricted rows, so a count cannot leak the size of the hidden set.

    python examples/security/row_level_security.py
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
        source = tpch("customer").select("c_custkey", "c_nationkey", "c_acctbal").head(2_000)
        source.write.parquet(table)

        analyst = bt.Principal("analyst", roles={"analyst"})
        admin = bt.Principal("admin", roles={"admin"})

        catalog = (
            bt.SecurityCatalog()
            .grant("analyst", on=table)
            .grant("admin", on=table)
            .filter_rows(table, lambda principal: col("c_nationkey") < 5, exempt=["admin"])
        )

        with bt.security(catalog, analyst):
            restricted = bt.read.parquet(table)
            visible = restricted.count()
            nations = set(restricted.select("c_nationkey").distinct().to_pydict()["c_nationkey"])
            restricted_total = restricted.agg(t=col("c_acctbal").sum()).to_pydict()["t"][0]

        with bt.security(catalog, admin):
            everything = bt.read.parquet(table)
            full = everything.count()
            full_total = everything.agg(t=col("c_acctbal").sum()).to_pydict()["t"][0]

        print(f"analyst sees {visible} of {full} rows")
        assert visible < full
        assert nations <= {0, 1, 2, 3, 4}

        # The aggregate is over the visible rows only, so it cannot reveal the rest.
        assert restricted_total < full_total

        # And the unrestricted view matches the ungoverned source exactly.
        assert full == source.count()


if __name__ == "__main__":
    main()

"""Recording who ran what, and proving the policy applied.

A governed read is a plan rewrite, so the evidence that a policy applied is in the plan
rather than in a log line someone remembered to write. Comparing the governed and ungoverned
plans is the audit.

    python examples/security/audit_trail.py
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
        source = (
            tpch("customer")
            .select("c_custkey", "c_name", "c_phone", "c_nationkey", "c_acctbal")
            .head(2_000)
        )
        source.write.parquet(table)

        analyst = bt.Principal("analyst", roles={"analyst"})
        catalog = (
            bt.SecurityCatalog()
            .grant("analyst", on=table)
            .tag(table, "c_phone", "pii")
            .mask_tag("pii", lambda c: bt.mask(c, show_last=4))
            .filter_rows(table, lambda principal: col("c_nationkey") < 10)
        )

        ungoverned = bt.read.parquet(table)
        ungoverned_plan = ungoverned.explain()
        ungoverned_rows = ungoverned.count()

        with bt.security(catalog, analyst):
            governed = bt.read.parquet(table)
            governed_plan = governed.explain()
            governed_rows = governed.count()

        print("ungoverned rows:", ungoverned_rows)
        print("governed rows:  ", governed_rows)

        # The policy is visible in the plan, not only in the result.
        assert governed_plan != ungoverned_plan
        assert "filter" in governed_plan.lower()
        assert governed_rows < ungoverned_rows

        # What the catalog says it governs, before anything runs.
        assert catalog.governs(table)
        visible = catalog.visible_columns(table, source.columns, analyst)
        print("visible columns:", visible)
        assert "c_custkey" in visible
        assert set(visible) <= set(source.columns)

        # And the row filter the principal will get.
        filters = catalog.row_filters_for(table, analyst)
        print("row filters:", len(filters))
        assert len(filters) == 1


if __name__ == "__main__":
    main()

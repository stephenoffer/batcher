"""Column masking and row filtering as a plan rewrite, not a wrapper.

Governance in Batcher is a *rewrite*: the policy is compiled into the plan before it runs,
so there is no unenforced path around it and no per-row Python check. A ``SecurityCatalog``
declares the policy, a ``Principal`` is the identity, and ``bt.security(...)`` installs
both for the duration of a block.

    python examples/governance/masking_and_filters.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import batcher as bt
from batcher import col


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        table = str(Path(tmp) / "customers")
        bt.from_pydict(
            {
                "id": [1, 2, 3],
                "email": ["ada@example.com", "grace@example.com", "alan@example.com"],
                "region": ["eu", "us", "eu"],
                "salary": [100, 200, 300],
            }
        ).write.parquet(table)

        analyst = bt.Principal("analyst", roles={"analyst"})
        admin = bt.Principal("admin", roles={"admin"})

        catalog = (
            bt.SecurityCatalog()
            # The first grant on a table switches it to deny-by-default.
            .grant("analyst", on=table, select=["id", "email", "region"])
            .grant("admin", on=table)
            # Classify once, then govern everything carrying the tag.
            .tag(table, "email", "pii")
            .mask_tag("pii", lambda c: bt.mask(c, show_last=6), exempt=["admin"])
            # Row-level security: analysts see only EU rows.
            .filter_rows(table, lambda principal: col("region") == "eu", exempt=["admin"])
        )

        with bt.security(catalog, analyst):
            restricted = bt.read.parquet(table).sort("id").to_pydict()
        print("analyst sees:", restricted)

        # `salary` was never granted, so it does not exist for this principal.
        assert "salary" not in restricted
        # Only EU rows survive the row filter.
        assert restricted["region"] == ["eu", "eu"]
        assert restricted["id"] == [1, 3]
        # The email is masked, keeping only the last six characters.
        assert restricted["email"][0] != "ada@example.com"
        assert restricted["email"][0].endswith("le.com")

        with bt.security(catalog, admin):
            full = bt.read.parquet(table).sort("id").to_pydict()
        print("admin sees:", full)

        assert "salary" in full
        assert len(full["id"]) == 3
        assert full["email"][0] == "ada@example.com"

        # Because enforcement is a rewrite, it survives composition: this aggregate runs
        # over the filtered rows rather than being applied afterwards.
        with bt.security(catalog, analyst):
            total = bt.read.parquet(table).select(t=col("id").sum()).to_pydict()
        print("analyst id total:", total)
        assert total["t"] == [4]  # ids 1 and 3


if __name__ == "__main__":
    main()

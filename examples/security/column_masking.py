"""Masking a sensitive column by tag, not by name.

Tagging a column once and governing the tag is what keeps policy from drifting: a new table
with an email column inherits the rule by being classified, rather than by someone
remembering to add it to a list.

    python examples/security/column_masking.py
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
        tpch("customer").select("c_custkey", "c_name", "c_phone", "c_acctbal").head(
            500
        ).write.parquet(table)

        analyst = bt.Principal("analyst", roles={"analyst"})
        admin = bt.Principal("admin", roles={"admin"})

        catalog = (
            bt.SecurityCatalog()
            .grant("analyst", on=table)
            .grant("admin", on=table)
            .tag(table, "c_phone", "pii")
            .mask_tag("pii", lambda c: bt.mask(c, show_last=4), exempt=["admin"])
        )

        with bt.security(catalog, analyst):
            masked = bt.read.parquet(table).select("c_custkey", "c_phone").head(3).to_pydict()
        with bt.security(catalog, admin):
            clear = bt.read.parquet(table).select("c_custkey", "c_phone").head(3).to_pydict()

        print("analyst sees:", masked["c_phone"])
        print("admin sees:  ", clear["c_phone"])

        # The masked value keeps its shape and its last four characters.
        assert masked["c_phone"] != clear["c_phone"]
        assert all(
            hidden[-4:] == visible[-4:]
            for hidden, visible in zip(masked["c_phone"], clear["c_phone"], strict=True)
        )
        assert all("X" in value for value in masked["c_phone"])

        # Masking is a plan rewrite, so it applies wherever the column is read — including
        # inside an aggregate, and including a column the query never projects.
        with bt.security(catalog, analyst):
            grouped = (
                bt.read.parquet(table)
                .group_by("c_phone")
                .agg(n=bt.count())
                .sort("n", descending=True)
                .head(1)
                .to_pydict()
            )
        assert "X" in grouped["c_phone"][0]
        assert col is not None


if __name__ == "__main__":
    main()

"""Reshaping the column list: rename, drop, and selecting by dtype.

Renaming is metadata-only, so it costs nothing at runtime. Dropping is the same: the
column simply stops being projected, and if the source is Parquet the reader never reads
its bytes at all.

    python examples/relational/rename_and_drop.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch


def main() -> None:
    customer = tpch("customer")

    renamed = customer.rename({"c_name": "name", "c_acctbal": "balance"})
    print(renamed.columns)
    assert "name" in renamed.columns
    assert "c_name" not in renamed.columns
    # The data is untouched; only the label moved.
    assert (
        renamed.select("balance").head(3).to_pydict()["balance"]
        == (customer.select("c_acctbal").head(3).to_pydict()["c_acctbal"])
    )

    trimmed = customer.drop("c_comment", "c_address")
    assert "c_comment" not in trimmed.columns
    assert len(trimmed.columns) == len(customer.columns) - 2

    # Select by type when the names are not known ahead of time.
    numeric = customer.select_dtypes("int64")
    print("int64 columns:", numeric.columns)
    assert set(numeric.columns) == {"c_custkey", "c_nationkey"}

    # `dtypes` and `schema` are the two ways to read the shape back.
    print(customer.dtypes[:3])
    assert len(customer.dtypes) == len(customer.columns)
    assert customer.schema.names == customer.columns


if __name__ == "__main__":
    main()

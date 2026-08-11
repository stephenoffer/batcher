"""Recovering from a failure without swallowing it.

Catching the narrowest exception that can happen is what lets a pipeline retry the one step
that is retryable and fail on everything else. A bare `except Exception` turns a schema bug
into a silent empty result.

    python examples/operations/typed_error_recovery.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col
from batcher._internal.errors import PlanError


def main() -> None:
    orders = tpch("orders")

    def revenue_by(column: str) -> bt.Dataset | None:
        """Group by a column that may not exist, reporting rather than crashing."""
        try:
            return orders.group_by(column).agg(total=col("o_totalprice").sum())
        except PlanError as error:
            print(f"  cannot group by {column!r}: {str(error)[:60]}")
            return None

    good = revenue_by("o_orderstatus")
    assert good is not None
    assert good.count() == orders.n_unique("o_orderstatus")

    missing = revenue_by("o_channel")
    assert missing is None

    # A file that is not there is an IO failure, not a plan failure — a different class,
    # and a retryable one.
    try:
        bt.read.parquet("/nonexistent/path/data.parquet").count()
    except Exception as error:
        print("  missing file:", type(error).__name__)
        assert not isinstance(error, PlanError)
    else:
        raise AssertionError("expected a failure for a missing file")

    # The pipeline continues on the paths that did work.
    assert good.agg(t=col("total").sum()).to_pydict()["t"][0] > 0


if __name__ == "__main__":
    main()

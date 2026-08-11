"""The typed exceptions, and catching the right one.

Every failure mode has its own class, so a caller can distinguish "you wrote the query
wrong" from "the data is not what you promised" from "the file is not there". Catching
bare `Exception` throws that away.

    python examples/operations/error_types.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col
from batcher._internal.errors import ColumnNotFoundError, DataQualityError, PlanError


def main() -> None:
    orders = tpch("orders").select("o_orderkey", "o_totalprice")

    # A column that does not exist is caught when the plan is built, not when it runs.
    try:
        orders.select("o_nonexistent")
    except ColumnNotFoundError as error:
        print("ColumnNotFoundError:", str(error)[:80])
    else:
        raise AssertionError("expected ColumnNotFoundError")

    # A malformed plan is a PlanError, and ColumnNotFoundError is one.
    assert issubclass(ColumnNotFoundError, PlanError)

    # A broken group-by key, same class of mistake.
    try:
        orders.group_by("not_a_column").agg(n=bt.count())
    except PlanError as error:
        print("PlanError:", str(error)[:80])
    else:
        raise AssertionError("expected PlanError")

    # A violated data contract is a different thing entirely: the query was fine, the
    # data was not.
    try:
        orders.dq.in_range("o_totalprice", 0.0, 1.0).fail()
    except DataQualityError as error:
        print("DataQualityError:", str(error)[:80])
    else:
        raise AssertionError("expected DataQualityError")

    # Errors name the available columns, so the message is the fix.
    try:
        orders.select(col("o_totalpric"))
    except ColumnNotFoundError as error:
        assert "o_totalprice" in str(error)
        print("the message suggests the right name")


if __name__ == "__main__":
    main()

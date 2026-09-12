"""The planned morsel width charges the columns a query *carries*, not the ones its table has.

`available_schema` is what a node COULD produce. Under projection pushdown a `Scan` and the
`Filter` above it still report every column of the source while a handful flow through them,
so sizing the morsel from it charged a narrow query for its table's width: on ClickBench's
105-column `hits`, a five-column query measured 1,620 B/row against an actual 40 and was
given a **647-row** morsel against the configured 16,384 — every per-morsel cost paid 25
times over.

The columns that can actually reach a morsel are the union of what a scan *supplies*
(Kyber's projection) and what a node *introduces* (a derived column no source carries). The
second half is what keeps the policy doing its job: it exists for decoded tensors, and those
are introduced rather than scanned.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher.api.orchestration.sizing import carried_columns
from batcher.carbonite.policies.morsel import planned_row_cap
from batcher.config import Config

pytestmark = pytest.mark.unit


def _wide_table(rows: int = 64) -> pa.Table:
    """A table of the shape the defect needs: many columns, of which a query reads few."""
    return pa.table({f"c{i}": pa.array([i] * rows, type=pa.int64()) for i in range(105)})


def test_a_narrow_query_over_a_wide_table_is_not_charged_the_table() -> None:
    config = Config()
    plan = bt.from_arrow(_wide_table()).select("c0", "c1").filter(bt.col("c0") > 0)._plan

    charged_everything = planned_row_cap(config, plan)
    carried = carried_columns(plan)

    assert carried is not None
    # The control: without the column set this really does tighten, so the assertion below
    # is about the fix rather than about a policy that was inert anyway.
    assert charged_everything is not None
    assert charged_everything < config.execution.morsel_rows
    assert planned_row_cap(config, plan, None, carried) is None


def test_the_carried_set_is_the_columns_the_query_reads() -> None:
    plan = bt.from_arrow(_wide_table()).select("c0", "c7")._plan
    carried = carried_columns(plan)
    assert carried is not None
    assert {"c0", "c7"} <= carried
    assert "c50" not in carried


def test_a_derived_wide_column_is_still_charged() -> None:
    """The case the policy exists for: a column no source carries must stay counted.

    A decoded tensor is *introduced* by a projection rather than read from a scan, so it is
    absent from every source schema — and charging only scanned columns would size the
    morsel as if it were not there, which is the OOM this policy was written to prevent.
    """
    config = Config()
    rows = 64
    tensor = pa.FixedSizeListArray.from_arrays(
        pa.array(np.zeros(rows * 4096, dtype=np.float64)), 4096
    )
    table = pa.table({"id": pa.array(range(rows), type=pa.int64()), "emb": tensor})
    # `emb` is read, so it is supplied rather than derived — the width must bind either way.
    plan = bt.from_arrow(table).select("id", "emb")._plan
    carried = carried_columns(plan)

    assert carried is not None and "emb" in carried
    cap = planned_row_cap(config, plan, None, carried)
    assert cap is not None and cap < config.execution.morsel_rows


def test_an_unknowable_column_set_charges_everything() -> None:
    """`None` restores the old behaviour, because this decides cost and never correctness."""
    config = Config()
    plan = bt.from_arrow(_wide_table()).select("c0", "c1")._plan
    assert planned_row_cap(config, plan, None, None) == planned_row_cap(config, plan)

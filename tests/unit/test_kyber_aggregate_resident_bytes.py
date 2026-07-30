"""What an aggregate actually *holds* — which is not its groups when the key is wide.

The parallel aggregate is `partial -> combine -> finalize`: every morsel builds its own group
table and all of them are live when `combine` merges them. So what it holds is the sum of the
per-morsel partials, and the reduction that decides how big those are is the one **within a
morsel**, not the global one.

That asymmetry was the defect. `GROUP BY k` over 24 M rows into 2 M groups reduces 12:1
globally, so budgeting the output gave 2 M x 16 B = 38 MB. But a 16,384-row morsel of a 2 M-group
key space holds ~16,300 distinct keys, so it reduces by nothing: every input row survives into a
partial and the live partial state is the size of the input again. Measured on that query: peak
RSS ~2.4 GB against a 537 MB envelope, and the query was never routed out of core because the
estimate said 38 MB.

`agg_par.rs` states this exact asymmetry about *CPU* -- "when grouping does not reduce, the
pre-aggregation is pure overhead" -- and it had never been applied to memory.

These tests pin both ends, because a fix that only raised the wide case would make every
`GROUP BY flag` spill for no reason.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher.config import active_config
from batcher.kyber.annotate import _resident_bytes
from batcher.kyber.cardinality import CardinalityEstimator
from batcher.plan.visitor import children

pytestmark = pytest.mark.unit

ROWS = 2_000_000


def _sized_source(distinct: int) -> tuple:
    """A `ROWS`-row source whose key column has `distinct` values, and its group-by plan."""
    rng = np.random.default_rng(0)
    table = pa.table(
        {
            "k": rng.integers(0, distinct, ROWS).astype("int64"),
            "v": rng.integers(0, 100, ROWS).astype("int64"),
        }
    )
    ds = bt.from_arrow(table).group_by("k").agg(sv=bt.col("v").sum())
    return ds._plan, ds._sources


def _resident(plan, sources, group_rows: float) -> tuple[int, float, int]:
    """`(resident bytes, width, morsel_rows)` for `plan`, with its group count forced.

    The group count is supplied rather than estimated: this is a test of the *memory* rule,
    and letting a cardinality estimate vary underneath it would make the assertion about the
    estimator instead.
    """
    cfg = active_config()
    est = CardinalityEstimator(sources=sources)
    width = est.row_width(plan, cfg.optimizer.row_bytes)
    morsel_rows = cfg.execution.morsel_rows
    return (
        _resident_bytes(plan, group_rows, width, est, morsel_rows),
        width,
        morsel_rows,
    )


def test_a_wide_group_key_is_budgeted_at_its_live_partial_state() -> None:
    """A key with more distinct values than a morsel has rows: the partials hold the input.

    Every morsel's partial keeps nearly every row it saw, and they are all live when `combine`
    runs, so the resident state is about the input's size -- not the (much smaller) group
    output.
    """
    plan, sources = _sized_source(distinct=ROWS // 2)
    group_rows = float(ROWS // 2)
    resident, width, _ = _resident(plan, sources, group_rows)

    input_bytes = ROWS * width
    output_bytes = group_rows * width
    assert group_rows < ROWS, "the fixture should still reduce globally"
    assert resident >= input_bytes * 0.9, (
        f"a {group_rows:,.0f}-group key over {ROWS:,} rows was budgeted at {resident:,} bytes, "
        f"below the ~{input_bytes:,.0f} its live per-morsel partials hold. Budgeting the "
        f"output ({output_bytes:,.0f}) is what let a query whose real peak was 4.6x the "
        f"envelope stay on the in-memory path."
    )


def test_a_narrow_group_key_is_still_budgeted_at_its_groups() -> None:
    """`GROUP BY flag` must not be pushed toward spilling.

    Three groups means a morsel of 16,384 rows collapses to three, so the live partial state
    is `3 / morsel_rows` of the input -- negligible. If the wide-key fix raised this one too,
    every reducing group-by in every plan would have gained a false envelope.
    """
    plan, sources = _sized_source(distinct=3)
    resident, width, _ = _resident(plan, sources, 3.0)

    input_bytes = ROWS * width
    assert resident <= input_bytes * 0.01, (
        f"a 3-group key was budgeted at {resident:,} bytes, a large fraction of the "
        f"{input_bytes:,.0f}-byte input -- a reducing group-by has gained an envelope it "
        f"does not need, which pushes it toward spilling for nothing"
    )
    # It is still budgeted at *something*: the groups themselves.
    assert resident >= 3 * width


def test_the_budget_rises_with_the_key_width_and_saturates_at_the_input() -> None:
    """The rule is monotone in distinctness and cannot exceed the input.

    Monotone because a wider key reduces less per morsel; capped because a partial cannot
    hold more rows than were fed to it. Checked across the range rather than at two points,
    so a rule that happened to be right at both ends but wrong between them fails here.
    """
    plan, sources = _sized_source(distinct=ROWS // 2)
    cfg = active_config()
    est = CardinalityEstimator(sources=sources)
    width = est.row_width(plan, cfg.optimizer.row_bytes)
    morsel_rows = cfg.execution.morsel_rows
    input_bytes = ROWS * width

    previous = 0
    for groups in (1, 10, 1_000, morsel_rows // 2, morsel_rows, morsel_rows * 4, ROWS):
        resident = _resident_bytes(plan, float(groups), width, est, morsel_rows)
        assert resident >= previous, f"budget fell going to {groups} groups"
        assert resident <= input_bytes * 1.05 or groups >= ROWS, (
            f"{groups} groups budgeted at {resident:,}, above the {input_bytes:,.0f}-byte "
            f"input its partials are built from"
        )
        previous = resident


def test_an_unknown_morsel_size_falls_back_to_the_output() -> None:
    """With no morsel size there is no per-morsel reduction to reason about, so the rule must
    decline rather than guess -- the estimator's standing contract is that a query is never
    pushed to spill on a guess."""
    plan, sources = _sized_source(distinct=ROWS // 2)
    cfg = active_config()
    est = CardinalityEstimator(sources=sources)
    width = est.row_width(plan, cfg.optimizer.row_bytes)
    group_rows = float(ROWS // 2)
    assert _resident_bytes(plan, group_rows, width, est, 0) == int(group_rows * width)


def test_the_rule_reads_the_aggregate_s_own_input() -> None:
    """The partial state is sized from the *aggregate's input*, not from the leaf source.

    A filter below the aggregate cuts what its partials are built from, and budgeting the
    unfiltered source would over-state the envelope for the most ordinary plan there is.
    """
    rng = np.random.default_rng(0)
    table = pa.table(
        {
            "k": rng.integers(0, ROWS // 2, ROWS).astype("int64"),
            "v": rng.integers(0, 100, ROWS).astype("int64"),
        }
    )
    wide = bt.from_arrow(table).group_by("k").agg(sv=bt.col("v").sum())
    filtered = bt.from_arrow(table).filter(bt.col("v") < 5).group_by("k").agg(sv=bt.col("v").sum())

    cfg = active_config()
    group_rows = float(ROWS // 2)
    est_w = CardinalityEstimator(sources=wide._sources)
    est_f = CardinalityEstimator(sources=filtered._sources)
    mr, rb = cfg.execution.morsel_rows, cfg.optimizer.row_bytes

    unfiltered = _resident_bytes(wide._plan, group_rows, est_w.row_width(wide._plan, rb), est_w, mr)
    narrowed = _resident_bytes(
        filtered._plan, group_rows, est_f.row_width(filtered._plan, rb), est_f, mr
    )
    assert list(children(filtered._plan)), "the fixture should have a child to read"
    assert narrowed < unfiltered, (
        f"a filtered aggregate was budgeted at {narrowed:,}, no less than the "
        f"{unfiltered:,} of the unfiltered one -- the rule is reading past its own input"
    )

"""An `explain()` on a large or deep plan has to stay readable, which is when it is read.

A plan small enough to take in at a glance rarely needs explaining. The output that matters
is the one for a 60-operator pipeline, and that is exactly the case the renderer failed --
in two ways at once, both of which turn the most informative column into the least.

**The spine grew without bound.** Indentation is three characters per level and the operator
column is capped at 48, so past depth 16 the indent alone filled the column. `fit` keeps the
*front* of a string, so what survived truncation was the indentation and what was discarded
was the operator's name: forty rows of a 61-operator plan rendered as bare ellipsis. A deep
linear chain is not an exotic shape -- it is what any pipeline of chained `with_columns` and
`filter` calls is.

**The estimate-miss ratio had no ceiling.** A selectivity estimate underflows toward zero
down a chain of independent predicates, each multiplying the last, so the denominator reaches
1e-13 and the ratio renders as ``205891132094649.4x under``: 24 characters of noise in a
column every operator on the plan then pays width for.

Both are asserted here against a synthetic profile rather than a query, because the defect is
in the renderer and a query would make the test depend on the optimizer's estimates too.
"""

from __future__ import annotations

import pytest

from batcher._internal.humanize import signed_ratio
from batcher.plan.profile import OpProfile, QueryProfile
from batcher.plan.profile.render import render_profile
from batcher.plan.profile.render.options import GLYPHS, MAX_SPINE_DEPTH
from batcher.plan.profile.render.tree import last_child_flags, spine

pytestmark = pytest.mark.unit

#: The width `layout._table` caps the operator column at, and the cells an operator's name
#: needs inside it to still be identifiable. Both are readability facts about the table, not
#: knobs -- which is why they are stated here rather than imported from the thing under test.
LABEL_COLUMN_CAP = 48
MIN_NAME_CELLS = 12


def _chain(depth: int) -> QueryProfile:
    """A measured linear plan `depth` operators deep, each with a distinguishable name."""
    ops = tuple(
        OpProfile(
            op_id=i,
            kind=f"filter_{i:02d}",
            depth=i,
            est_rows=100.0,
            measured=True,
            rows_in=100,
            rows_out=100,
            elapsed_ms=1.0,
        )
        for i in range(depth)
    )
    return QueryProfile(ops=ops, rows=100, total_ms=float(depth), measured=True)


def test_every_operator_in_a_deep_plan_is_still_named():
    """The property the unbounded spine destroyed: you can tell the rows apart."""
    rendered = render_profile(_chain(40), analyze=True, width=120)
    for i in range(40):
        assert f"filter_{i:02d}" in rendered, f"operator {i} lost its name:\n{rendered}"


def test_the_spine_stops_indenting_rather_than_pushing_the_name_out():
    """Indentation is bounded, so the label column cannot be consumed by it.

    Asserted on `spine` itself rather than on a rendered line, because the rendered line is
    already clipped to the label column -- so measuring it would report a bounded indent
    whether or not the spine is bounded, which is a test that cannot fail. Run against the
    unclamped spine this fails at depth 11 and every depth after it.
    """
    ops = _chain(40).ops
    flags = last_child_flags(ops)
    widths = [len(spine(ops, i, flags, GLYPHS)) for i in range(len(ops))]
    # An absolute ceiling, deliberately *not* derived from `MAX_SPINE_DEPTH`. Asserting
    # against the constant would make this pass for any value of it, including the
    # unbounded one -- a test that cannot fail, which is the failure mode this whole file
    # is about. The number that matters is a layout fact: `_table` caps the operator column
    # at LABEL_COLUMN_CAP cells, and a name needs room inside it.
    assert max(widths) <= LABEL_COLUMN_CAP - MIN_NAME_CELLS, (
        f"the spine takes {max(widths)} of the operator column's {LABEL_COLUMN_CAP} cells, "
        f"leaving under {MIN_NAME_CELLS} for the operator's name"
    )
    # And it really is still a tree for the depths that fit, not a flattened list.
    assert widths[5] > widths[4]
    # The clamp engages rather than the chain merely being shallow enough to fit.
    assert widths[-1] == widths[MAX_SPINE_DEPTH + 1]


def test_the_deep_rows_say_that_indentation_was_elided():
    """A clamped row is marked, so a reader is not told the plan is flatter than it is."""
    rendered = render_profile(_chain(40), analyze=True, width=120)
    assert "⋯" in rendered


def test_a_shallow_plan_is_drawn_exactly_as_before():
    """The clamp must not touch the plans that were already fine -- the common case.

    The control for the two tests above: they would both pass if the spine had simply been
    removed, which would be a worse renderer, not a better one.
    """
    rendered = render_profile(_chain(4), analyze=True, width=120)
    assert "⋯" not in rendered
    assert "   └─ filter_02" in rendered


def test_an_estimate_below_one_row_compares_against_one_row():
    """A row count is a count, so a sub-one estimate has no usable denominator.

    Down a chain of independent predicates the selectivity estimate underflows -- each one
    multiplies the last -- so the denominator reaches 1e-13 and the ratio rendered as
    ``205891132094649.4x under``: fifteen significant digits of an artifact. Flooring the
    denominator at one row is what makes the figure agree with the ``est≈0`` printed beside
    it.
    """
    assert signed_ratio(200, 1e-12) == "200.0x under"
    assert signed_ratio(200, 0.5) == "200.0x under"


def test_a_real_miss_is_still_reported_in_full():
    """The control this file exists for, and the reason a rendered cap was the wrong fix.

    An estimate of 1,000,000 against 200 actual rows is a legitimate 5,000x miss and is
    exactly the signal `explain(analyze=True)`'s diagnosis section is there to raise. A
    ceiling on the printed figure suppresses it along with the artifact, which is a worse
    renderer, not a safer one.
    """
    assert signed_ratio(200, 1_000_000) == "5000.0x over"
    assert signed_ratio(1000, 300) == "3.3x under"
    assert signed_ratio(100, 100) == "exact"


def test_the_miss_column_cannot_widen_without_bound():
    """What the cap is for: one bad estimate must not cost every row its width."""
    ops = tuple(
        OpProfile(
            op_id=i,
            kind="filter",
            depth=i,
            est_rows=1e-13,
            measured=True,
            rows_in=200,
            rows_out=200,
            elapsed_ms=1.0,
        )
        for i in range(6)
    )
    profile = QueryProfile(ops=ops, rows=200, total_ms=6.0, measured=True)
    rendered = render_profile(profile, analyze=True, width=120)
    assert "205891132094649" not in rendered
    assert "200.0x under" in rendered


def test_the_critical_path_mark_is_explained_where_it_is_drawn():
    """An unexplained glyph invites the wrong reading, and both readings are actionable.

    The mark appears only on a branching plan, so the legend has to appear under exactly the
    same condition -- which is why a linear plan is asserted not to carry it.
    """
    branching = QueryProfile(
        ops=(
            OpProfile(op_id=0, kind="hash_join", depth=0, measured=True, elapsed_ms=1.0),
            OpProfile(op_id=1, kind="scan", depth=1, measured=True, elapsed_ms=9.0),
            OpProfile(op_id=2, kind="scan", depth=1, measured=True, elapsed_ms=0.1),
        ),
        rows=10,
        total_ms=10.0,
        measured=True,
    )
    rendered = render_profile(branching, analyze=True, width=120)
    assert "▶" in rendered
    assert "critical path" in rendered

    linear = render_profile(_chain(3), analyze=True, width=120)
    assert "▶" not in linear
    assert "critical path" not in linear

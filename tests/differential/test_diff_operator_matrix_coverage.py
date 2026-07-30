"""Every `RelOp` tag must appear in a cross-product matrix — checked, not asserted.

`test_diff_operator_matrix.py` and `test_diff_reshape_matrix.py` are the two files that
run an operator across *every execution path* against *every edge-case input*. Their
docstrings record five wrong-answer bugs that lived precisely in that cross-product, all
of which passed the per-operator tests. The matrices only close that gap for operators
somebody remembered to add a row for, and five tags had silently never been added.

This file is the guard. It lowers every builder in both matrices to JSON IR, collects the
``op`` tags that actually appear, and fails if any tag in `plan.ir_tags.Op` is missing.

The coverage set is **derived, not declared**: a row earns its tag by producing a plan
that contains it. A hand-maintained list of "operators we cover" is exactly the artifact
that goes stale without failing, which is the failure this file exists to prevent — so
adding a name here cannot fake coverage, and neither can renaming a test.

Adding a relational operator therefore means adding it to a matrix, or explicitly
exempting it below with a reason that says who covers it instead.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

pytestmark = pytest.mark.differential

bt = pytest.importorskip("batcher")

from test_diff_operator_matrix import (  # noqa: E402
    BASE,
    UNORDERED_OPS,
)
from test_diff_reshape_matrix import (  # noqa: E402
    ASOF_LEFT,
    ASOF_OPS,
    ASOF_RIGHT,
    INTERVALS,
    POINTS,
    RANGE_QUERIES,
    RESHAPE,
    RESHAPE_OPS,
    SAMPLE_OPS,
)

from batcher.plan.ir_tags import Op  # noqa: E402  (after the native-engine skip guard)

#: Tags no matrix row can produce, each with the reason and the file that does cover it.
#: Keep this empty where possible; an entry here is a gap held open on purpose.
EXEMPT: dict[str, str] = {}


def _ir_ops(ir: object) -> set[str]:
    """Every ``op`` tag appearing in a lowered plan."""
    found: set[str] = set()
    if isinstance(ir, dict):
        op = ir.get("op")
        if isinstance(op, str):
            found.add(op)
        for value in ir.values():
            found |= _ir_ops(value)
    elif isinstance(ir, list):
        for value in ir:
            found |= _ir_ops(value)
    return found


def _all_relop_tags() -> set[str]:
    """Every `RelOp` discriminator Python can emit, read off the wire-contract vocabulary."""
    return {
        value
        for name, value in vars(Op).items()
        if not name.startswith("_") and isinstance(value, str)
    }


def _covered_tags() -> set[str]:
    """The tags the two matrices actually exercise, by lowering each of their builders."""
    covered: set[str] = set()

    for build, _ in UNORDERED_OPS.values():
        covered |= _ir_ops(build(bt.from_arrow(BASE))._plan.to_ir())

    # The sibling matrix asserts sorts separately, since an unordered compare is blind to
    # a sort bug; that path still exercises the `sort` tag.
    covered |= _ir_ops(bt.from_arrow(BASE).sort(bt.col("k"))._plan.to_ir())

    for build, _ in RESHAPE_OPS.values():
        covered |= _ir_ops(build(bt.from_arrow(RESHAPE))._plan.to_ir())
    for build in SAMPLE_OPS.values():
        covered |= _ir_ops(build(bt.from_arrow(RESHAPE))._plan.to_ir())
    for build, _ in ASOF_OPS.values():
        covered |= _ir_ops(build(bt.from_arrow(ASOF_LEFT), bt.from_arrow(ASOF_RIGHT))._plan.to_ir())

    # The range-join rewrite is an optimizer pass, so its tag only appears after Kyber runs.
    from batcher.kyber.optimizer import optimize_logical

    for query in RANGE_QUERIES.values():
        ds = bt.sql(query, pt=POINTS, iv=INTERVALS)
        covered |= _ir_ops(optimize_logical(ds._plan).to_ir())

    return covered


def test_every_relational_operator_is_in_a_cross_product_matrix():
    """A `RelOp` with no matrix row is an operator no test runs on the spilled or streamed
    path — the exact shape of every bug the matrices were written to catch."""
    missing = _all_relop_tags() - _covered_tags() - set(EXEMPT)
    assert not missing, (
        f"relational operators with no cross-product coverage: {sorted(missing)}. "
        f"Add a row to test_diff_operator_matrix.py or test_diff_reshape_matrix.py so the "
        f"operator runs on collect / spill / iter_batches against nulls, empty and "
        f"single-row input — or add it to EXEMPT with the file that covers it instead."
    )


def test_the_exemption_list_stays_honest():
    """An exemption for a tag that *is* covered, or that no longer exists, is stale."""
    tags = _all_relop_tags()
    for tag, reason in EXEMPT.items():
        assert tag in tags, f"EXEMPT names {tag!r}, which is not a RelOp tag"
        assert reason.strip(), f"EXEMPT[{tag!r}] needs a reason naming what covers it"
    assert not (set(EXEMPT) & _covered_tags()), (
        f"exempted but actually covered, so the exemption should go: "
        f"{sorted(set(EXEMPT) & _covered_tags())}"
    )


def test_the_coverage_probe_can_actually_fail():
    """The guard is only worth having if an uncovered operator makes it red.

    A derived-coverage check that silently returns every tag would pass forever. This
    plants a tag that no matrix builds and confirms the comparison reports it.
    """
    covered = _covered_tags()
    assert "no_such_operator" not in covered
    missing = ({"no_such_operator"} | _all_relop_tags()) - covered - set(EXEMPT)
    assert missing == {"no_such_operator"}, (
        f"the probe should isolate exactly the planted tag, got {sorted(missing)}"
    )


def test_lowering_a_plan_reports_its_own_operator():
    """`_ir_ops` must read the tag it is given — a probe that returns nothing would make
    every coverage assertion above vacuously green."""
    ir = bt.from_arrow(pa.table({"a": [1]})).filter(bt.col("a") > 0)._plan.to_ir()
    ops = _ir_ops(ir)
    assert Op.FILTER in ops and Op.SCAN in ops, f"expected filter over scan, got {sorted(ops)}"

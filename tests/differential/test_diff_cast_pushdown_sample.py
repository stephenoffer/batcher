"""A narrowing cast must not be pushed below a `Sample`.

`push_down_narrowing_cast` relocates `cast(col, narrower)` into the projection that
produces the column, walking down through operators that "do not read the column". A
`Sample` names no column anywhere in the plan, so it looked transparent — but a *fraction*
sample decides each row's fate from a hash of that row's **encoded values**. Narrowing a
column from int64 to int32 underneath it changes the bytes being hashed, and so changes
which rows are sampled: the same query returned 99 rows optimized and 95 unoptimized.

An optimizer rule that changes the answer is the worst kind of bug, and it is invisible to
any test that does not put a *value-changing* rewrite underneath a *value-reading*
operator. That cross-product is what this pins.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, core
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.registry import DEFAULT_REGISTRY

pytestmark = pytest.mark.differential

_RULE = "push_down_narrowing_cast"


def _without_the_rule(ds) -> pa.Table:
    """The same query planned with the cast-pushdown rule removed — the oracle."""
    rules = [r for r in DEFAULT_REGISTRY.rules() if r.name != _RULE]
    phys = Optimizer(sources=ds._sources, rules=rules).optimize(ds._plan)
    batches = core.execute_local(phys, [s.read() for s in ds._sources])
    return pa.Table.from_batches(batches) if batches else pa.table({})


@pytest.fixture
def base():
    return bt.from_arrow(pa.table({"v": pa.array(list(range(2000)), type=pa.int64())}))


def test_narrowing_cast_below_a_fraction_sample_does_not_change_the_rows(base):
    """The producing projection sits below the sample, so the rule tries to push into it."""
    q = (
        base.select(v=col("v") + 1)  # the producer the cast would be folded into
        .sample(fraction=0.05, seed=1)  # samples on the row's *encoded* values
        .select(v=col("v").cast("int32"))
    )
    optimized = q.collect()
    reference = _without_the_rule(q)
    assert optimized.num_rows == reference.num_rows
    assert optimized.to_pydict() == reference.to_pydict()


def test_narrowing_cast_below_a_fixed_count_sample_does_not_change_the_rows(base):
    q = base.select(v=col("v") + 1).sample(n=100, seed=7).select(v=col("v").cast("int32"))
    optimized = q.collect()
    reference = _without_the_rule(q)
    assert optimized.num_rows == reference.num_rows
    assert optimized.to_pydict() == reference.to_pydict()


def test_the_cast_still_pushes_below_a_limit(base):
    """`Limit` really is transparent (a positional prefix) — the rule must keep firing."""
    from batcher.plan.expr_ir import Cast
    from batcher.plan.logical import Project
    from batcher.plan.visitor import walk

    q = base.select(v=col("v") + 1).limit(50).select(v=col("v").cast("int32"))
    plan = Optimizer(sources=q._sources).optimize_full(q._plan)[1]

    # The cast was folded into the producing projection *below* the limit, so the topmost
    # projection no longer carries one.
    projects = [n for n in walk(plan) if isinstance(n, Project)]
    assert any(any(isinstance(it.expr, Cast) for it in p.items) for p in projects), (
        "the cast should still be pushed into a producer through a Limit"
    )
    assert q.collect().to_pydict() == _without_the_rule(q).to_pydict()

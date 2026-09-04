"""A refusal must say which of the two things happened, and not the other one.

`dist.executor._unsupported` refuses rather than silently running a whole query on one node.
Two quite different situations reach it, and a caller can only act on the difference:

* the shape **has** a distributed path -- the staged one -- and it did not stage here. Either
  the caller forced `adaptive=False`, or staging is already on and some *stage* of the plan
  has no decomposition of its own. `_unsupported` cannot tell those apart, so the message
  names both remedies rather than asserting one: an earlier draft told a caller who was
  already staging to "re-run with `adaptive=True`", which is worse than saying less.
* the shape genuinely has no distributed decomposition, such as an unpartitioned `lag`, whose
  ordered bucket would have to read rows it does not hold. The fix is to file the operator, or
  run single-node deliberately.

The message used to open with "distributed execution has no path for this plan shape (an
unsupported operator combination)" in **both** cases, and only then, in the tail, explain that
the first kind distributes stage by stage after all. A headline that contradicts its own
remedy, and precisely the implication the code comment beside it says not to make. A reader
who stops at the first clause -- which is where a reader stops -- concludes the query cannot be
distributed at all, and goes looking for a missing operator that is not missing.

Three ordinary shapes hit it: a nested aggregate, a join followed by a window, and two global
windows. None is exotic and all three distribute perfectly well.

Ray-free: `_unsupported` decides from the plan and the sources, so calling it directly needs
no cluster, and the message is what is under test rather than the execution.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.dist.executor import _unsupported
from batcher.dist.executors.plan_analysis import requires_staging

pytestmark = pytest.mark.unit

#: The reason string the dispatcher passes for a shape it did not otherwise recognise. It is
#: deliberately uninformative, which is why the staged branch must not repeat it.
_GENERIC = "an unsupported operator combination"


@pytest.fixture(scope="module")
def splittable(tmp_path_factory):
    """Real splittable sources -- `_unsupported` runs in-memory plans single-node by design."""
    left = pa.table({"k": ["a", "b", "a", "c"] * 10, "v": list(range(40))})
    right = pa.table({"k": ["a", "b", "c"], "w": [10, 20, 30]})
    ldir = tmp_path_factory.mktemp("l")
    rdir = tmp_path_factory.mktemp("r")
    for part in range(4):
        pq.write_table(left, ldir / f"p{part}.parquet")
    pq.write_table(right, rdir / "r.parquet")
    return str(ldir), str(rdir)


def _staged_shape(name: str, ldir: str, rdir: str):
    ds = bt.read.parquet(ldir)
    if name == "nested_aggregate":
        return ds.group_by("k").agg(s=bt.col("v").sum()).group_by("k").agg(m=bt.col("s").max())
    if name == "join_then_window":
        return ds.join(bt.read.parquet(rdir), on="k").with_columns(
            r=bt.row_number().over(partition_by="k", order_by="v")
        )
    if name == "two_global_windows":
        return ds.with_columns(
            a=bt.row_number().over(order_by="v"), b=bt.lag(bt.col("v"), 1).over(order_by="v")
        )
    raise AssertionError(name)


_STAGED = ("join_then_window", "nested_aggregate", "two_global_windows")


@pytest.mark.parametrize("shape", _STAGED)
def test_a_staged_shape_is_not_reported_as_having_no_path(shape, splittable):
    ds = _staged_shape(shape, *splittable)
    assert requires_staging(ds._plan), (
        f"{shape} no longer requires staging, so it is the wrong fixture for this test"
    )

    with pytest.raises(PlanError) as raised:
        _unsupported(ds._plan, ds._sources, _GENERIC)
    message = str(raised.value)

    assert "has no path for this plan shape" not in message, (
        f"{shape} distributes stage by stage, but the refusal claims there is no path -- a "
        "reader stopping at the first clause concludes the operator is missing"
    )
    assert _GENERIC not in message, (
        "the staged branch repeated the dispatcher's placeholder reason, which says nothing"
    )
    assert "stage by stage" in message, f"the refusal must name the mechanism; got: {message}"
    # Both remedies, because `_unsupported` cannot tell which caller it has: one forced
    # `adaptive=False`, and one is already staging but has a sub-stage with no decomposition
    # (what `batcher.graph`'s refusing algorithms hit). Naming only the first tells the
    # second caller to do what they already did.
    assert "adaptive=False" in message and "materializing the intermediate" in message, (
        f"the refusal must cover both callers without asserting which one it has; got: {message}"
    )


def test_a_shape_with_genuinely_no_path_still_says_so(splittable):
    """The other branch, and the reason this file is not just asserting a string was deleted.

    Removing "has no path" everywhere would pass the tests above and lose the one case where
    it is the true and useful thing to say.
    """
    ldir, _rdir = splittable
    ds = bt.read.parquet(ldir).with_columns(p=bt.lag(bt.col("v"), 1).over(order_by="v"))
    assert not requires_staging(ds._plan)

    reason = "a global window (no PARTITION BY) over lag"
    with pytest.raises(PlanError) as raised:
        _unsupported(ds._plan, ds._sources, reason)
    message = str(raised.value)

    assert "has no path for this plan shape" in message
    assert reason in message, "the caller's specific reason must survive into the message"
    assert "adaptive=True" not in message, (
        "re-enabling staging is not the remedy for a shape with no decomposition"
    )


def test_an_in_memory_plan_is_not_refused_at_all(splittable):
    """The third outcome, which neither branch above covers.

    With no splittable source there is no distributed data, so running on one node is the
    right answer rather than a fallback -- `_unsupported` must return a result, not raise.
    Without this, both assertions above would still pass if `_unsupported` had been changed
    to raise unconditionally.
    """
    ds = bt.from_arrow(pa.table({"k": ["a", "b"], "v": [1, 2]}))
    nested = ds.group_by("k").agg(s=bt.col("v").sum()).group_by("s").agg(n=bt.col("s").count())
    out = _unsupported(nested._plan, nested._sources, _GENERIC)
    assert out.num_rows >= 1

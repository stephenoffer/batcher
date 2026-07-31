"""The translator must only use operations cuDF actually implements, not pandas' superset.

Every other GPU test runs the kernels on **pandas**, which is what makes them runnable in CI —
and is also a hole exactly the size of the difference between the two libraries' APIs. cuDF is
a subset, and the gaps do not raise on the verification backend, so a kernel can be green here
and decline on every device.

That is not hypothetical. `sum` and `product` were computed with pandas' `min_count=1`, which
is how SQL's "a sum over nothing is null, not zero" is spelled there. **cuDF has no `min_count`
and raises `NotImplementedError` for it.** So every `sum` on a device raised, the shard fell
back to the CPU engine, and — since `sum` is in essentially every analytical query — the GPU
backend was doing round trips to devices that never computed anything. Measured on this
cluster: TPC-H q6 took 13.5 s to produce an answer the host had in 106 ms, and every millisecond
of it was dispatch to a device that declined.

These tests close the hole by making the host backend *refuse what cuDF refuses*. The kernels
then have to be written against the intersection of the two libraries, which is the contract
they always claimed to honor.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher.core.gpu_plan import DfBackend, gpu_plan_ops
from batcher.core.gpu_plan.execute import run_chain

pytestmark = pytest.mark.unit

#: Keyword arguments pandas' `GroupBy` reductions accept and cuDF's do not. Passing one is a
#: `NotImplementedError` on a device and a silent success here, which is the whole problem.
CUDF_ABSENT_GROUPBY_KWARGS = ("min_count",)


@pytest.fixture
def strict_be(monkeypatch):
    """A pandas backend whose `GroupBy` rejects what cuDF's rejects.

    Patches the reductions themselves rather than wrapping the frame, so the check applies
    wherever the kernels reach them — including through code paths a wrapper would not see.
    """
    import pandas as pd
    from pandas.core.groupby.generic import SeriesGroupBy

    for name in ("sum", "prod", "min", "max", "mean", "median"):
        original = getattr(SeriesGroupBy, name)

        def guarded(self, *args, __original=original, __name=name, **kwargs):
            for absent in CUDF_ABSENT_GROUPBY_KWARGS:
                if absent in kwargs:
                    raise NotImplementedError(f"{absent} parameter is not implemented yet")
            return __original(self, *args, **kwargs)

        monkeypatch.setattr(SeriesGroupBy, name, guarded)
    return DfBackend(pd)


def _run(build, table, be):
    ds = build(bt.from_arrow(table))
    spec = gpu_plan_ops(ds._plan)
    assert spec is not None, "shape should be GPU-translatable"
    return be.to_arrow(run_chain(table, spec[1], be)), ds.collect()


def _rows(table: pa.Table):
    """Rows as comparable tuples, floats rounded.

    The two paths sum a group in different orders and floating-point addition is not
    associative, so the last ulp legitimately differs. Everything that is not a float is
    compared exactly.
    """
    cols = table.to_pydict()

    def canon(v):
        return round(v, 10) if isinstance(v, float) else v

    return sorted(
        [tuple(canon(cols[c][i]) for c in table.column_names) for i in range(table.num_rows)],
        key=repr,
    )


def _table():
    rng = np.random.default_rng(0)
    return pa.table(
        {
            "g": rng.integers(0, 5, 200).astype("int64"),
            "v": rng.random(200),
            "n": pa.array([None if i % 3 == 0 else float(i) for i in range(200)], pa.float64()),
        }
    )


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda ds: ds.group_by("g").agg(s=col("v").sum()), id="sum"),
        pytest.param(lambda ds: ds.group_by("g").agg(s=col("n").sum()), id="sum-with-nulls"),
        pytest.param(lambda ds: ds.agg(s=col("v").sum()), id="global-sum"),
        pytest.param(lambda ds: ds.group_by("g").agg(m=col("v").min()), id="min"),
        pytest.param(lambda ds: ds.group_by("g").agg(m=col("v").max()), id="max"),
        pytest.param(lambda ds: ds.group_by("g").agg(a=col("v").mean()), id="mean"),
    ],
)
def test_the_common_reductions_run_within_cudfs_api(build, strict_be):
    """Each of these must compute, not raise — and must still equal the CPU engine.

    `sum` is the one that was broken, and the parametrization is the point: a fix that made
    `sum` avoid `min_count` while leaving `product` on it would pass a single-case test.
    """
    got, exp = _run(build, _table(), strict_be)
    assert _rows(got.select(exp.column_names)) == _rows(exp)


def test_a_sum_over_an_all_null_group_is_null_not_zero(strict_be):
    """The rule `min_count=1` was there for, still enforced without it.

    A group whose values are every one of them null has summed nothing, and SQL says the answer
    is unknown. `0.0` is not unknown — it is a measurement, and it is wrong.
    """
    table = pa.table(
        {
            "g": pa.array([1, 1, 2, 2], pa.int64()),
            "v": pa.array([None, None, 3.0, 4.0], pa.float64()),
        }
    )
    got, exp = _run(lambda ds: ds.group_by("g").agg(s=col("v").sum()), table, strict_be)
    assert _rows(got.select(exp.column_names)) == _rows(exp)
    by_group = dict(zip(got.column("g").to_pylist(), got.column("s").to_pylist(), strict=True))
    assert by_group[1] is None, "an all-null group sums to null, not to the identity"
    assert by_group[2] == 7.0


def test_a_product_over_an_all_null_group_is_null_not_one(strict_be):
    """The same rule for the other reduction that had a `min_count`."""
    table = pa.table(
        {
            "g": pa.array([1, 1, 2], pa.int64()),
            "v": pa.array([None, None, 5.0], pa.float64()),
        }
    )
    got, _exp = _run(lambda ds: ds.group_by("g").agg(p=col("v").product()), table, strict_be)
    by_group = dict(zip(got.column("g").to_pylist(), got.column("p").to_pylist(), strict=True))
    assert by_group[1] is None
    assert by_group[2] == 5.0


def test_the_guard_itself_would_have_caught_the_bug(strict_be):
    """The fixture must actually refuse `min_count`, or every test above passes vacuously."""
    import pandas as pd

    grouped = pd.DataFrame({"g": [1, 1], "v": [1.0, 2.0]}).groupby("g")["v"]
    with pytest.raises(NotImplementedError, match="min_count"):
        grouped.sum(min_count=1)
    # ...and without it, the same call works, so the fixture bans the parameter and not the
    # reduction.
    assert grouped.sum().iloc[0] == 3.0

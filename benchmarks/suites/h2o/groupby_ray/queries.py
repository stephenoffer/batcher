"""The ten h2o ``groupby`` questions as Ray Data pipelines.

Each is the benchmark's own question, expressed with Ray Data's native ``groupby`` and
aggregates. Where Ray Data has no aggregate for what the question asks -- q7's
``max(v1) - min(v2)`` and q9's ``corr`` -- the shuffle still runs in Ray and only the
arithmetic over the (small) grouped result is done here.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
from ray.data.aggregate import Count, Max, Mean, Min, Quantile, Std, Sum

from suites.h2o.groupby_ray.base import impl, rename, to_arrow


def _agg(ds: Any, keys: list[str] | str, *aggs: Any) -> pa.Table:
    return to_arrow(ds.groupby(keys).aggregate(*aggs))


@impl("h2o-gb-q1")
def q1(t: dict[str, Any]) -> pa.Table:
    """``SELECT id1, sum(v1) FROM x GROUP BY id1``."""
    return rename(_agg(t["x"], "id1", Sum("v1")), {"id1": "id1", "sum(v1)": "v1"})


@impl("h2o-gb-q2")
def q2(t: dict[str, Any]) -> pa.Table:
    """``SELECT id1, id2, sum(v1) FROM x GROUP BY id1, id2``."""
    got = _agg(t["x"], ["id1", "id2"], Sum("v1"))
    return rename(got, {"id1": "id1", "id2": "id2", "sum(v1)": "v1"})


@impl("h2o-gb-q3")
def q3(t: dict[str, Any]) -> pa.Table:
    """``SELECT id3, sum(v1), avg(v3) FROM x GROUP BY id3``."""
    got = _agg(t["x"], "id3", Sum("v1"), Mean("v3"))
    return rename(got, {"id3": "id3", "sum(v1)": "v1", "mean(v3)": "v3"})


@impl("h2o-gb-q4")
def q4(t: dict[str, Any]) -> pa.Table:
    """``SELECT id4, avg(v1), avg(v2), avg(v3) FROM x GROUP BY id4``."""
    got = _agg(t["x"], "id4", Mean("v1"), Mean("v2"), Mean("v3"))
    return rename(got, {"id4": "id4", "mean(v1)": "v1", "mean(v2)": "v2", "mean(v3)": "v3"})


@impl("h2o-gb-q5")
def q5(t: dict[str, Any]) -> pa.Table:
    """``SELECT id6, sum(v1), sum(v2), sum(v3) FROM x GROUP BY id6``."""
    got = _agg(t["x"], "id6", Sum("v1"), Sum("v2"), Sum("v3"))
    return rename(got, {"id6": "id6", "sum(v1)": "v1", "sum(v2)": "v2", "sum(v3)": "v3"})


@impl("h2o-gb-q6")
def q6(t: dict[str, Any]) -> pa.Table:
    """``SELECT id4, id5, median(v3), stddev(v3) FROM x GROUP BY id4, id5``.

    ``Quantile(q=0.5)`` is Ray Data's median; ``Std`` defaults to the sample standard
    deviation (``ddof=1``), which is what SQL's ``stddev`` is.
    """
    got = _agg(t["x"], ["id4", "id5"], Quantile("v3", q=0.5), Std("v3"))
    return rename(
        got,
        {
            "id4": "id4",
            "id5": "id5",
            "quantile(v3)": "median_v3",
            "std(v3)": "sd_v3",
        },
    )


@impl("h2o-gb-q7")
def q7(t: dict[str, Any]) -> pa.Table:
    """``SELECT id3, max(v1) - min(v2) FROM x GROUP BY id3``.

    Ray Data has no expression surface over aggregate outputs, so the subtraction runs on
    the grouped result -- one row per ``id3``, not per input row.
    """
    got = _agg(t["x"], "id3", Max("v1"), Min("v2"))
    diff = pc.subtract(got.column("max(v1)"), got.column("min(v2)"))
    return pa.table({"id3": got.column("id3"), "range_v1_v2": diff})


@impl("h2o-gb-q8")
def q8(t: dict[str, Any]) -> pa.Table:
    """The largest two ``v3`` per ``id6``.

    The benchmark writes this as a ``row_number()`` window filtered to ``<= 2``. Ray Data
    has no window functions, so the equivalent is a grouped top-2, which is what
    ``map_groups`` expresses: the shuffle is still Ray's.
    """

    def top2(batch: pa.Table) -> pa.Table:
        order = pc.sort_indices(batch, sort_keys=[("v3", "descending")])
        return batch.take(order[:2]).select(["id6", "v3"])

    ds = t["x"].filter(lambda r: r["v3"] is not None)
    got = to_arrow(ds.groupby("id6").map_groups(top2, batch_format="pyarrow"))
    return rename(got, {"id6": "id6", "v3": "largest2_v3"})


@impl("h2o-gb-q9")
def q9(t: dict[str, Any]) -> pa.Table:
    """``SELECT id2, id4, pow(corr(v1, v2), 2) FROM x GROUP BY id2, id4``.

    Ray Data has no correlation aggregate, so the group's sufficient statistics are
    aggregated natively and Pearson's r is formed from them afterwards -- the per-row work
    stays in Ray, and only the 10,000-row grouped result is touched here.

    The count comes from the **same** aggregate rather than a second `groupby().count()`.
    Two aggregations need not return their groups in the same order, and combining them by
    position would pair each group's sums with another group's count -- a wrong answer that
    the row count and the column names both agree with.
    """
    import numpy as np

    def moments(batch: pa.Table) -> pa.Table:
        v1 = batch.column("v1").cast("float64")
        v2 = batch.column("v2").cast("float64")
        return pa.table(
            {
                "id2": batch.column("id2"),
                "id4": batch.column("id4"),
                "v1": v1,
                "v2": v2,
                "xy": pc.multiply(v1, v2),
                "xx": pc.multiply(v1, v1),
                "yy": pc.multiply(v2, v2),
            }
        )

    ds = t["x"].map_batches(moments, batch_format="pyarrow")
    got = _agg(
        ds,
        ["id2", "id4"],
        Sum("v1"),
        Sum("v2"),
        Sum("xy"),
        Sum("xx"),
        Sum("yy"),
        Count(),
    )
    n = np.asarray(got.column("count()"), dtype="float64")
    sx = np.asarray(got.column("sum(v1)"), dtype="float64")
    sy = np.asarray(got.column("sum(v2)"), dtype="float64")
    sxy = np.asarray(got.column("sum(xy)"), dtype="float64")
    sxx = np.asarray(got.column("sum(xx)"), dtype="float64")
    syy = np.asarray(got.column("sum(yy)"), dtype="float64")
    cov = sxy - sx * sy / n
    vx = sxx - sx * sx / n
    vy = syy - sy * sy / n
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where((vx > 0) & (vy > 0), (cov * cov) / (vx * vy), np.nan)
    return pa.table({"id2": got.column("id2"), "id4": got.column("id4"), "r2": pa.array(r2)})


@impl("h2o-gb-q10")
def q10(t: dict[str, Any]) -> pa.Table:
    """``GROUP BY id1..id6`` with ``sum(v3)`` and ``count(*)`` -- the near-unique-key case.

    Both aggregates come from **one** pass. The obvious spelling -- aggregate, count
    separately, then join the two on the six keys -- is a trap at this cardinality: the keys
    are near-unique over 10 M rows, so the join is over ~10 M groups and costs more than the
    aggregate it is joining. `Count` is an aggregate like any other; ask for it alongside.
    """
    keys = ["id1", "id2", "id3", "id4", "id5", "id6"]
    got = to_arrow(t["x"].groupby(keys).aggregate(Sum("v3"), Count()))
    mapping = {k: k for k in keys}
    mapping["sum(v3)"] = "v3"
    mapping["count()"] = "count"
    return rename(got, mapping)

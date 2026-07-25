"""The streaming and materializing executors must return the same rows.

`execution.streaming` chooses between two executors, and its own comment calls turning it off
"a bisecting escape hatch, not a tuning knob" -- which is only true if the two agree on every
answer. Nothing checked that. The same applies to `execution.shrink_output_dtypes`, documented
as lossless but data-dependent: it re-narrows output columns to their source width, so a value
that survived the round trip incorrectly would be a wrong answer rather than a wrong dtype.

Both are two paths computing the same thing, which is the shape that turned up the `hex`
constant-fold bug: each side is plausible alone and only the comparison finds a divergence.

Sorts are compared in order, because an order-independent comparison cannot see a sort bug.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher.config import Config, config_context

pytestmark = pytest.mark.differential

_N = 3000


@pytest.fixture(scope="module")
def table() -> pa.Table:
    rng = np.random.default_rng(71)
    return pa.table(
        {
            "k": pa.array(rng.integers(0, 40, _N).astype("int64"), pa.int64()),
            "g": pa.array(rng.integers(0, 5, _N).astype("int64"), pa.int64()),
            "v": pa.array(rng.normal(0, 100, _N), pa.float64()),
            "narrow": pa.array(rng.integers(-100, 100, _N).astype("int32"), pa.int32()),
            "s": pa.array([f"row{i % 97}" for i in range(_N)]),
            "nl": pa.array([None if i % 13 == 0 else int(i % 7) for i in range(_N)], pa.int64()),
        }
    )


_QUERIES = {
    "filter": lambda d: d.filter(bt.col("k") > bt.lit(20)),
    "project": lambda d: d.select("k", "v", w=bt.col("v") * bt.lit(2.0)),
    "group_by": lambda d: d.group_by("g").agg(t=bt.col("v").sum(), n=bt.col("k").count()),
    "group_by wide": lambda d: d.group_by("k").agg(
        t=bt.col("v").sum(), m=bt.col("v").mean(), lo=bt.col("v").min(), hi=bt.col("v").max()
    ),
    "group_by null key": lambda d: d.group_by("nl").agg(n=bt.col("k").count()),
    "global agg": lambda d: d.agg(
        t=bt.col("v").sum(), n=bt.col("k").count(), u=bt.col("k").n_unique()
    ),
    "distinct": lambda d: d.select("k", "g").distinct(),
    "sort asc": lambda d: d.sort("k", "v"),
    "sort desc": lambda d: d.sort("k", descending=True),
    "sort then limit": lambda d: d.sort("v", descending=True).limit(50),
    "limit": lambda d: d.limit(37),
    "union": lambda d: d.select("k", "g").union(d.select("k", "g")),
    "inner join": lambda d: d.select("k", "v").join(
        d.select("k", g=bt.col("g")), on="k", how="inner"
    ),
    "left join": lambda d: d.select("k", "v").join(
        d.filter(bt.col("k") > bt.lit(30)).select("k", g2=bt.col("g")), on="k", how="left"
    ),
    "window": lambda d: d.select("g", "v", r=bt.col("v").sum().over(partition_by=["g"])),
    "having": lambda d: d.group_by("g").agg(t=bt.col("v").sum()).filter(bt.col("t") > bt.lit(0.0)),
    "narrow passthrough": lambda d: d.select("narrow", "k"),
    "string group": lambda d: d.group_by("s").agg(n=bt.col("k").count()),
}


def _run(table: pa.Table, build, **execution):
    base = Config()
    cfg = base.replace(execution=replace(base.execution, **execution))
    with config_context(cfg):
        return build(bt.from_arrow(table)).collect()


def _rows(result) -> tuple[list[tuple], list[str]]:
    data = result.to_pydict()
    names = sorted(data)
    return [tuple(row) for row in zip(*[data[n] for n in names], strict=True)], names


def _canonical(rows: list[tuple], *, ordered: bool) -> list[tuple]:
    if ordered:
        return rows

    def key(row):
        return tuple(
            (
                v is None,
                "" if v is None or isinstance(v, float) else repr(v),
                0.0 if v is None or not isinstance(v, float) or math.isnan(v) else v,
            )
            for v in row
        )

    return sorted(rows, key=key)


def _assert_same_rows(left: list[tuple], right: list[tuple], message: str) -> None:
    assert len(left) == len(right), f"{message}: {len(left)} rows against {len(right)}"
    for index, (a, b) in enumerate(zip(left, right, strict=True)):
        for x, y in zip(a, b, strict=True):
            if x is None or y is None:
                assert x is None and y is None, f"{message}: row {index}, {x!r} against {y!r}"
            elif isinstance(x, float) and isinstance(y, float):
                if math.isnan(x) and math.isnan(y):
                    continue
                assert x == pytest.approx(y, rel=1e-9, abs=1e-12), f"{message}: row {index}"
            else:
                assert x == y, f"{message}: row {index}, {x!r} against {y!r}"


@pytest.mark.parametrize("query", sorted(_QUERIES))
def test_the_two_executors_agree(table, query) -> None:
    """`streaming=False` is documented as a bisecting aid, so it must not change the answer."""
    build = _QUERIES[query]
    ordered = query.startswith("sort")

    streamed, names_a = _rows(_run(table, build, streaming=True))
    materialized, names_b = _rows(_run(table, build, streaming=False))

    assert names_a == names_b, f"{query}: the two executors returned different columns"
    _assert_same_rows(
        _canonical(streamed, ordered=ordered),
        _canonical(materialized, ordered=ordered),
        f"{query}: streaming and materializing disagree",
    )


@pytest.mark.parametrize("query", sorted(_QUERIES))
def test_shrinking_output_dtypes_does_not_change_a_value(table, query) -> None:
    """Re-narrowing a column to its source width is documented as lossless in values."""
    build = _QUERIES[query]
    ordered = query.startswith("sort")

    wide, names_a = _rows(_run(table, build, shrink_output_dtypes=False))
    narrow, names_b = _rows(_run(table, build, shrink_output_dtypes=True))

    assert names_a == names_b, f"{query}: shrinking changed the column set"
    _assert_same_rows(
        _canonical(wide, ordered=ordered),
        _canonical(narrow, ordered=ordered),
        f"{query}: shrink_output_dtypes changed a value",
    )

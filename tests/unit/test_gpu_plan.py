"""The cuDF-plan translator matches the native CPU engine (verified on pandas, no GPU needed).

`gpu_plan_ops` detects a supported linear chain; `_execute_df_plan` replays its RelOp/Expr IR on
a dataframe. Run against pandas here so the translation logic is CI-testable without a GPU; the
GPU path runs the identical `_execute_df_plan` on cuDF. Unsupported shapes must return `None`
(→ CPU fallback), never a wrong answer.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher.core.gpu_plan import _execute_df_plan, gpu_plan_ops

pytestmark = pytest.mark.unit


def _table():
    rng = np.random.default_rng(0)
    return pa.table(
        {
            "x": rng.integers(0, 10, 300).astype("int64"),
            "y": rng.random(300),
            "z": rng.integers(0, 3, 300).astype("int64"),
        }
    )


def _norm(df):
    return df.sort_values(list(df.columns)).reset_index(drop=True).round(6)


@pytest.mark.parametrize(
    "build",
    [
        lambda ds: ds.filter(col("y") > 0.5),
        lambda ds: ds.with_columns(w=col("y") * 2.0 + 1.0),
        lambda ds: ds.select("x", "y"),
        lambda ds: ds.filter(col("x") > 3).with_columns(w=col("y") - col("x")),
        lambda ds: ds.group_by("x", "z").agg(
            s=col("y").sum(), c=col("y").count(), m=col("y").mean()
        ),
        lambda ds: ds.sort("y", descending=True).limit(10),
        lambda ds: ds.select("x", "z").distinct(),
        lambda ds: ds.filter(col("y") > 0.3).group_by("x").agg(s=col("y").sum(), mx=col("y").max()),
    ],
)
def test_translator_matches_cpu_engine(build):
    import pandas as pd

    t = _table()
    ds = build(bt.from_arrow(t))
    spec = gpu_plan_ops(ds._plan)
    assert spec is not None, "shape should be GPU-executable"
    _scan, ops = spec
    got = _execute_df_plan(t, ops, pd)
    exp = ds.collect().to_pandas()
    assert _norm(got[exp.columns]).equals(_norm(exp))


def test_unsupported_shapes_return_none():
    t = _table()
    ds = bt.from_arrow(t)
    # a join is two-source -> unsupported here
    assert gpu_plan_ops(ds.join(bt.from_arrow(t), on="x")._plan) is None
    # a map_batches UDF -> unsupported
    assert gpu_plan_ops(ds.map_batches(lambda b: b)._plan) is None
    # a group key that is an EXPRESSION (not a plain column) -> unsupported (safe fallback)
    assert gpu_plan_ops(ds.group_by(x2=col("x") + 1).agg(s=col("y").sum())._plan) is None


@pytest.mark.parametrize(
    "build",
    [
        lambda a, b: a.join(b, on="id", how="inner"),
        lambda a, b: a.join(b, on="id", how="left"),
        lambda a, b: a.join(b, on="id", how="inner").filter(col("w") > 100),
        lambda a, b: a.join(b, on="id", how="inner").group_by("w").agg(s=col("v").sum()),
    ],
)
def test_join_translator_matches_cpu_engine(build):
    import pandas as pd

    from batcher.core.gpu_plan import _execute_join_plan, gpu_join_spec

    fact = pa.table({"id": np.array([1, 2, 3, 1, 2], "int64"), "v": np.array([1.0, 2, 3, 4, 5])})
    dim = pa.table({"id": np.array([1, 2, 3], "int64"), "w": np.array([100, 200, 300], "int64")})
    ds = build(bt.from_arrow(fact), bt.from_arrow(dim))
    spec = gpu_join_spec(ds._plan)
    assert spec is not None
    ls, rs, jir, ops = spec
    lt = fact if ls.source_id == 0 else dim
    rt = dim if rs.source_id == 1 else fact
    got = _execute_join_plan(lt, rt, jir, ops, pd)
    exp = ds.collect().to_pandas()
    assert _norm(got[exp.columns]).equals(_norm(exp))

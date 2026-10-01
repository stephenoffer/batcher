"""Repeated subplans shared past the old size thresholds still return what DuckDB returns.

`tests/unit/test_reuse_scale_thresholds.py` pins the three decisions; this file runs the
shapes they decide on and holds each answer against DuckDB, with a control showing the rewrite
really fired -- every assertion about the answer would also pass on a rewrite that declined.

* a CTE referenced twice at the head of a comma join (TPC-DS q59), where the shared subtree
  must be the CTE and not the cross product as written;
* a windowed CTE read three times (TPC-DS q47);
* a ROLLUP whose shared finest aggregate is over the fixed cap (TPC-DS q67), held because the
  budget now rises with the memory budget.

The fixtures are 120,000 rows, past `MIN_ROWS_TO_SHARD` (65,536), so the parallel executor
really shards them; nulls sit in the grouping and join keys.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, assert_same_for_query
from batcher import core
from batcher.api.subplan_reuse import reuse_common_subplans
from batcher.config import Config, config_context
from batcher.plan.logical import Aggregate
from batcher.plan.visitor import walk

pytestmark = pytest.mark.differential

_N = 120_000


def _tables() -> dict[str, pa.Table]:
    rng = np.random.default_rng(11)
    k = rng.integers(0, 300, _N).astype("float64")
    k[rng.random(_N) < 0.02] = np.nan  # nulls in the join key
    s = rng.integers(0, 20, _N).astype("float64")
    s[rng.random(_N) < 0.02] = np.nan  # and in a grouping key
    fact = pa.table(
        {
            "k": pa.array(k, pa.float64(), from_pandas=True).cast(pa.int64()),
            "s": pa.array(s, pa.float64(), from_pandas=True).cast(pa.int64()),
            "v": pa.array(rng.integers(-50, 100, _N).astype("float64")),
        }
    )
    dim = pa.table({"dk": np.arange(300), "dw": np.arange(300) // 7, "dm": np.arange(300) % 12})
    names = [f"s{i % 7}" if i % 5 else None for i in range(20)]  # duplicates and nulls
    st = pa.table({"sk": np.arange(20), "name": pa.array(names, pa.string())})
    return {"fact": fact, "dim": dim, "st": st}


@pytest.fixture(scope="module")
def tables():
    return _tables()


def _session(tables) -> bt.Session:
    s = bt.Session()
    for name, t in tables.items():
        s.register(name, t)
    return s


def _register(duck, tables) -> None:
    for name, t in tables.items():
        duck.register(name, t)


def _rewritten(ds):
    ctx = core.ExecutionContext(columns=ds.columns, hub=core.default_hub())
    return reuse_common_subplans(ds._plan, ds._sources, ctx)


# TPC-DS q59: `w` is referenced twice as `FROM w, st, dim d`, its join condition in the WHERE.
_COMMA_CTE = """
WITH w AS (SELECT dw, s, sum(v) tot, count(*) n FROM fact, dim WHERE dk = k GROUP BY dw, s)
SELECT a.name, a.dw, a.tot, b.tot AS tot2, a.n FROM
  (SELECT name, w.dw, tot, n FROM w, st, dim d WHERE d.dw = w.dw AND s = sk AND d.dk < 150) a,
  (SELECT name, w.dw, tot FROM w, st, dim d WHERE d.dw = w.dw AND s = sk AND d.dk >= 150) b
WHERE a.name = b.name AND a.dw = b.dw - 5
"""


def test_a_cte_shared_across_comma_joins_matches_duckdb(duck, tables):
    _register(duck, tables)
    assert_same(_session(tables).sql(_COMMA_CTE).collect(), duck.sql(_COMMA_CTE))


def test_the_cte_itself_is_what_gets_materialized(tables):
    """The control: the shared subtree is the CTE's aggregate, so none is left to recompute."""
    ds = _session(tables).sql(_COMMA_CTE)
    assert sum(isinstance(n, Aggregate) for n in walk(ds._plan)) == 2, "the CTE, read twice"
    plan, sources = _rewritten(ds)
    assert len(sources) > len(ds._sources), "nothing was materialized"
    assert not any(isinstance(n, Aggregate) for n in walk(plan)), (
        "the CTE's aggregate is still in the plan, so it runs once per reference"
    )


# TPC-DS q47: a windowed CTE read three times, the reads joined on a rank offset.
_WINDOWED_CTE = """
WITH v1 AS (
  SELECT s, dm, sum(v) sv,
         avg(sum(v)) OVER (PARTITION BY s) av,
         rank() OVER (PARTITION BY s ORDER BY dm) rn
  FROM fact, dim WHERE dk = k GROUP BY s, dm)
SELECT v1.s, v1.dm, v1.sv, v1.av, lag.sv psv, lead.sv nsv
FROM v1, v1 lag, v1 lead
WHERE v1.s = lag.s AND v1.s = lead.s AND v1.rn = lag.rn + 1 AND v1.rn = lead.rn - 1
ORDER BY v1.s NULLS FIRST, v1.dm
"""


def test_a_windowed_cte_read_three_times_matches_duckdb(duck, tables):
    _register(duck, tables)
    got = _session(tables).sql(_WINDOWED_CTE).collect()
    assert got.num_rows > 100, "the fixture must produce a non-trivial answer"
    assert_same_for_query(got, duck.sql(_WINDOWED_CTE), _WINDOWED_CTE)


# TPC-DS q67: a ROLLUP over a join, every level rolled up from one shared finest aggregate.
_ROLLUP = """
SELECT st.name, d.dw, d.dm, f.s, sum(f.v) sv, count(f.v) nv, min(f.v) lo
FROM fact f, dim d, st
WHERE f.k = d.dk AND f.s = st.sk
GROUP BY ROLLUP(st.name, d.dw, d.dm, f.s)
"""


def _budgeted(*, fraction: float) -> Config:
    """A 1 KiB fixed cap, far under the shared aggregate, with the memory share on or off."""
    cfg = Config()
    return cfg.replace(
        memory=dataclasses.replace(cfg.memory, max_memory_bytes=8 << 30),
        optimizer=dataclasses.replace(
            cfg.optimizer, common_subplan_max_bytes=1024, common_subplan_memory_fraction=fraction
        ),
    )


def test_a_rollup_over_the_fixed_cap_matches_duckdb(duck, tables):
    _register(duck, tables)
    with config_context(_budgeted(fraction=1 / 16)):
        got = _session(tables).sql(_ROLLUP).collect()
    assert_same(got, duck.sql(_ROLLUP))


def test_the_memory_share_is_what_lets_the_rollup_share(tables):
    """Positive and negative control: the same plan shares only when the share is on."""
    with config_context(_budgeted(fraction=1 / 16)):
        ds = _session(tables).sql(_ROLLUP)
        _plan, shared = _rewritten(ds)
    assert len(shared) > len(ds._sources), "the finest aggregate was not held"
    with config_context(_budgeted(fraction=0.0)):
        ds = _session(tables).sql(_ROLLUP)
        _plan, fixed = _rewritten(ds)
    assert len(fixed) == len(ds._sources), "a 1 KiB fixed cap must refuse it"


def test_the_rollup_answer_does_not_depend_on_sharing(tables):
    """A/B on the engine alone: shared and recomputed levels return the same rows."""
    with config_context(_budgeted(fraction=1 / 16)):
        shared = _session(tables).sql(_ROLLUP).collect()
    with config_context(_budgeted(fraction=0.0)):
        recomputed = _session(tables).sql(_ROLLUP).collect()

    def rows(t):
        return sorted(t.to_pylist(), key=repr)

    assert rows(shared) == rows(recomputed)


# TPC-DS q77: a genuine cross product (`FROM cs, cr`) inside the shared ROLLUP input.
_ROLLUP_OVER_PRODUCT = """
WITH cs AS (SELECT s, sum(v) t, count(*) n FROM fact GROUP BY s),
     cr AS (SELECT dm, sum(v) r FROM fact, dim WHERE k = dk GROUP BY dm)
SELECT channel, s, sum(t) tt, sum(r) rr, sum(n) nn FROM (
  SELECT 'catalog' channel, s, t, r, n FROM cs, cr
  UNION ALL SELECT 'store' channel, s, t, 0.0 r, n FROM cs
) u GROUP BY ROLLUP(channel, s)
"""


def test_a_rollup_over_a_real_cross_product_matches_duckdb(duck, tables):
    _register(duck, tables)
    got = _session(tables).sql(_ROLLUP_OVER_PRODUCT).collect()
    assert got.num_rows > 20
    assert_same(got, duck.sql(_ROLLUP_OVER_PRODUCT))

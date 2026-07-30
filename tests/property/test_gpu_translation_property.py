"""Randomly generated chains through the GPU translator must equal the CPU engine.

The case-by-case suite checks the shapes someone thought of. This checks the ones nobody did:
it builds chains from the translated vocabulary at random and compares each against the engine.
Every divergence found this way so far has been a *semantic* one — a dataframe library's
default quietly disagreeing with Arrow — rather than a missing feature, and those are exactly
the ones a hand-written case list does not cover, because writing the case requires already
knowing the disagreement exists.

Chains are a deterministic function of their seed, so a failure names a seed that reproduces
it. A chain the plan builder rejects (a `group_by` drops a column a later step names) is
skipped rather than counted: the generator is deliberately loose, and validating it here would
just restate the plan builder.
"""

from __future__ import annotations

import random

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher import coalesce, col, greatest, least, lit, when
from batcher.core.gpu_plan import DfBackend, Unsupported, gpu_plan_ops
from batcher.core.gpu_plan.execute import run_chain

pytestmark = [pytest.mark.property, pytest.mark.unit]

#: How many seeds one run covers. Kept modest so the suite stays fast; the same generator has
#: been run over thousands of seeds by hand, and a regression that only shows at seed 900 will
#: show at seed 90 soon enough.
SEEDS = 120

#: Rows are chosen to make the null and NaN paths live: `a` and `b` carry nulls at coprime
#: strides so their null sets overlap partially, and the row count is well past a single batch.
ROWS = 400


@pytest.fixture(scope="module")
def be():
    import pandas as pd

    return DfBackend(pd)


@pytest.fixture(scope="module")
def table():
    rng = np.random.default_rng(99)
    return pa.table(
        {
            "a": pa.array(
                [
                    None if i % 37 == 0 else int(v)
                    for i, v in enumerate(rng.integers(-20, 20, ROWS))
                ],
                type=pa.int64(),
            ),
            "b": pa.array(
                [
                    None if i % 23 == 0 else float(v)
                    for i, v in enumerate(rng.random(ROWS) * 100 - 50)
                ],
                type=pa.float64(),
            ),
            "g": pa.array(rng.integers(0, 5, ROWS).tolist(), type=pa.int64()),
            "s": pa.array(
                [None if i % 29 == 0 else f"k{v}" for i, v in enumerate(rng.integers(0, 7, ROWS))],
                type=pa.string(),
            ),
        }
    )


def _numeric(rng, depth=0):
    """A numeric expression drawn from the translated vocabulary."""
    leaves = [
        lambda: col("a"),
        lambda: col("b"),
        lambda: col("a").cast("float64"),
        lambda: lit(rng.choice([1, 2, 3, 7])),
        lambda: lit(rng.choice([0.5, 2.5, -1.5])),
    ]
    if depth >= 2:
        return rng.choice(leaves)()
    kind = rng.randrange(10)
    if kind == 0:
        return _numeric(rng, depth + 1) + _numeric(rng, depth + 1)
    if kind == 1:
        return _numeric(rng, depth + 1) - _numeric(rng, depth + 1)
    if kind == 2:
        return _numeric(rng, depth + 1) * _numeric(rng, depth + 1)
    if kind == 3:
        return col("b").abs()
    if kind == 4:
        return col("b").round(rng.choice([0, 1, 2]))
    if kind == 5:
        return col("a") % rng.choice([2, 3, 5])
    if kind == 6:
        return col("a") // rng.choice([2, 3])
    if kind == 7:
        return coalesce(col("b"), lit(0.0))
    if kind == 8:
        return rng.choice(
            [
                lambda: greatest(col("b"), lit(0.0)),
                lambda: least(col("b"), col("a").cast("float64")),
                lambda: when(col("a") > 0).then(col("b")).otherwise(lit(-1.0)),
                lambda: col("b").sqrt(),
            ]
        )()
    return col("b").floor() if rng.random() < 0.5 else col("b").ceil()


def _predicate(rng):
    choices = [
        lambda: _numeric(rng) > rng.choice([-10.0, 0.0, 5.0]),
        lambda: _numeric(rng) <= rng.choice([-5.0, 1.0, 20.0]),
        lambda: col("a").is_null(),
        lambda: col("b").is_not_null(),
        lambda: col("s").str.contains("k1"),
        lambda: col("g").is_in([0, 2, 4]),
        lambda: (col("a") > 0) & (col("b") < 10.0),
        lambda: col("s").str.starts_with("k"),
        lambda: col("b").is_nan(),
        lambda: col("s").str.len() > 1,
        lambda: (col("a") % 2) == 0,
        lambda: ~(col("g") == 3),
    ]
    return rng.choice(choices)()


def _step(rng):
    """One operator, as a function from Dataset to Dataset."""
    kind = rng.randrange(8)
    if kind == 0:
        predicate = _predicate(rng)
        return lambda ds: ds.filter(predicate)
    if kind == 1:
        expr = _numeric(rng)
        return lambda ds: ds.with_columns(w=expr)
    if kind == 2:
        keys = rng.choice([["g"], ["g", "a"], ["s"]])
        func = rng.choice(["sum", "mean", "min", "max", "count", "product"])
        return lambda ds: ds.group_by(*keys).agg(r=getattr(col("b"), func)(), n=bt.count())
    if kind == 3:
        key, desc = rng.choice(["a", "b", "g"]), rng.choice([True, False])
        return lambda ds: ds.sort(key, descending=desc)
    if kind == 4:
        n = rng.choice([1, 5, 50])
        return lambda ds: ds.limit(n)
    if kind == 5:
        return lambda ds: ds.distinct()
    if kind == 6:
        key = rng.choice(["g", "s"])
        func = rng.choice(["var", "std", "median", "count_distinct"])
        return lambda ds: ds.group_by(key).agg(r=getattr(col("b"), func)())
    part, order = rng.choice(["g", "a"]), rng.choice(["b", "a"])
    func = rng.choice(["row_number", "rank", "dense_rank", "percent_rank", "cume_dist", "ntile"])
    return lambda ds: ds.window(
        partition_by=[part], order_by=[(order, False)], functions={"rk": func}
    )


def _chain(seed: int, ds):
    rng = random.Random(seed)
    for _ in range(rng.randrange(1, 4)):
        ds = _step(rng)(ds)
    return ds


def _canon(v):
    if isinstance(v, float):
        return "__nan__" if v != v else float(f"{v:.12e}")
    return v


def _rows(t: pa.Table) -> list[tuple]:
    return [tuple(_canon(v) for v in r) for r in zip(*t.to_pydict().values(), strict=True)]


#: Operators whose row order is part of the result, so their cases compare row-for-row.
_ORDERED_TOPS = frozenset({"sort", "window", "limit", "project"})


def test_random_chains_match_the_cpu_engine(table, be):
    """Every generated chain the translator accepts must produce the engine's own answer."""
    checked = 0
    for seed in range(SEEDS):
        try:
            ds = _chain(seed, bt.from_arrow(table))
            spec = gpu_plan_ops(ds._plan)
        except Exception:
            continue  # the generator is loose; a plan the builder rejects is not a case
        if spec is None:
            continue
        try:
            got = be.to_arrow(run_chain(table, spec[1], be))
        except Unsupported:
            continue  # declined is a valid outcome; it means the CPU engine runs the stage
        expected = ds.collect()
        assert not set(expected.column_names) - set(got.column_names), f"seed {seed}"
        g, e = _rows(got.select(expected.column_names)), _rows(expected)
        if ds._plan.to_ir().get("op") not in _ORDERED_TOPS:
            g, e = sorted(g, key=repr), sorted(e, key=repr)
        assert g == e, f"seed {seed}"
        checked += 1
    # The generator is loose enough that some seeds are skipped; if almost all were, the test
    # would be passing by checking nothing.
    assert checked > SEEDS // 2, f"only {checked} of {SEEDS} seeds produced a case"

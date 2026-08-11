"""The H2O.ai db-benchmark input tables, built to the benchmark's published spec.

`db-benchmark <https://github.com/h2oai/db-benchmark>`_ is the standard groupby/join
workload for dataframe engines — the one Polars, DuckDB, data.table and pandas all publish
numbers against. It ships no data: every published run generates its own from the two R
scripts in ``_data/``, so generating it here is running the benchmark as specified, not
inventing a substrate (the distinction ``sources.tables`` draws for TPC-DS's ``dsdgen``).

Both generators below follow ``groupby-datagen.R`` and ``join-datagen.R`` column for
column — the same cardinalities (``id1``/``id2`` at K groups, ``id3`` at N/K, the integer
``id4``/``id5``/``id6`` mirrors), the same value ranges (``v1`` in [1,5], ``v2`` in [1,15],
``v3`` uniform on [0,100] rounded to 6 decimals), and the same LHS/RHS key construction for
the join task (90% of keys shared, 10% left-only, 10% right-only, so an inner join drops
rows and a left join keeps them).

They are **not** byte-identical to the published CSVs: R's ``set.seed(108)`` sampler cannot
be reproduced from NumPy, so the draws differ. That does not weaken the comparison the suite
makes. What the correctness gate needs is that every engine sees the same input, which a
fixed seed and one shared Arrow table give exactly; what the benchmark needs is the data's
*shape*, which is specified above and reproduced here. Absolute times are therefore
comparable across the engines in a run, and not against h2o's published leaderboard.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

__all__ = ["build_groupby", "build_join"]

# The benchmark's smallest published size (`Rscript groupby-datagen.R 1e7 1e2 0 0`), which
# `--scale 1` reproduces; scale 10 is its 1e8 tier. Its K (group-count) parameter is 1e2.
ROWS_PER_SCALE = 10_000_000
GROUPS = 100
_SEED = 108  # the seed the benchmark's own generators set


def _ids(rng: np.random.Generator, n: int, distinct: int, width: int) -> pa.Array:
    """``n`` draws from ``id{1..distinct}``, zero-padded to ``width`` (the R ``sprintf``)."""
    labels = np.array([f"id{i + 1:0{width}d}" for i in range(distinct)], dtype=object)
    return pa.array(labels[rng.integers(0, distinct, size=n)], type=pa.string())


def build_groupby(scale: float) -> dict[str, pa.Table]:
    """Build the groupby task's single table ``x`` (``G1_1e7_1e2_0_0`` at scale 1).

    Args:
        scale: Row-count multiplier over the benchmark's 1e7-row tier.

    Returns:
        The one table the ten groupby queries read, keyed as ``x``.
    """
    n = int(ROWS_PER_SCALE * scale)
    k = GROUPS
    rng = np.random.default_rng(_SEED)
    table = pa.table(
        {
            "id1": _ids(rng, n, k, 3),  # large groups (char)
            "id2": _ids(rng, n, k, 3),  # small groups (char)
            "id3": _ids(rng, n, n // k, 10),  # large groups (char)
            "id4": pa.array(rng.integers(1, k + 1, size=n), type=pa.int32()),
            "id5": pa.array(rng.integers(1, k + 1, size=n), type=pa.int32()),
            "id6": pa.array(rng.integers(1, n // k + 1, size=n), type=pa.int32()),
            "v1": pa.array(rng.integers(1, 6, size=n), type=pa.int32()),
            "v2": pa.array(rng.integers(1, 16, size=n), type=pa.int32()),
            "v3": pa.array(np.round(rng.uniform(0.0, 100.0, size=n), 6), type=pa.float64()),
        }
    )
    return {"x": table}


def _split_xlr(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The benchmark's key split: 90% shared, 10% LHS-only, 10% RHS-only, over ``1.1 * n``."""
    keys = rng.permutation(int(n * 1.1)) + 1
    shared = int(n * 0.9)
    return keys[:shared], keys[shared:n], keys[n:]


def _sample_all(rng: np.random.Generator, pool: np.ndarray, size: int) -> np.ndarray:
    """``size`` draws covering every member of ``pool`` at least once (R's ``sample_all``).

    ``size < len(pool)`` cannot satisfy that, and the R original asserts against it. Here the
    negative draw count raises out of NumPy rather than being clamped: clamping would silently
    return a *subset* of the key universe, which shows up much later as a join whose
    cardinality quietly does not match the benchmark's.
    """
    drawn = np.concatenate([pool, rng.choice(pool, size=size - len(pool), replace=True)])
    return rng.permutation(drawn)


def _labelled(keys: np.ndarray) -> pa.Array:
    """The ``id4``/``id5``/``id6`` string mirrors of an integer key column."""
    return pa.array(np.char.add("id", keys.astype("U")), type=pa.string())


def build_join(scale: float) -> dict[str, pa.Table]:
    """Build the join task's LHS ``x`` and its three RHS tables.

    The RHS tables are the benchmark's three size tiers relative to the LHS: ``small`` at
    N/1e6 rows joined on ``id1``, ``medium`` at N/1e3 joined on ``id2``, and ``big`` at N
    joined on ``id3`` — so the five join queries cover a broadcast-shaped join, a mid-size
    one, an outer join, a join on a string key, and a full-size join.

    Args:
        scale: Row-count multiplier over the benchmark's 1e7-row tier.

    Returns:
        The four tables keyed as ``x``, ``small``, ``medium``, ``big``.
    """
    n = int(ROWS_PER_SCALE * scale)
    rng = np.random.default_rng(_SEED)
    # Key universes, sized so `small`/`medium`/`big` have exactly n/1e6, n/1e3 and n distinct
    # join keys. `_split_xlr` splits each into the shared / LHS-only / RHS-only thirds.
    #
    # The floor of 10 is what the benchmark's own `stopifnot(N>=1e7)` implies: at its
    # smallest published tier n/1e6 is already 10, so the floor binds only *below* the
    # official sizes, where a 0.9/0.1/0.1 split of one key would otherwise leave the RHS
    # with no keys at all. A sub-tier scale is for smoke-testing the wiring, not for a
    # number anyone quotes.
    small_n, medium_n = max(n // 1_000_000, 10), max(n // 1_000, 10)
    k1_x, k1_l, k1_r = _split_xlr(rng, small_n)
    k2_x, k2_l, k2_r = _split_xlr(rng, medium_n)
    k3_x, k3_l, k3_r = _split_xlr(rng, n)

    lhs_id1 = _sample_all(rng, np.concatenate([k1_x, k1_l]), n)
    lhs_id2 = _sample_all(rng, np.concatenate([k2_x, k2_l]), n)
    lhs_id3 = _sample_all(rng, np.concatenate([k3_x, k3_l]), n)
    x = pa.table(
        {
            "id1": pa.array(lhs_id1, type=pa.int32()),
            "id2": pa.array(lhs_id2, type=pa.int32()),
            "id3": pa.array(lhs_id3, type=pa.int32()),
            "id4": _labelled(lhs_id1),
            "id5": _labelled(lhs_id2),
            "id6": _labelled(lhs_id3),
            "v1": pa.array(np.round(rng.uniform(0.0, 100.0, size=n), 6), type=pa.float64()),
        }
    )

    def rhs(rows: int, keys: list[tuple[str, np.ndarray]]) -> pa.Table:
        cols: dict[str, pa.Array] = {}
        for name, pool in keys:
            drawn = _sample_all(rng, pool, rows)
            cols[name] = pa.array(drawn, type=pa.int32())
            cols[f"id{int(name[-1]) + 3}"] = _labelled(drawn)
        cols["v2"] = pa.array(np.round(rng.uniform(0.0, 100.0, size=rows), 6), type=pa.float64())
        return pa.table(cols)

    r1 = np.concatenate([k1_x, k1_r])
    r2 = np.concatenate([k2_x, k2_r])
    r3 = np.concatenate([k3_x, k3_r])
    return {
        "x": x,
        "small": rhs(small_n, [("id1", r1)]),
        "medium": rhs(medium_n, [("id1", r1), ("id2", r2)]),
        "big": rhs(n, [("id1", r1), ("id2", r2), ("id3", r3)]),
    }

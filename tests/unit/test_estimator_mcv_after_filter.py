"""A learned most-common-value frequency must not be applied to rows a filter already cut.

An MCV frequency is a share of the rows it was measured over — the whole base table. Once a
predicate constrains the column, the survivors are a different population: `year = 1999` is
~0.5% of a 200-year calendar but a third of a 1998-2000 slice. Carried through unchanged, the
frequency priced TPC-DS q47's `d_year = 1999 OR (d_year = 1998 AND d_moy = 12) OR ...` at one
row against 427, flipped the build side of the fact join, and ran the CTE 4.7x slower.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.unit


def _calendar() -> pa.Table:
    days = np.arange(365 * 200)
    return pa.table({"year": 1900 + days // 365, "moy": 1 + (days % 365) // 31})


def _estimate(ds) -> float:
    from batcher import core
    from batcher.api.source_stats import collect_source_stats, column_bounds_needed
    from batcher.kyber.learning import load_learned_stats
    from batcher.kyber.stats.estimator import StatsEstimator

    plan, sources = ds._plan, ds._sources
    stats = collect_source_stats(sources, None, need_columns=column_bounds_needed(plan))
    learned = load_learned_stats(core.default_hub())
    return StatsEstimator(sources, learned, source_stats=stats).estimate(plan).rows


def test_an_equality_after_a_range_on_the_same_column_uses_the_narrowed_domain():
    base = bt.from_arrow(_calendar())
    # Seed the learned statistics the way a real run does: equality columns get MCVs.
    base.filter(bt.col("year") == 1950).collect()
    probe = base.filter(bt.col("year") == 1999)
    unconstrained = _estimate(probe)
    # Positive control: MCVs were learned and do price an unconstrained equality (~365 rows).
    assert 200 <= unconstrained <= 700

    narrowed = base.filter((bt.col("year") >= 1998) & (bt.col("year") <= 2000)).filter(
        (bt.col("year") == 1999) | ((bt.col("year") == 1998) & (bt.col("moy") == 12))
    )
    actual = narrowed.count()
    estimate = _estimate(narrowed)
    assert actual / 3 <= estimate <= actual * 3, (estimate, actual)

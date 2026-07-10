"""End-to-end: executing a query makes Kyber's next estimate of it better.

The unit tests pin the algebra of the correction loop. This one proves the loop is
actually *wired*: Core measures per-operator cardinality during a real execution, the
MetadataHub persists it, and the next `optimize` of the same shape estimates closer to
the truth — with results unchanged.

The query is a two-key group-by on functionally dependent keys (`g2 == g1`). The
structural estimator multiplies the keys' distinct counts under an independence
assumption, so it over-estimates the group count. No amount of per-column statistics
fixes that: the error is in the *correlation*, which only measuring the operator's actual
output reveals. This is precisely what DuckDB, Polars, and Daft do not carry across
executions, and what Spark AQE corrects only within a single query.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, core
from batcher.kyber import Optimizer
from batcher.kyber.learning import CARDINALITY_CORRECTION_KEY, load_learned_stats
from batcher.metadata import MetadataHub
from batcher.metadata.backends.in_process import InProcessBackend

pytest.importorskip("batcher._native", reason="native engine not built")

_ROWS = 20_000
_REAL_GROUPS = 100


def _table() -> pa.Table:
    g1 = [i % _REAL_GROUPS for i in range(_ROWS)]
    return pa.table({"g1": g1, "g2": list(g1), "v": list(range(_ROWS))})


def _query(table: pa.Table):
    # The aggregate is deliberately *not* the plan root: `record_execution` stores an
    # absolute row count for the root signature only, and that measurement would shadow
    # (and suppress) the correction this test is about.
    return bt.from_arrow(table).group_by("g1", "g2").agg(s=col("v").sum()).filter(col("s") >= 0)


def _root_estimate(dataset, hub: MetadataHub) -> float:
    """Kyber's estimated output rows for `dataset`, given what `hub` has learned."""
    _plan, stats = Optimizer(sources=dataset._sources, hub=hub).logical_stats(dataset._plan)
    return stats.rows


@pytest.mark.integration
def test_correction_sharpens_the_estimate_across_executions():
    table = _table()

    # Cold: an empty hub has learned nothing, so this is the purely structural estimate.
    cold = _root_estimate(_query(table), MetadataHub(InProcessBackend()))

    # Executing through the ordinary terminal path is what feeds the loop; the conductor
    # records each operator's measured cardinality into the process hub.
    for _ in range(3):
        _query(table).collect()

    hub = core.default_hub()
    corrections = load_learned_stats(hub).get(CARDINALITY_CORRECTION_KEY, {})
    assert corrections, "executing the query must teach Kyber at least one correction"

    warm = _root_estimate(_query(table), hub)

    # The structural estimate is too high; the corrected one must be strictly closer to
    # the 100 groups the aggregate really produces.
    assert cold > _REAL_GROUPS
    assert abs(warm - _REAL_GROUPS) < abs(cold - _REAL_GROUPS)


@pytest.mark.integration
def test_correction_never_changes_results():
    table = _table()
    expected = _query(table).collect().to_pydict()
    for _ in range(4):  # by now a correction is applied to the aggregate
        got = _query(table).collect().to_pydict()
    assert len(got["g1"]) == _REAL_GROUPS
    assert sorted(got["g1"]) == sorted(expected["g1"])
    assert sorted(got["s"]) == sorted(expected["s"])

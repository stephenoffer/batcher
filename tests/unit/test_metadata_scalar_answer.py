"""The scalar metadata shortcuts fire only from EXACT stats, and fall back otherwise.

Pins the provenance firewall for the scalar column terminals (`min`/`max`/
`null_count`/`n_unique`/`has_nulls`/`all_null`): a Parquet footer answers them with no
scan, a filter (which downgrades away from EXACT) makes them return `None` so the caller
executes, and the learned-quantile shortcut is explicitly approximate. Also unit-tests
the pure interpolation helper.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher.api.terminal.metadata_answer import (
    metadata_all_null,
    metadata_has_nulls,
    metadata_learned_quantile,
    metadata_max,
    metadata_min,
    metadata_n_unique,
    metadata_null_count,
)
from batcher.kyber.metadata_answer import _value_at_quantile, answer_learned_quantile

pytestmark = pytest.mark.unit


@pytest.fixture
def pq_path(tmp_path):
    table = pa.table(
        {
            "x": pa.array([3, 1, 2, None, 5], type=pa.int64()),
            "g": pa.array([10, 10, 20, 20, 20], type=pa.int64()),
        }
    )
    path = str(tmp_path / "t.parquet")
    pq.write_table(table, path)
    return path


def _ds(pq_path):
    return bt.read.parquet(pq_path)


# --- the shortcut FIRES from a real footer (no scan) ---


def test_min_max_fire_from_footer(pq_path):
    ds = _ds(pq_path)
    assert metadata_min(ds._plan, ds._sources, "x") == 1
    assert metadata_max(ds._plan, ds._sources, "x") == 5


def test_null_count_and_has_nulls_fire_from_footer(pq_path):
    ds = _ds(pq_path)
    assert metadata_null_count(ds._plan, ds._sources, "x") == 1  # one null
    assert metadata_has_nulls(ds._plan, ds._sources, "x") is True
    assert metadata_has_nulls(ds._plan, ds._sources, "g") is False


def test_all_null_fires_false_from_footer(pq_path):
    ds = _ds(pq_path)
    # A real EXACT answer (not None): the column is not entirely null.
    assert metadata_all_null(ds._plan, ds._sources, "x") is False


# --- the firewall: a filter downgrades EXACT → the shortcut returns None ---


def test_filter_disables_every_scalar_shortcut(pq_path):
    ds = _ds(pq_path).filter(bt.col("x") > 1)
    assert metadata_min(ds._plan, ds._sources, "x") is None
    assert metadata_max(ds._plan, ds._sources, "x") is None
    assert metadata_null_count(ds._plan, ds._sources, "x") is None
    assert metadata_has_nulls(ds._plan, ds._sources, "x") is None
    assert metadata_all_null(ds._plan, ds._sources, "x") is None
    # ...but the public terminals still return the correct executed answer.
    assert ds.min("x") == 2
    assert ds.max("x") == 5
    assert ds.n_null("x") == 0
    assert ds.has_nulls("x") is False


def test_exact_ndv_required_for_n_unique(pq_path):
    # Parquet footers carry no EXACT distinct count → no metadata answer (execute).
    ds = _ds(pq_path)
    assert metadata_n_unique(ds._plan, ds._sources, "x") is None
    assert ds.n_unique("x") == 4  # but execution is correct


def test_in_memory_scalars_answered_from_learned_bounds():
    ds = bt.from_arrow(pa.table({"x": pa.array([1, 1, None, 4], type=pa.int64())}))
    # An in-memory source now exposes EXACT column bounds (learned once, cached per instance),
    # so an unfiltered `min` is answered from metadata and equals a full run. `n_unique` has
    # no ndv to derive from, so it still executes — correct either way.
    assert metadata_min(ds._plan, ds._sources, "x") == 1
    assert ds.min("x") == 1
    assert ds.n_null("x") == 1
    assert ds.n_unique("x") == 2
    assert ds.all_null("x") is False


# --- learned quantile: explicitly approximate, None when unlearned ---


def test_learned_quantile_none_when_unlearned():
    from batcher.metadata import MetadataHub
    from batcher.metadata.backends import InProcessBackend

    assert answer_learned_quantile("x", 0.5, MetadataHub(InProcessBackend())) is None
    assert metadata_learned_quantile("__never_learned_col__", 0.5) is None


def test_learned_quantile_reads_grid():
    from batcher.kyber.learning import record_column_stats
    from batcher.metadata import MetadataHub
    from batcher.metadata.backends import InProcessBackend

    hub = MetadataHub(InProcessBackend())
    # probs/values are ascending KLL boundaries (0..100 over [0, 1]).
    record_column_stats(hub, {}, {"q": {"probs": [0.0, 0.5, 1.0], "values": [0.0, 50.0, 100.0]}})
    assert answer_learned_quantile("q", 0.5, hub) == 50.0
    assert answer_learned_quantile("q", 0.25, hub) == 25.0  # interpolated
    assert answer_learned_quantile("q", 0.0, hub) == 0.0
    assert answer_learned_quantile("q", 1.0, hub) == 100.0


def test_value_at_quantile_edges():
    probs = [0.0, 0.5, 1.0]
    values = [10.0, 20.0, 40.0]
    assert _value_at_quantile(0.0, probs, values) == 10.0
    assert _value_at_quantile(1.0, probs, values) == 40.0
    assert _value_at_quantile(0.75, probs, values) == 30.0  # midpoint of [20, 40]
    assert _value_at_quantile(0.5, probs, values) == 20.0
    # Unusable grids → None.
    assert _value_at_quantile(0.5, [0.0], [10.0]) is None
    assert _value_at_quantile(0.5, [0.0, 1.0], [10.0]) is None


# --- candidate set widening is a gate only (answer_aggregate stays authoritative) ---


def test_sum_is_candidate_but_executes_without_recorded_total(pq_path):
    # `sum` joined `_METADATA_DERIVABLE_AGGS`, so it is *attempted*; a Parquet footer
    # records no exact total, so the answer is None and execution produces the value.
    from batcher.api.terminal.metadata_answer import (
        is_global_aggregate,
        metadata_aggregate_table,
    )

    ds = _ds(pq_path).agg(s=bt.col("x").sum())
    assert is_global_aggregate(ds._plan) is True  # now a candidate
    assert metadata_aggregate_table(ds._plan, ds._sources) is None  # not derivable → execute
    assert ds.to_pydict() == {"s": [11]}

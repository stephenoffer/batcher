"""A row callback returns the same relation on one node and on many.

`ds.ml.filter` is a new operator surface, and `map`/`flat_map` gained declarations
(`input_columns`) and a budget (`max_errored_rows`) that the distributed path has to honour
the same way the local one does. Both are the kind of thing that works locally and diverges
across workers: the filter's `preserves_columns` declaration invites the optimizer to move a
predicate, and the error budget is explicitly *per worker*, so its behaviour is only visible
once there is more than one.

The assertion is the mergeable-algebra invariant every other operator is held to — same row
multiset, same column names, same column types — not a re-test of the callback itself.
"""

from __future__ import annotations

import pytest

import batcher as bt
from _ray_cluster import init_test_ray, shutdown_test_ray
from batcher import col

pytest.importorskip("ray", reason="ray not installed")
pytest.importorskip("batcher._native", reason="native engine not built")


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    started = init_test_ray(2)
    yield
    shutdown_test_ray(started)


def _ds(n: int = 500):
    return bt.from_pydict(
        {"id": list(range(n)), "g": [i % 7 for i in range(n)], "s": [f"r{i}" for i in range(n)]}
    )


def _same(plan) -> None:
    one, many = plan.collect(), plan.collect(distributed=True)
    assert one.schema == many.schema
    assert sorted(one.to_pylist(), key=repr) == sorted(many.to_pylist(), key=repr)


def _keep_even(row: dict) -> bool:
    return row["id"] % 2 == 0


def _double(row: dict) -> dict:
    return {"id": row["id"] * 2}


def test_a_row_filter_is_identical_across_workers():
    _same(_ds().ml.filter(_keep_even))


def test_a_row_filter_under_a_pushed_predicate_is_identical():
    """The `preserves_columns` declaration lets the optimizer move the vectorized filter
    below the Python one. If that rewrite were unsound, this is where it would show."""
    _same(_ds().ml.filter(_keep_even).filter(col("g") < 3))


def test_a_declared_row_map_is_identical_across_workers():
    _same(_ds().ml.map(_double, input_columns=["id"], output_columns=["id"]))


def test_a_row_filter_composed_with_an_aggregate_is_identical():
    _same(_ds().ml.filter(_keep_even).group_by("g").agg(n=bt.count(), total=col("id").sum()))

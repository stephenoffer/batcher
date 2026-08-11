"""A variable-shape tensor column crosses the cluster unchanged.

The representation is a plain struct of a binary buffer, a shape, and a dtype — chosen so the
engine needs no knowledge of it to carry it. That claim is only worth what it is tested at:
single-node it obviously works, because the column is built and read in the same process. The
distributed path is where a representation that needs special handling would be found out,
because the batch is serialized, shuffled, and reassembled by code that has never heard of it.

So this asserts the invariant the mergeable algebra promises for every other column: the
result is identical whether produced on one node or several — same shapes, same dtypes, same
bytes.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from _ray_cluster import init_test_ray, shutdown_test_ray
from batcher.io.formats.ml.ragged import is_ragged_tensor_column, ragged_to_numpy

pytest.importorskip("ray", reason="ray not installed")
pytest.importorskip("batcher._native", reason="native engine not built")


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    started = init_test_ray(2)
    yield
    shutdown_test_ray(started)


def _images(n: int) -> list[np.ndarray]:
    """`n` arrays whose shapes and dtypes deliberately do not agree."""
    return [np.full((i % 3 + 1, i % 4 + 1), i, "uint8") for i in range(n)]


def _dataset(n: int = 64):
    return bt.from_pydict({"id": list(range(n)), "img": _images(n)})


def _decoded(table):
    return ragged_to_numpy(table.column("img"))


def test_a_ragged_column_is_identical_single_node_and_distributed():
    plan = _dataset().filter(bt.col("id") % 2 == 0)
    single, many = plan.collect(), plan.collect(distributed=True)

    assert single.schema.field("img").type == many.schema.field("img").type
    assert is_ragged_tensor_column(many.schema.field("img").type)

    one, lots = _decoded(single), _decoded(many)
    assert [a.shape for a in one] == [a.shape for a in lots]
    assert [a.dtype for a in one] == [a.dtype for a in lots]
    assert all(np.array_equal(a, b) for a, b in zip(one, lots, strict=True))


def _decode(batch):
    """Build each row's array from the row's own `id`, never from its position in the batch.

    Deriving the shape from ``range(batch.num_rows)`` would make the *result* a function of
    the partitioning, so the two sides could only ever be compared loosely — and a test that
    compares loosely is the one that misses the divergence it exists to catch.
    """
    ids = batch.column("id").to_pylist()
    return {
        "id": batch.column("id"),
        "img": [np.full((i % 3 + 1, 2), i % 251, "uint8") for i in ids],
    }


def test_it_survives_a_udf_stage_on_the_distributed_path():
    """The shape that produces a ragged column in the first place, run across workers."""
    plan = bt.from_pydict({"id": list(range(32))}).map_batches(
        _decode, output_columns=["id", "img"]
    )
    single, many = plan.collect().sort_by("id"), plan.collect(distributed=True).sort_by("id")

    assert single.num_rows == many.num_rows == 32
    assert single.column("id").to_pylist() == many.column("id").to_pylist()
    one, lots = ragged_to_numpy(single.column("img")), ragged_to_numpy(many.column("img"))
    assert [a.shape for a in one] == [a.shape for a in lots]
    assert all(np.array_equal(a, b) for a, b in zip(one, lots, strict=True))

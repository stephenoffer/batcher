"""Competitor differential: the UDF verbs against Ray Data itself, on a local Ray.

Ray Data is the oracle for what a ported pipeline must return. Two shapes carry most ports:
`filter(fn)` and `map_batches` over a callable class configured with `fn_constructor_args`.
Batcher runs each single-node and with `distributed=True`, and both must equal Ray's rows.

The one deliberate difference is recorded as a test rather than papered over: Ray's
`filter(fn)` calls `fn` per row, Batcher's calls it per batch and expects a boolean mask, so
the ported predicate is rewritten, and the rows it keeps are what is compared.
"""

from __future__ import annotations

import numpy as np
import pyarrow.compute as pc
import pytest

import batcher as bt
from _ray_cluster import init_test_ray, shutdown_test_ray

pytestmark = pytest.mark.integration

pytest.importorskip("ray", reason="ray not installed")
pytest.importorskip("ray.data", reason="ray.data not installed")

ROWS = [{"id": i, "v": float(i % 7)} for i in range(300)]


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    """A local cluster, with this module pickled by value so its UDFs reach a worker."""
    import sys

    from ray import cloudpickle

    started = init_test_ray(2)
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    yield
    cloudpickle.unregister_pickle_by_value(sys.modules[__name__])
    shutdown_test_ray(started)


def _ray_rows(ds) -> list[tuple]:
    return sorted((int(r["id"]), float(r["v"])) for r in ds.take_all())


def _bt_rows(ds: bt.Dataset, *, distributed: bool) -> list[tuple]:
    table = ds.collect(distributed=distributed)
    return sorted(zip(table.column("id").to_pylist(), table.column("v").to_pylist(), strict=True))


class Scale:
    """A callable-class UDF built once per worker from constructor arguments."""

    def __init__(self, factor, *, offset=0.0):
        self.factor = factor
        self.offset = offset

    def __call__(self, batch):
        return {"id": batch["id"], "v": batch["v"] * self.factor + self.offset}


@pytest.mark.parametrize("distributed", [False, True])
def test_filter_with_a_callable_keeps_the_rows_ray_keeps(distributed):
    import ray.data

    expected = _ray_rows(ray.data.from_items(ROWS).filter(lambda row: row["v"] > 3))
    got = bt.from_pylist(ROWS).filter(lambda batch: pc.greater(batch["v"], 3))
    assert _bt_rows(got, distributed=distributed) == expected
    assert len(expected) == sum(r["v"] > 3 for r in ROWS)  # Ray did filter something


@pytest.mark.parametrize("distributed", [False, True])
def test_map_batches_with_constructor_args_matches_ray(distributed):
    import ray.data

    expected = _ray_rows(
        ray.data.from_items(ROWS).map_batches(
            Scale,
            fn_constructor_args=(2.0,),
            fn_constructor_kwargs={"offset": 1.0},
            concurrency=2,
            batch_format="numpy",
        )
    )
    got = bt.from_pylist(ROWS).map_batches(
        Scale,
        fn_constructor_args=(2.0,),
        fn_constructor_kwargs={"offset": 1.0},
        concurrency=2,
        batch_format="numpy",
    )
    assert isinstance(got._plan.fn, type)  # still a class: built once per worker
    assert _bt_rows(got, distributed=distributed) == expected
    assert expected != sorted((r["id"], r["v"]) for r in ROWS)  # the UDF changed values


@pytest.mark.parametrize("distributed", [False, True])
def test_a_class_filter_with_constructor_args_matches_ray(distributed):
    import ray.data

    class Above:
        def __init__(self, threshold):
            self.threshold = threshold

        def __call__(self, item):
            if isinstance(item, dict) and isinstance(item["v"], np.ndarray):
                return item["v"] > self.threshold  # Batcher: one boolean per row
            return item["v"] > self.threshold  # Ray: one row

    expected = _ray_rows(
        ray.data.from_items(ROWS).filter(Above, fn_constructor_args=(4.0,), concurrency=2)
    )
    got = bt.from_pylist(ROWS).filter(
        Above, fn_constructor_args=(4.0,), concurrency=2, batch_format="numpy"
    )
    assert _bt_rows(got, distributed=distributed) == expected

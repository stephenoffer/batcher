"""A load-once model is released on the streaming path, not just on `collect`.

`iter_batches` builds every class UDF once up front (`core.prebuild_factories`) so the model
loads a single time for the whole stream. Nothing then released it. `teardown_udf` declines a
*prebuilt* instance on purpose — "that owner's to tear down at its lifetime end, not here" —
and the owner never did, so a GPU model streamed through `iter_batches` held its allocation
for the life of the process. Garbage collection did not reclaim it either: the plan holds the
instance and the caller usually holds the plan.

This is the cross-product the project guard names: the same class UDF behaved differently on
`collect` and on `iter_batches`, and only one of them was covered.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt


class Model:
    """A load-once 'model' that records its own construction and release."""

    builds = 0
    closes = 0

    def __init__(self) -> None:
        Model.builds += 1

    def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        return batch

    def close(self) -> None:
        Model.closes += 1


@pytest.fixture(autouse=True)
def _reset() -> None:
    Model.builds = Model.closes = 0


@pytest.fixture
def ds() -> bt.Dataset:
    return bt.from_pydict({"x": list(range(200))})


def test_collect_releases_the_model(ds: bt.Dataset) -> None:
    ds.map_batches(Model).to_pydict()
    assert Model.closes == 1


def test_a_fully_consumed_stream_releases_the_model(ds: bt.Dataset) -> None:
    assert sum(b.num_rows for b in ds.map_batches(Model).iter_batches()) == 200
    assert Model.closes == 1


def test_an_abandoned_stream_releases_the_model(ds: bt.Dataset) -> None:
    """Stopping early is the common shape for a streamed read, and the worst case to leak."""
    stream = ds.map_batches(Model).iter_batches()
    next(stream)
    stream.close()
    assert Model.closes == 1


def test_every_stage_of_a_chain_is_released(ds: bt.Dataset) -> None:
    list(ds.map_batches(Model).map_batches(Model).iter_batches())
    assert Model.closes == 2


def test_the_model_still_loads_only_once(ds: bt.Dataset) -> None:
    """Releasing must not have turned a load-once model into a per-window rebuild."""
    list(ds.map_batches(Model).iter_batches())
    assert Model.builds == 1


def test_reiterating_a_dataset_rebuilds_and_releases_each_time(ds: bt.Dataset) -> None:
    """The obvious way to get this wrong is to close a model the next iteration still needs.

    `prebuild_factories` returns a *new* plan, so each `iter_batches()` builds its own
    instance and releases its own. Builds and closes must stay balanced across calls, and
    the second pass must still produce rows.
    """
    plan = ds.map_batches(Model)
    list(plan.iter_batches())
    assert (Model.builds, Model.closes) == (1, 1)
    assert sum(b.num_rows for b in plan.iter_batches()) == 200
    assert (Model.builds, Model.closes) == (2, 2)


def test_a_model_without_close_streams_fine(ds: bt.Dataset) -> None:
    class Bare:
        def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
            return batch

    assert sum(b.num_rows for b in ds.map_batches(Bare).iter_batches()) == 200


def test_a_failing_close_does_not_fail_the_stream(ds: bt.Dataset) -> None:
    """The rows are already produced, so teardown is best-effort by contract."""

    class Angry:
        def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
            return batch

        def close(self) -> None:
            raise RuntimeError("teardown exploded")

    assert sum(b.num_rows for b in ds.map_batches(Angry).iter_batches()) == 200

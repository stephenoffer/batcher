"""The shared vector-store contract, held to what it promises before any store is dialed.

Every connector in `io.formats.vector` validates a shard, checks the target, sends batches
with idempotent retries, and reports failures point by point through one implementation, so
those promises are pinned here once, through a recording sink, and each store's suite only
covers what is specific to it.
"""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from batcher._internal.errors import PlanError
from batcher.io.formats.vector import contract
from batcher.io.formats.vector.contract import (
    PointFailure,
    RemoteTarget,
    VectorSink,
    VectorWriteError,
    check_remote,
    prepare,
    resolve_metric,
    send_with_retries,
    stable_point_id,
)

pytestmark = pytest.mark.io


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(contract.time, "sleep", lambda _s: None)


class RecordingSink(VectorSink):
    """A sink whose 'store' is a list, failing the batches a test asks it to."""

    format_name = "recording"
    native_metrics = {"cosine": "cos", "dot": "ip"}  # noqa: RUF012 - test double

    def __init__(self, *, remote: RemoteTarget | None = None, fail: dict | None = None, **kw: Any):
        super().__init__(**kw)
        self.remote = remote
        self.fail = dict(fail or {})  # batch index -> how many attempts fail
        self.sent: list[list[Any]] = []
        self.attempts: list[list[Any]] = []
        self.opened = 0

    def _client(self) -> Any:
        self.opened += 1
        return object()

    def _describe(self, client: Any, path: str, payload: list[str]) -> RemoteTarget | None:
        return self.remote

    def _send(self, client: Any, path: str, chunk: pa.Table) -> None:
        ids = chunk.column(self.id_column).to_pylist()
        self.attempts.append(ids)
        key = ids[0]
        if self.fail.get(key, 0) > 0:
            self.fail[key] -= 1
            raise ConnectionError(f"batch starting {key} refused")
        self.sent.append(ids)


def _frame(n: int = 4, dim: int = 2) -> pa.Table:
    return pa.table(
        {
            "id": list(range(n)),
            "embedding": [[float(i)] * dim for i in range(n)],
            "title": [f"t{i}" for i in range(n)],
        }
    )


# --- the frame's shape ----------------------------------------------------------------


def test_a_list_of_doubles_is_normalized_to_fixed_size_float32() -> None:
    prepared = prepare(_frame(), id_column="id", vector_columns=["embedding"])
    assert prepared.table.schema.field("embedding").type == pa.list_(pa.float32(), 2)
    assert prepared.dimensions == {"embedding": 2}
    assert prepared.table.column("embedding").to_pylist()[3] == [3.0, 3.0]


def test_integer_fixed_size_lists_and_one_dimensional_tensors_are_accepted() -> None:
    ints = pa.table(
        {"id": [1, 2], "embedding": pa.array([[1, 2], [3, 4]], pa.list_(pa.int32(), 2))}
    )
    assert prepare(ints, id_column="id", vector_columns=["embedding"]).dimensions == {
        "embedding": 2
    }
    tensor_type = pa.fixed_shape_tensor(pa.float32(), [3])
    storage = pa.array([[1, 2, 3], [4, 5, 6]], pa.list_(pa.float32(), 3))
    tensors = pa.table(
        {"id": [1, 2], "embedding": pa.ExtensionArray.from_storage(tensor_type, storage)}
    )
    prepared = prepare(tensors, id_column="id", vector_columns=["embedding"])
    assert prepared.table.column("embedding").type == pa.list_(pa.float32(), 3)


def test_a_two_dimensional_tensor_or_a_string_column_is_not_a_vector() -> None:
    tensor_type = pa.fixed_shape_tensor(pa.float32(), [2, 2])
    storage = pa.array([[1, 2, 3, 4]], pa.list_(pa.float32(), 4))
    bad = pa.table({"id": [1], "embedding": pa.ExtensionArray.from_storage(tensor_type, storage)})
    with pytest.raises(PlanError, match="one-dimensional"):
        prepare(bad, id_column="id", vector_columns=["embedding"])
    with pytest.raises(PlanError, match="fixed_size_list<float32"):
        prepare(
            pa.table({"id": [1], "embedding": ["x"]}), id_column="id", vector_columns=["embedding"]
        )


def test_a_missing_column_or_an_unusable_id_type_is_refused_by_name() -> None:
    with pytest.raises(PlanError, match="'vec'"):
        prepare(_frame(), id_column="id", vector_columns=["vec"])
    floats = pa.table({"id": [1.5], "embedding": [[1.0]]})
    with pytest.raises(PlanError, match="integer or string ids"):
        prepare(floats, id_column="id", vector_columns=["embedding"])


def test_every_invalid_point_is_named_and_nothing_is_written() -> None:
    frame = pa.table(
        {
            "id": ["a", None, "c", "c", "e", "f", "g", "h"],
            "embedding": [
                [1.0, 2.0],
                [1.0, 2.0],
                [1.0, 2.0],
                [1.0, 2.0],
                None,
                [1.0, 2.0, 3.0],
                [float("nan"), 1.0],
                [1e300, 1.0],  # finite in float64, infinite once it is float32
            ],
        }
    )
    sink = RecordingSink()
    with pytest.raises(VectorWriteError) as raised:
        sink.write(frame, "t")
    reasons = {f.id: f.reason for f in raised.value.failures}
    assert reasons == {
        None: "null id",
        "c": "duplicate id in this write",
        "e": "null vector",
        "f": "vector does not have 2 values",
        "g": "vector holds a NaN, infinity or null",
        "h": "vector holds a NaN, infinity or null",
    }
    assert len(raised.value.failures) == 7  # both occurrences of the duplicate
    assert raised.value.written == 0
    assert sink.opened == 0 and sink.attempts == [], "a bad shard must not reach the store"


def test_a_declared_dimension_is_enforced() -> None:
    fixed = pa.table({"id": [1], "embedding": pa.array([[1.0, 2.0]], pa.list_(pa.float32(), 2))})
    with pytest.raises(PlanError, match="dimension=3"):
        prepare(fixed, id_column="id", vector_columns=["embedding"], dimension=3)
    with pytest.raises(VectorWriteError, match="does not have 3 values"):
        prepare(_frame(1), id_column="id", vector_columns=["embedding"], dimension=3)


# --- metrics and the remote check -----------------------------------------------------


def test_a_metric_is_accepted_by_portable_name_or_the_stores_spelling() -> None:
    native = {"cosine": "Cosine", "euclidean": "Euclid", "dot": "Dot"}
    assert resolve_metric("euclidean", native, store="qdrant") == "Euclid"
    assert resolve_metric("euclid", native, store="qdrant") == "Euclid"
    assert resolve_metric(None, native, store="qdrant") is None
    with pytest.raises(PlanError, match="no distance metric 'hamming'") as raised:
        resolve_metric("hamming", native, store="qdrant")
    assert "cosine" in raised.value.available


def test_the_target_is_checked_before_any_batch_is_sent() -> None:
    sink = RecordingSink(remote=RemoteTarget(dimensions={"": 3}, metric="cos"))
    with pytest.raises(PlanError, match="3-dimensional vectors"):
        sink.write(_frame(), "t")
    assert sink.attempts == []
    sink = RecordingSink(remote=RemoteTarget(dimensions={"": 2}, metric="cos"), metric="dot")
    with pytest.raises(PlanError, match="'cos' metric, not 'ip'"):
        sink.write(_frame(), "t")
    assert sink.attempts == []


def test_a_target_that_says_nothing_is_not_held_against_the_frame() -> None:
    check_remote(None, dimensions={"": 2}, metric="cos", store="s", target="t")
    check_remote(RemoteTarget(), dimensions={"": 2}, metric="cos", store="s", target="t")


# --- batching, retries, and the per-point report --------------------------------------


def test_a_write_is_sent_in_batches_and_counted() -> None:
    sink = RecordingSink(batch_size=3)
    written = sink.write(_frame(7), "t")
    assert sink.sent == [[0, 1, 2], [3, 4, 5], [6]]
    assert written.rows == 7


def test_a_transient_failure_is_retried_with_the_same_ids() -> None:
    sink = RecordingSink(batch_size=2, fail={2: 2})
    assert sink.write(_frame(4), "t").rows == 4
    assert sink.attempts == [[0, 1], [2, 3], [2, 3], [2, 3]]


def test_a_batch_that_keeps_failing_is_reported_point_by_point_and_the_rest_lands() -> None:
    sink = RecordingSink(batch_size=2, max_retries=1, fail={2: 5})
    with pytest.raises(VectorWriteError) as raised:
        sink.write(_frame(6), "t")
    assert sink.sent == [[0, 1], [4, 5]]
    assert [f.id for f in raised.value.failures] == [2, 3]
    assert "batch starting 2 refused" in raised.value.failures[0].reason
    assert raised.value.written == 4
    assert "2 of 6 points were not written (4 were)" in str(raised.value)


def test_an_unretryable_mode_is_sent_once() -> None:
    class InsertOnly(RecordingSink):
        supported_modes = ("upsert", "append")
        unretryable_modes = frozenset({"append"})

    sink = InsertOnly(mode="append", fail={0: 1})
    with pytest.raises(VectorWriteError):
        sink.write(_frame(2), "t")
    assert sink.attempts == [[0, 1]]


def test_send_with_retries_returns_the_last_error() -> None:
    calls = []

    def boom() -> None:
        calls.append(1)
        raise ValueError(f"attempt {len(calls)}")

    error = send_with_retries(boom, retries=2)
    assert str(error) == "attempt 3"


def test_a_delete_needs_only_ids() -> None:
    sink = RecordingSink(mode="delete")
    assert sink.write(pa.table({"id": [1, 2]}), "t").rows == 2


def test_an_empty_shard_opens_no_connection() -> None:
    sink = RecordingSink()
    assert sink.write(_frame(0), "t").rows == 0
    assert sink.opened == 0


def test_the_failure_report_survives_pickling_for_a_distributed_write() -> None:
    err = VectorWriteError("m", failures=[PointFailure("a", "null vector")], written=3)
    back = pickle.loads(pickle.dumps(err))
    assert back.failures == (PointFailure("a", "null vector"),)
    assert back.written == 3


def test_overwrite_and_append_are_declined_by_name() -> None:
    from batcher._internal.errors import BackendError

    with pytest.raises(BackendError, match="cannot express"):
        RecordingSink(mode="overwrite")


def test_bad_knobs_are_refused_at_construction() -> None:
    for kwargs in ({"batch_size": 0}, {"max_retries": -1}, {"dimension": 0}):
        with pytest.raises(PlanError):
            RecordingSink(**kwargs)


# --- stable ids -----------------------------------------------------------------------


def test_a_string_id_maps_to_one_uuid_forever_and_ints_and_uuids_pass_through() -> None:
    assert stable_point_id(5) == 5
    assert stable_point_id("doc-1") == stable_point_id("doc-1") != stable_point_id("doc-2")
    # Pinned: a change here re-keys every point a previous run wrote.
    assert stable_point_id("doc-1") == "d13c0d8f-e1b2-5b8c-b5cd-9b6c2d168a93"
    uid = "123e4567-e89b-12d3-a456-426614174000"
    assert stable_point_id(uid) == uid
    with pytest.raises(PlanError, match="negative"):
        stable_point_id(-1)


def test_validation_is_vectorized_over_a_large_shard() -> None:
    """A 200k-point shard validates in Arrow; this would be slow as a Python loop."""
    n = 200_000
    values = pa.array(np.random.default_rng(0).random(n * 8, dtype=np.float32))
    frame = pa.table(
        {"id": pa.array(np.arange(n)), "embedding": pa.FixedSizeListArray.from_arrays(values, 8)}
    )
    assert prepare(frame, id_column="id", vector_columns=["embedding"]).dimensions == {
        "embedding": 8
    }

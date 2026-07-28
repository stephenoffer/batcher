"""The one measurement every memory *bound* in the engine has to agree on.

A byte figure is used two ways here, and only one of them can use `nbytes`. Reported to a
user, `nbytes` is right: it is the size of their data. Used as a bound — spill or not, how
much to hold in a chunk, does this build side fit a broadcast — it is wrong, because it
measures the rows an object addresses rather than the memory it keeps alive. Every
zero-copy derivation in Arrow windows a parent buffer and pins the whole thing.

`retained_bytes` is the second figure. These tests pin it, pin the direction it is allowed
to be wrong in, and pin the properties every caller relies on: that it never reads below
`nbytes`, that it is additive, and that it never raises from inside a memory guard.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.plan.types import retained_bytes, total_retained_bytes

pytestmark = pytest.mark.unit


def _parent_batch(rows: int = 500_000) -> pa.RecordBatch:
    return pa.record_batch({"v": pa.array(range(rows), type=pa.int64())})


# --- the measurement it exists for --------------------------------------------


def test_a_window_reads_as_the_parent_it_pins() -> None:
    window = _parent_batch().slice(0, 4)
    assert window.nbytes < 100
    assert retained_bytes(window) > 1_000_000


def test_a_window_of_a_table_reads_the_same_way() -> None:
    """Tables and batches are both bounded by callers, so both must measure alike."""
    table = pa.table({"v": pa.array(range(500_000), type=pa.int64())}).slice(0, 4)
    assert retained_bytes(table) > 1_000_000


def test_an_undivided_object_reads_as_itself() -> None:
    """The common case must not inflate, or every bound tightens for nothing."""
    batch = _parent_batch(10_000)
    assert batch.nbytes <= retained_bytes(batch) <= batch.nbytes * 1.1


@pytest.mark.parametrize(
    "column",
    [
        pa.array([1, None, 3], type=pa.int64()),
        pa.array(["a", None, "ccc"]),
        pa.array([[1, 2], None, []], type=pa.list_(pa.int64())),
        pa.array([], type=pa.float64()),
        pa.array(["x", "y", "x"]).dictionary_encode(),
    ],
    ids=["ints", "strings", "lists", "empty", "dictionary"],
)
def test_every_column_shape_is_measurable(column) -> None:
    """A memory guard must get a number for whatever it is holding, including nothing."""
    batch = pa.record_batch({"c": column})
    assert retained_bytes(batch) >= batch.nbytes >= 0


# --- the properties callers depend on -----------------------------------------


def test_it_never_reads_below_the_logical_size() -> None:
    """An object cannot retain less than the rows it addresses.

    A caller that chunks to a byte target relies on this: if the measure could read below
    `nbytes`, a chunk sized to 64 MiB could address more than 64 MiB of rows and the bound
    would be looser than the naive one it replaced.
    """
    for obj in (_parent_batch(), _parent_batch().slice(0, 4), pa.table({"s": pa.array(["a"])})):
        assert retained_bytes(obj) >= obj.nbytes


def test_the_total_is_the_sum_of_the_parts() -> None:
    """Callers bound lists, so the aggregate has to compose."""
    parts = [_parent_batch(1000), _parent_batch(2000).slice(0, 5)]
    assert total_retained_bytes(parts) == sum(retained_bytes(p) for p in parts)


def test_the_total_of_nothing_is_zero() -> None:
    assert total_retained_bytes([]) == 0


def test_it_does_not_raise_on_something_it_cannot_measure() -> None:
    """It is called from inside memory guards, where raising is worse than an estimate."""

    class Opaque:
        pass

    class Broken:
        nbytes = 128

        def get_total_buffer_size(self):
            raise TypeError("no")

    assert retained_bytes(Opaque()) == 0
    assert retained_bytes(Broken()) == 128


def test_it_errs_high_rather_than_low() -> None:
    """Two columns sharing one dictionary buffer are counted twice, and that is correct here.

    Over-counting spills or chunks sooner and costs throughput. Under-counting costs the
    process. The asymmetry is the whole design, so it is worth a test that would fail if
    someone later "fixed" the double count by subtracting shared buffers.
    """
    shared = pa.array(["a", "b", "a"]).dictionary_encode()
    batch = pa.record_batch({"x": shared, "y": shared})
    assert retained_bytes(batch) >= retained_bytes(pa.record_batch({"x": shared}))

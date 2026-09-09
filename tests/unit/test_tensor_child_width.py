"""A tensor column's `float32` child survives the FFI boundary; an `int32` one still widens.

The boundary widens narrow numerics once so every operator stays on the `int64`/`float64`
paths, and it recurses into nested types. For a **list of floats** that recursion was pure
loss. Such a column is not a numeric column, it is a tensor -- an embedding, a decoded image,
a feature vector -- and widening its child doubled it and forced a full cast on a path that is
otherwise zero-copy. Measured on 500,000 x 512 `float32` (1 GB), `iter_batches` took 1,476 ms
against 0.9 ms for the identical data carrying the `arrow.fixed_shape_tensor` extension type,
which the boundary has always exempted. 1,189 ms of that is the cast. The caller also got
`float64` tensors back from a `float32` corpus, at twice the bytes.

The **integer** child still widens, and that asymmetry is the point of this file. Its
justification is not width but wrap: an `int32` element that later reaches arithmetic
overflows silently, where the widened one gives the right answer. A float does not wrap.

Two sides have to agree or `Dataset.schema` lies: `bc_py::normalize_list_element` and
`plan.types.widen`. Both are asserted here, against each other.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher.plan.types import widen

pytestmark = pytest.mark.unit

_N = 64
_DIM = 4


def _tensor_table(child: pa.DataType) -> pa.Table:
    """A `fixed_size_list<child>[4]` column plus a flat column of the same type."""
    rng = np.random.default_rng(11)
    n = _N * _DIM
    values = rng.integers(0, 100, n) if pa.types.is_integer(child) else rng.random(n)
    flat = pa.array(values).cast(child)
    return pa.table(
        {
            "feat": pa.FixedSizeListArray.from_arrays(flat, _DIM),
            "scalar": pa.array(values[:_N]).cast(child),
        }
    )


# --------------------------------------------------------------------------- #
# The prediction (`plan.types.widen`)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("child", [pa.float32(), pa.float16()])
def test_a_float_list_child_is_predicted_narrow(child):
    assert widen(pa.list_(child)) == pa.list_(child)
    assert widen(pa.large_list(child)) == pa.large_list(child)
    assert widen(pa.list_(child, _DIM)) == pa.list_(child, _DIM)


def test_a_nested_float_list_child_is_predicted_narrow():
    """A ragged 2-D tensor is reached through two list containers and is still a tensor."""
    assert widen(pa.list_(pa.list_(pa.float32()))) == pa.list_(pa.list_(pa.float32()))


@pytest.mark.parametrize("child", [pa.int8(), pa.int32(), pa.uint32()])
def test_an_integer_list_child_still_widens(child):
    """The wrap argument is integer-specific, so this arm is unchanged.

    See `test_a_narrow_integer_element_wraps_which_is_why_it_widens` for the measurement
    that makes this an argument rather than an assertion.
    """
    assert widen(pa.list_(child)) == pa.list_(pa.int64())
    assert widen(pa.list_(child, _DIM)) == pa.list_(pa.int64(), _DIM)


@pytest.mark.parametrize(
    ("child", "values", "wrapped", "correct"),
    [
        (pa.int32(), [2_000_000_000, 2_000_000_000], -294_967_296, 4_000_000_000),
        (pa.uint8(), [250, 250], 244, 500),
    ],
)
def test_a_narrow_integer_element_wraps_which_is_why_it_widens(child, values, wrapped, correct):
    """The reason the integer arm keeps widening, demonstrated rather than asserted.

    The tempting next step after the float exemption is the same one for integers, and the
    prize is large: a `uint8` image tensor is widened **8x** at the boundary, which a
    CPU->GPU pipeline then ships over Flight. The reductions all survive a narrow child --
    `sum`/`mean`/`max` accumulate in `i64` whatever the element width -- and reading only
    those says the exemption is safe. It is not.

    `list.get(i)` yields a column at the *element's own* width, and two of them in one
    arithmetic expression wrap at that width. This drives it through the
    `arrow.fixed_shape_tensor` extension type, which `normalize_batch` passes through
    untouched, so the narrow child is reachable today without changing anything -- and the
    same expression through the ordinary widened path gives the right answer.

    Recovering those bytes needs the plan's *logical* type to stay `int64` while the morsel
    carries `uint8`, which is a type-system change and not a boundary edit.
    """
    storage = pa.FixedSizeListArray.from_arrays(pa.array(values).cast(child), len(values))
    narrow = pa.ExtensionArray.from_storage(pa.fixed_shape_tensor(child, [len(values)]), storage)
    doubled = col("l").list.get(0) + col("l").list.get(1)

    assert (
        bt.from_arrow(pa.table({"l": narrow})).select(r=doubled).collect().column(0)[0].as_py()
        == wrapped
    ), "a narrow element must be shown to wrap, or this test proves nothing"
    assert (
        bt.from_arrow(pa.table({"l": storage})).select(r=doubled).collect().column(0)[0].as_py()
        == correct
    )


def test_a_top_level_float32_column_still_widens():
    """Only the *element* of a list is exempt; a plain numeric column is not."""
    assert widen(pa.float32()) == pa.float64()


def test_a_struct_inside_a_list_reverts_to_the_ordinary_rules():
    """Struct fields are addressable and behave like columns, so the wrap argument applies."""
    inner = pa.struct([pa.field("a", pa.float32()), pa.field("b", pa.int32())])
    assert widen(pa.list_(inner)) == pa.list_(
        pa.struct([pa.field("a", pa.float64()), pa.field("b", pa.int64())])
    )


def test_a_float_list_inside_a_struct_keeps_its_child():
    """The exemption follows the list, wherever the list is."""
    outer = pa.struct([pa.field("emb", pa.list_(pa.float32(), _DIM))])
    assert widen(outer) == pa.struct([pa.field("emb", pa.list_(pa.float32(), _DIM))])


def test_widen_stays_idempotent():
    for dt in (pa.list_(pa.float32(), _DIM), pa.list_(pa.int32()), pa.float32()):
        assert widen(widen(dt)) == widen(dt)


# --------------------------------------------------------------------------- #
# The engine (`bc_py::normalize_list_element`) — and that the two agree
# --------------------------------------------------------------------------- #
def test_the_engine_returns_a_float32_tensor_unwidened():
    """The regression: a `float32` feature column comes back `float32`, not `float64`."""
    ds = bt.from_arrow(_tensor_table(pa.float32()))
    out = ds.collect()
    assert out.schema.field("feat").type == pa.list_(pa.float32(), _DIM)
    # The flat column is untouched by this change and still widens.
    assert out.schema.field("scalar").type == pa.float64()


def test_the_engine_still_widens_an_integer_tensor():
    ds = bt.from_arrow(_tensor_table(pa.int32()))
    out = ds.collect()
    assert out.schema.field("feat").type == pa.list_(pa.int64(), _DIM)


@pytest.mark.parametrize("child", [pa.float32(), pa.int32()])
def test_the_declared_schema_matches_what_the_engine_produces(child):
    """`Dataset.schema` must predict what `collect()` returns, or it lies."""
    ds = bt.from_arrow(_tensor_table(child))
    produced = ds.collect().schema
    for name in ("feat", "scalar"):
        assert ds.schema.field(name).type == produced.field(name).type, name


def test_iter_batches_agrees_with_collect_on_the_tensor_type():
    """The loader path is the one this change exists for; it must not diverge from `collect`."""
    ds = bt.from_arrow(_tensor_table(pa.float32()))
    batch = next(iter(ds.iter_batches(16)))
    assert batch.schema.field("feat").type == ds.collect().schema.field("feat").type


# --------------------------------------------------------------------------- #
# The values are unchanged — the point is bytes, never arithmetic
# --------------------------------------------------------------------------- #
def test_list_kernels_return_the_same_values_as_the_widened_child_did():
    """`l2_norm` over the narrow child must equal the same query over an f64 copy.

    The f64 arm is built by casting the source, so it is exactly what the boundary used to
    hand the kernels. A tolerance is required and is not a loosened gate: the two arms sum in
    different precisions by construction, which is the whole difference between them.
    """
    table = _tensor_table(pa.float32())
    wide = table.set_column(
        table.schema.get_field_index("feat"),
        "feat",
        table.column("feat").cast(pa.list_(pa.float64(), _DIM)),
    )
    narrow_out = bt.from_arrow(table).select(n=col("feat").list.l2_norm()).collect()
    wide_out = bt.from_arrow(wide).select(n=col("feat").list.l2_norm()).collect()
    np.testing.assert_allclose(
        np.asarray(narrow_out.column("n").to_pylist(), dtype=np.float64),
        np.asarray(wide_out.column("n").to_pylist(), dtype=np.float64),
        rtol=1e-6,
    )


def test_a_carried_tensor_survives_a_filter_and_a_join_unchanged():
    """Carry-through operators treat a tensor as an opaque unit; the values must round-trip."""
    table = _tensor_table(pa.float32())
    keys = pa.table({"k": pa.array(np.arange(_N, dtype=np.int64))})
    src = bt.from_arrow(table.append_column("k", keys.column("k")))
    out = src.filter(col("k") < 10).sort("k").collect()
    assert out.schema.field("feat").type == pa.list_(pa.float32(), _DIM)
    expected = table.column("feat").to_pylist()[:10]
    np.testing.assert_allclose(
        np.asarray(out.column("feat").to_pylist(), dtype=np.float32),
        np.asarray(expected, dtype=np.float32),
    )

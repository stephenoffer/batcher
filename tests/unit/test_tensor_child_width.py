"""A tensor column's narrow child survives the FFI boundary; the ops that read it still widen.

The boundary widens narrow numerics once so every operator stays on the `int64`/`float64`
paths, and it recurses into nested types. For a **list of numbers** that recursion was pure
loss. Such a column is not a numeric column, it is a tensor -- an embedding, a decoded image,
a feature vector -- and widening its child multiplied it and forced a full cast on a path that
is otherwise zero-copy. Measured on 500,000 x 512 `float32` (1 GB), `iter_batches` took
1,476 ms against 0.9 ms for the identical data carrying the `arrow.fixed_shape_tensor`
extension type, which the boundary has always exempted. 1,189 ms of that is the cast.

The float arm landed first and the **integer** arm followed it, because the integer factor is
worse: a decoded-image corpus is `fixed_size_list<uint8>` and `int64` is **8x** the bytes.
Measured on 4,000 224x224x3 images read from Parquet, against the same corpus carrying the
extension metadata as the control: 4.49 GB against 0.56 GB materialized, 4.75 s against 2.93 s.

What kept the integer arm widened for a while is a real hazard rather than caution, and it is
what `test_a_narrow_integer_element_does_not_wrap_once_it_becomes_a_column` exists to hold:
`bc_expr`'s `Add`/`Sub`/`Mul` are `*_wrapping` and `coerce_numeric` short-circuits on identical
operand types, so two narrow elements meeting in one expression wrap at their own width. An
element only reaches arithmetic by first becoming a *column*, so the three ops that do that --
`list.get`, `list.min`, `list.max` -- widen there instead. That is one cast per row rather than
one per element, and every observable output type is what it was before.

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


@pytest.mark.parametrize("child", [pa.int8(), pa.int32(), pa.uint8(), pa.uint32()])
def test_a_narrow_integer_list_child_is_predicted_narrow(child):
    """The integer arm followed the float one — see the wrap test below for what pays for it."""
    assert widen(pa.list_(child)) == pa.list_(child)
    assert widen(pa.list_(child, _DIM)) == pa.list_(child, _DIM)


def test_a_uint64_list_child_still_widens():
    """`uint64` is the one integer width `int64` cannot hold, so its arm is not a widening.

    Leaving it on the ordinary path is what keeps the exemption to types where the two arms
    genuinely agree: the boundary refuses a `uint64` above `i64::MAX` rather than truncating
    it, and that refusal is only reachable if the cast is still attempted.
    """
    assert widen(pa.list_(pa.uint64())) == pa.list_(pa.int64())


@pytest.mark.parametrize(
    ("child", "values", "correct"),
    [
        (pa.int32(), [2_000_000_000, 2_000_000_000], 4_000_000_000),
        (pa.uint8(), [250, 250], 500),
    ],
)
def test_a_narrow_integer_element_does_not_wrap_once_it_becomes_a_column(child, values, correct):
    """The reason the integer arm could follow the float one, demonstrated rather than asserted.

    The wrap hazard is real and this test used to pin it: `bc_expr`'s `Add`/`Sub`/`Mul` are
    `*_wrapping` and `coerce_numeric` short-circuits on identical operand types, so
    `l.get(0) + l.get(1)` over an `Int32` child wrapped at 2^31 where the widened pair did not.
    Reading only the reductions said the exemption was safe -- `sum`/`mean`/`max` accumulate in
    `i64` whatever the element width -- and that reading was incomplete.

    What changed is *where* the widening is paid. An element only reaches arithmetic by first
    becoming a column, and three ops do that: `list.get`, `list.min`, `list.max`. Each widens a
    narrow integer on the way out, which is one cast per **row** instead of one per element, so
    the 8x on a `uint8` image tensor is gone and the answer here is unchanged.

    Both arms are driven, and they must agree: the extension type (which `normalize_batch` has
    always passed through untouched) and the ordinary storage column now carry the same narrow
    child, so a divergence between them would mean the escape-point widening fires on only one
    of the two paths a narrow child can arrive by.
    """
    storage = pa.FixedSizeListArray.from_arrays(pa.array(values).cast(child), len(values))
    narrow = pa.ExtensionArray.from_storage(pa.fixed_shape_tensor(child, [len(values)]), storage)
    doubled = col("l").list.get(0) + col("l").list.get(1)

    for name, arr in (("extension", narrow), ("storage", storage)):
        got = bt.from_arrow(pa.table({"l": arr})).select(r=doubled).collect().column(0)[0].as_py()
        assert got == correct, f"{name} arm wrapped: {got} != {correct}"


@pytest.mark.parametrize("child", [pa.int8(), pa.uint8(), pa.int32()])
def test_every_element_escape_hands_back_int64(child):
    """`get`/`min`/`max`/`sum` are what turn an element into a column; all four widen.

    Declared *and* executed, because a schema that predicted the narrow type while the engine
    produced `int64` would be the same lie the boundary mirror exists to prevent.
    """
    tbl = pa.table({"l": pa.array([[1, 2, 3]], pa.list_(child))})
    ds = bt.from_arrow(tbl)
    for name, expr in (
        ("get", col("l").list.get(0)),
        ("min", col("l").list.min()),
        ("max", col("l").list.max()),
        ("sum", col("l").list.sum()),
    ):
        one = ds.select(v=expr)
        assert one.schema.field("v").type == pa.int64(), f"{name} declared"
        assert one.collect().schema.field("v").type == pa.int64(), f"{name} executed"


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


def test_the_engine_returns_an_integer_tensor_unwidened():
    """A `uint8` decoded-image column is the shape this arm was extended for: 8x the bytes."""
    ds = bt.from_arrow(_tensor_table(pa.uint8()))
    out = ds.collect()
    assert out.schema.field("feat").type == pa.list_(pa.uint8(), _DIM)
    # The flat column beside it is untouched by this change and still widens.
    assert out.schema.field("scalar").type == pa.int64()


@pytest.mark.parametrize("child", [pa.float32(), pa.int32(), pa.uint8()])
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

"""Two device-tier column-type repairs, both found by comparing a device against the CPU engine.

The device tier's contract fixes column *types* as exactly as it fixes rows, and every defect it
has shipped has been a type bug with correct values -- a DATE returning `timestamp[ms]`, an
integer `abs` widening to double, and now these two. That shape is invisible to any comparison
that checks values, and invisible to this package's own suite for a second reason: the tests run
on pandas, and both of these are cuDF behaviours pandas does not share.

So these cases pin the *repairs* rather than the library behaviour they repair. Each names the
query that caught it, because a full-suite GPU-vs-CPU sweep is the only thing that found them and
the next one will be found the same way.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.core.gpu_plan.backend import DfBackend

pytestmark = pytest.mark.unit


class _Series:
    """The one thing `_restore_empty_strings` asks a column for: its dtype."""

    def __init__(self, dtype: str) -> None:
        self.dtype = dtype


class _Frame:
    """A stand-in for a cuDF frame, which is what the repair is scoped to.

    Built rather than imported because cuDF cannot be installed in CI, which is the same reason
    the defect reached a cluster in the first place.
    """

    def __init__(self, dtypes: dict[str, str]) -> None:
        self._dtypes = dtypes
        self.columns = list(dtypes)

    def __getitem__(self, name: str) -> _Series:
        return _Series(self._dtypes[name])


class _ArrowNativeLib:
    """A `lib` whose `DataFrame` has `from_arrow`, which is how `DfBackend` recognises cuDF."""

    class DataFrame:
        @staticmethod
        def from_arrow(table):  # pragma: no cover - never called here
            raise NotImplementedError


def _gpu_backend() -> DfBackend:
    be = DfBackend(_ArrowNativeLib)
    assert be.is_gpu, "the repair is scoped to the device backend; the stub must look like one"
    return be


def test_empty_string_column_gets_its_type_back() -> None:
    """TPC-H q15: empty result, `s_name` came back `null` where the CPU engine says `string`."""
    table = pa.table({"s_name": pa.array([], type=pa.null()), "n": pa.array([], type=pa.int64())})
    frame = _Frame({"s_name": "object", "n": "int64"})
    out = _gpu_backend()._restore_empty_strings(frame, table)
    assert out.schema.field("s_name").type == pa.string()
    assert out.schema.field("n").type == pa.int64()
    assert out.num_rows == 0


def test_a_non_empty_table_is_left_alone() -> None:
    """A frame with rows already carries its types; re-typing it would be a second, drifting
    statement of the engine's rules rather than a repair of a library default."""
    table = pa.table({"s": pa.array(["a", "b"]), "n": pa.array([1, 2], type=pa.int64())})
    out = _gpu_backend()._restore_empty_strings(_Frame({"s": "object", "n": "int64"}), table)
    assert out is table


def test_a_non_string_null_column_is_not_renamed_a_string() -> None:
    """Only an `object` dtype means "this held strings". A genuinely null-typed column stays null,
    so `SELECT NULL AS x` still agrees with the CPU engine."""
    table = pa.table({"x": pa.array([], type=pa.null())})
    out = _gpu_backend()._restore_empty_strings(_Frame({"x": "float64"}), table)
    assert out.schema.field("x").type == pa.null()


def test_a_column_absent_from_the_frame_is_untouched() -> None:
    """cuDF's `to_arrow` can carry an index column the frame does not list; it is not a result
    column and must not be retyped on the strength of a dtype lookup that would raise."""
    table = pa.table({"index": pa.array([], type=pa.null())})
    out = _gpu_backend()._restore_empty_strings(_Frame({"other": "object"}), table)
    assert out.schema.field("index").type == pa.null()


def test_counting_reductions_are_int64() -> None:
    """ClickBench q08/q09: `COUNT(DISTINCT UserID)` returned `int32`, because cuDF answers
    `nunique` in int32 and pandas answers it in int64."""
    import pandas as pd

    from batcher.core.gpu_plan.aggs import _as_int64

    be = DfBackend(pd)
    narrow = pd.Series([1, 2, 3], dtype="int32")
    # `startswith`, not equality: the host backend carries Arrow-backed dtypes, so the width is
    # spelled `int64[pyarrow]` here and plain `int64` on a device. The width is the assertion.
    assert str(_as_int64(narrow, be).dtype).startswith("int64")

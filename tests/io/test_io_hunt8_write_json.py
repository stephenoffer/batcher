"""Wave-14 JSON writer fidelity: floats must round-trip bit-for-bit.

``JSONSink`` encoded through pandas' ``DataFrame.to_json``, which rounds floats to
``double_precision`` (default 10) *decimal places* — so ``3.141592653589793`` was
written as ``3.1415926536`` and read back changed. (Raising the cap to its maximum
15 does not help: it rounds the largest double up to ``inf``.) Python's ``json``
renders each float with ``repr`` — the shortest round-tripping form — so a float
written and read back is the value we started with.

Each test fails on the pre-fix (pandas) encoder and passes on the stdlib encoder.
"""

from __future__ import annotations

import io
import math

import pyarrow as pa
import pyarrow.json as pajson
import pytest

from batcher.io.formats.base import SINKS, SOURCES

pytestmark = pytest.mark.unit


def _write_read(tbl: pa.Table, tmp_path) -> pa.Table:
    p = str(tmp_path / "out.json")
    SINKS.get("json")().write(tbl, p)
    batches = list(SOURCES.get("json")(p).read())
    # A 0-row JSON file carries no schema (JSON has no way to encode one without data),
    # so the source yields no batches — build a 0-row table rather than crashing on
    # `from_batches([])`. The file being readable (not 0 bytes) is what this asserts.
    if not batches:
        return pa.table({})
    return pa.Table.from_batches(batches)


def test_float_precision_survives_json_round_trip(tmp_path):
    vals = [
        3.141592653589793,
        1.0 / 3.0,
        123456789.123456789,
        2.718281828459045,
        1.7976931348623157e308,  # max double — dp=15 would round this to inf
        5e-324,  # min subnormal
    ]
    tbl = pa.table({"x": pa.array(vals, pa.float64())})
    got = _write_read(tbl, tmp_path)
    assert got.column("x").to_pylist() == vals


def test_json_float_bytes_are_exact(tmp_path):
    from batcher.io.formats.semistructured.json import _ndjson_bytes

    b = _ndjson_bytes(pa.table({"x": [3.141592653589793]}))
    # The pre-fix pandas encoder wrote the truncated "3.1415926536".
    assert b"3.141592653589793" in b


def test_nested_struct_float_precision_survives(tmp_path):
    vals = [{"a": 3.141592653589793}, None, {"a": 1.0 / 3.0}]
    tbl = pa.table({"s": pa.array(vals, pa.struct([("a", pa.float64())]))})
    got = _write_read(tbl, tmp_path)
    assert got.column("s").to_pylist() == vals


def test_zero_row_float_write_is_a_readable_file(tmp_path):
    # The exact encoder must not emit a 0-byte file for an empty float table:
    # `pyarrow.json.read_json` rejects that as "Empty JSON file".
    tbl = pa.table({"x": pa.array([], pa.float64())})
    p = str(tmp_path / "empty.json")
    SINKS.get("json")().write(tbl, p)
    batches = SOURCES.get("json")(p).read()  # must not raise "Empty JSON file"
    assert sum(b.num_rows for b in batches) == 0


def test_nonfinite_floats_become_null_not_unreadable(tmp_path):
    # NaN/±Inf have no JSON form; the exact encoder must emit `null` (readable),
    # not the stdlib default `NaN`/`Infinity` literals that pyarrow cannot parse.
    from batcher.io.formats.semistructured.json import _ndjson_bytes

    tbl = pa.table({"x": pa.array([float("nan"), float("inf"), -1.5, -0.0], pa.float64())})
    b = _ndjson_bytes(tbl)
    back = pajson.read_json(io.BytesIO(b)).column("x").to_pylist()
    assert back[0] is None and back[1] is None
    assert back[2] == -1.5
    assert back[3] == -0.0 and math.copysign(1.0, back[3]) == -1.0

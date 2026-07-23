"""Process-pool machinery for the JSON **write** path.

Encoding NDJSON is CPU-bound and releases no GIL, so a large write farms shards out to a
process pool. That is a self-contained concern with no bearing on reading, and it is
separated here for a plain structural reason: `json.py` outgrew the module size limit
once the reader learned to stream, and this is the part of it nothing else references.

`_ndjson_bytes` deliberately stays in `json.py` — two existing test modules import it
from there, and moving it would break them for no benefit.
"""

from __future__ import annotations

import atexit
import json
import math
from typing import Any

import pyarrow as pa

from batcher.io.filesystem import resolve_filesystem

__all__ = [
    "_JSON_POOL_SIZE",
    "_disable_json_proc",
    "_json_encode_shard",
    "_json_pool",
    "_json_proc_disabled",
    "_json_write_part",
    "_ndjson_bytes",
    "_size_or_zero",
]


_JSON_POOL: Any = None
_JSON_POOL_SIZE = 0
# Set once if the JSON process pool proves unusable this session (e.g. a non-import-safe
# entrypoint forkserver/spawn can't fork a child from). After that every JSON write stays
# on the serial/thread path — never re-attempting (and re-breaking) the pool per write.
_JSON_PROC_DISABLED = False


def _disable_json_proc() -> None:
    """Disable the JSON process pool for the rest of the session (idempotent)."""
    global _JSON_POOL, _JSON_POOL_SIZE, _JSON_PROC_DISABLED
    _JSON_PROC_DISABLED = True
    if _JSON_POOL is not None:
        _JSON_POOL.shutdown(wait=False)
        _JSON_POOL = None
        _JSON_POOL_SIZE = 0


def _sanitize_nonfinite(value: Any) -> Any:
    """Replace NaN/±Inf floats with None (JSON has no non-finite; match pandas → ``null``)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [_sanitize_nonfinite(v) for v in value]
    if isinstance(value, dict):
        return {k: _sanitize_nonfinite(v) for k, v in value.items()}
    return value


def _nullable_int_mapper(arrow_type: pa.DataType) -> Any:
    """Map an integer Arrow type to pandas' nullable integer dtype (else default).

    ``Table.to_pandas`` upcasts an integer column that contains a null to float64 —
    which silently turns ``9007199254740993`` into ``9007199254740992.0`` and changes
    the column's type on a JSON round-trip. Mapping integer columns to pandas' nullable
    integer extension dtypes keeps every value exact and integer-typed through
    ``to_json``.
    """
    import pandas as pd

    if pa.types.is_integer(arrow_type):
        return pd.ArrowDtype(arrow_type)
    return None


def _schema_has_float(schema: pa.Schema) -> bool:
    """True if `schema` holds a floating-point value anywhere (nested included)."""

    def _has_float(t: pa.DataType) -> bool:
        if pa.types.is_floating(t):
            return True
        if pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t):
            return _has_float(t.value_type)
        if pa.types.is_struct(t):
            return any(_has_float(t.field(i).type) for i in range(t.num_fields))
        if pa.types.is_map(t):
            return _has_float(t.key_type) or _has_float(t.item_type)
        return False

    return any(_has_float(f.type) for f in schema)


def _table_to_ndjson_exact(table: pa.Table) -> bytes:
    """Encode via the stdlib, so every float round-trips bit-for-bit.

    pandas' ``to_json`` rounds floats to ``double_precision`` (default 10) decimal
    places — ``3.141592653589793`` becomes ``3.1415926536`` — and even the maximum
    ``double_precision=15`` can round the largest double up to ``inf``. Python's
    ``json.dumps`` renders each float with ``repr`` (the shortest round-tripping form),
    so the value read back equals the value written. Raises on non-JSON-native leaves
    (timestamp/decimal/bytes), letting the caller fall back to the pandas encoder.
    """
    if table.num_rows == 0:
        # A 0-byte file is not valid NDJSON (`pyarrow.json.read_json` rejects it as
        # "Empty JSON file"); emit a single newline for a readable empty file, matching
        # the pandas encoder's output so a float-schema empty write reads back cleanly.
        return b"\n"
    return b"".join(
        (json.dumps(_sanitize_nonfinite(row)) + "\n").encode("utf-8") for row in table.to_pylist()
    )


def _table_to_ndjson(table: pa.Table) -> bytes:
    """Encode `table` as newline-delimited JSON bytes via pandas' C-accelerated writer."""
    df = table.to_pandas(types_mapper=_nullable_int_mapper)
    ndjson = df.to_json(orient="records", lines=True)
    if ndjson and not ndjson.endswith("\n"):
        ndjson += "\n"  # so shard outputs concatenate into valid NDJSON
    return ndjson.encode("utf-8")


def _ndjson_bytes(table: pa.Table) -> bytes:
    """Encode `table` as NDJSON, preserving float precision, with a pandas fast path.

    Float columns route through the exact stdlib encoder (pandas' ``to_json`` silently
    truncates them); float-free tables take pandas' faster C encoder. Either way falls
    back to the other on failure so a missing pandas or a non-JSON-native leaf still
    produces output.
    """
    if _schema_has_float(table.schema):
        try:
            return _table_to_ndjson_exact(table)
        except (TypeError, ValueError):
            pass  # mixed with a non-JSON-native leaf (e.g. timestamp) — use pandas
    try:
        return _table_to_ndjson(table)
    except Exception:
        return _table_to_ndjson_exact(table)


def _json_pool(n: int) -> Any:
    """A process-lifetime pool for JSON encoding, grown lazily and reused across writes.

    Standing the forkserver pool up once (not per write) is what keeps a JSON write from
    paying ~1s of child-spawn each time; torn down at interpreter exit.
    """
    global _JSON_POOL, _JSON_POOL_SIZE
    from concurrent.futures import ProcessPoolExecutor

    if _JSON_POOL is None or n > _JSON_POOL_SIZE:
        if _JSON_POOL is not None:
            _JSON_POOL.shutdown(wait=False)
        else:
            atexit.register(lambda: _JSON_POOL and _JSON_POOL.shutdown(wait=False))
        from batcher._internal.hardware import process_start_method_context

        ctx = process_start_method_context()
        _JSON_POOL = ProcessPoolExecutor(max_workers=n, mp_context=ctx)
        _JSON_POOL_SIZE = n
    return _JSON_POOL


def _json_encode_shard(task: tuple[bytes, str]) -> str:
    """Worker: decode an IPC chunk, encode it to NDJSON, write it to `out_path`."""
    ipc, out_path = task
    with pa.ipc.open_stream(pa.py_buffer(ipc)) as reader:
        table = reader.read_all()
    with open(out_path, "wb") as fh:
        fh.write(_ndjson_bytes(table))
    return out_path


def _json_write_part(task: tuple[bytes, str, bool]) -> tuple[str, int, int]:
    """Worker: encode an IPC chunk to NDJSON and write it as one output part file.

    Uses the filesystem's atomic writer so a part is either complete or absent (resume-safe
    like the base sink). Returns ``(path, rows, bytes)`` for the manifest.
    """

    ipc, path, resume = task
    fs = resolve_filesystem(path)
    with pa.ipc.open_stream(pa.py_buffer(ipc)) as reader:
        table = reader.read_all()
    if resume and fs.exists(path):
        return path, table.num_rows, _size_or_zero(fs, path)
    data = _ndjson_bytes(table)
    with fs.atomic_writer(path) as fh:
        fh.write(data)
    return path, table.num_rows, len(data)


def _size_or_zero(fs: Any, path: str) -> int:
    try:
        return fs.size(path)
    except (OSError, ValueError):
        return 0


def _json_proc_disabled() -> bool:
    """Whether the process pool has been disabled for this interpreter.

    Exposed as a function, not the flag itself: `_disable_json_proc` rebinds the module
    global, so a caller that imported the *value* would keep reading a stale copy and
    quietly go on using a pool that was meant to be off.
    """
    return _JSON_PROC_DISABLED

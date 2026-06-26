"""The dtype-name ↔ Arrow-type vocabulary — the canonical cast-name table.

Arrow types ARE Batcher's types (see `plan/schema.py`); this module owns the small
*name* vocabulary the public `cast` surface and the JSON IR use to refer to them.
`bc_expr::Expr::Cast` carries the dtype as a raw string on the wire, so this name
set is part of the IR contract with the Rust engine. The canonical table lives in
Rust (`bc_arrow::dtype_from_name`); `DTYPE_REGISTRY` mirrors it here so a bad dtype
fails at plan-build time with a clear message instead of surfacing as an opaque FFI
error mid-execution, and a parity test pins this set to the live engine vocabulary
(`bc-py::supported_cast_dtypes`).
"""

from __future__ import annotations

import pyarrow as pa

__all__ = ["CAST_DTYPES", "DTYPE_REGISTRY"]

# Cast dtype name → Arrow type. Mirrors `bc_arrow::dtype_from_name` exactly,
# including the aliases (`long`/`int64`, `double`/`float64`, ...). Pinned to the
# engine by `tests/unit/test_dtype_registry_parity` so the two cannot drift.
DTYPE_REGISTRY: dict[str, pa.DataType] = {
    "int64": pa.int64(),
    "long": pa.int64(),
    "int32": pa.int32(),
    "int": pa.int32(),
    "float64": pa.float64(),
    "double": pa.float64(),
    "float32": pa.float32(),
    "float": pa.float32(),
    "bool": pa.bool_(),
    "boolean": pa.bool_(),
    "string": pa.string(),
    "utf8": pa.string(),
    "date": pa.date32(),
    "date32": pa.date32(),
    "timestamp": pa.timestamp("us"),
    "datetime": pa.timestamp("us"),
}

# The set of accepted cast dtype names — the spelling the API validates against.
CAST_DTYPES: frozenset[str] = frozenset(DTYPE_REGISTRY)

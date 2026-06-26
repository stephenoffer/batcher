"""The dtype-name vocabulary stays in lockstep across the tiers.

`Expr.cast` carries the target dtype as a raw string on the JSON IR wire, so the
set of accepted names is part of the contract with the Rust engine. The canonical
table lives in `bc_arrow::dtype_from_name`; `plan.types.DTYPE_REGISTRY` mirrors it
on the Python side. This test pins the Python set to the *live* engine vocabulary
(via the `bc-py::supported_cast_dtypes` introspection helper) so the two cannot
drift — a name added to one side without the other fails here, not opaquely at
execution time.
"""

from __future__ import annotations

import batcher._native as _native
import pyarrow as pa

from batcher.plan.types import CAST_DTYPES, DTYPE_REGISTRY


def test_python_cast_dtypes_match_engine_vocabulary():
    engine = set(_native.supported_cast_dtypes())
    assert engine == set(CAST_DTYPES), (
        "plan.types.CAST_DTYPES drifted from bc_arrow::dtype_from_name; "
        f"python-only={set(CAST_DTYPES) - engine}, engine-only={engine - set(CAST_DTYPES)}"
    )


def test_registry_keys_equal_cast_dtypes():
    assert set(DTYPE_REGISTRY) == set(CAST_DTYPES)


def test_registry_aliases_collapse_to_one_type():
    assert DTYPE_REGISTRY["long"] == DTYPE_REGISTRY["int64"] == pa.int64()
    assert DTYPE_REGISTRY["double"] == DTYPE_REGISTRY["float64"] == pa.float64()
    assert DTYPE_REGISTRY["int"] == DTYPE_REGISTRY["int32"] == pa.int32()
    assert DTYPE_REGISTRY["utf8"] == DTYPE_REGISTRY["string"] == pa.string()
    assert DTYPE_REGISTRY["datetime"] == DTYPE_REGISTRY["timestamp"] == pa.timestamp("us")

"""The schema contract reconciles `large_string`/`string`, and only offset-width differences.

cuDF 26.08 returns every string column as `large_string`. On an A10G shadow-verify run of TPC-H
sf10 the contract refused q16 (`p_brand`) and q22 (`cntrycode`) for exactly that, so two correct
device results fell back to the CPU. The values are identical and the cast is lossless, so the
contract now casts such a column to the declared type; every other type difference is still a
refusal, which is the property the contract exists for.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher.api.terminal.gpu_backend.verify import enforce_schema_contract


def _plan():
    return bt.from_pydict({"s": ["a", "b"], "n": [1, 2]})._plan


def test_a_large_string_column_is_returned_as_the_declared_string():
    device = pa.table({"s": pa.array(["a", "b"], pa.large_string()), "n": pa.array([1, 2])})
    out = enforce_schema_contract(device, _plan())
    assert out is not None, "an offset-width difference alone must not refuse a correct result"
    assert out.schema.field("s").type == pa.string()
    assert out.column("s").to_pylist() == ["a", "b"]


def test_a_real_type_difference_is_still_refused():
    device = pa.table({"s": pa.array([1.0, 2.0]), "n": pa.array([1, 2])})
    assert enforce_schema_contract(device, _plan()) is None


def test_a_matching_result_passes_through_unchanged():
    device = pa.table({"s": pa.array(["a", "b"]), "n": pa.array([1, 2])})
    assert enforce_schema_contract(device, _plan()) is device

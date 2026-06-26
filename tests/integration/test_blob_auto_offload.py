"""Automatic blob offload placement around a Sort (opt-in, result-identical).

With ``auto_offload_blobs`` on, a ``large_binary`` column flowing through a sort is
rewritten to ride the breaker as a content-addressed handle and read back after — the
explicit offload/materialize, placed automatically. It must be result-identical to the
plain sort and schema-transparent.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.api.terminal.blob_offload import insert_blob_offload
from batcher.config import Config, ExecutionConfig, config_context
from batcher.plan.logical import MapBatches, Sort

pytestmark = pytest.mark.integration


def _ds():
    return bt.from_arrow(
        pa.table(
            {
                "k": [3, 1, 2],
                "payload": pa.array([b"gamma", b"alpha", b"beta"], pa.large_binary()),
            }
        )
    )


def _auto_on():
    return config_context(Config().replace(execution=ExecutionConfig(auto_offload_blobs=True)))


def test_transform_wraps_sort_over_large_binary(tmp_path):
    plan = _ds().sort("k")._plan
    rewritten = insert_blob_offload(plan, root=str(tmp_path))
    # materialize( Sort( offload(input) ) ): MapBatches over a Sort over a MapBatches.
    assert isinstance(rewritten, MapBatches)
    assert isinstance(rewritten.input, Sort)
    assert isinstance(rewritten.input.input, MapBatches)


def test_transform_skips_sort_keyed_on_blob(tmp_path):
    # If the sort keys on the blob column, it is read by the breaker — not offloadable.
    plan = _ds().sort("payload")._plan
    assert insert_blob_offload(plan, root=str(tmp_path)) is plan


def test_transform_skips_when_no_large_binary(tmp_path):
    plan = bt.from_pydict({"k": [3, 1, 2], "v": [10, 20, 30]}).sort("k")._plan
    assert insert_blob_offload(plan, root=str(tmp_path)) is plan


def test_auto_offload_result_identical():
    expected = _ds().sort("k").collect()
    with _auto_on():
        out = _ds().sort("k").collect()
    assert out.column("k").to_pylist() == expected.column("k").to_pylist() == [1, 2, 3]
    # Payload survives the offload→sort→materialize round-trip, sorted with its row.
    assert out.column("payload").to_pylist() == [b"alpha", b"beta", b"gamma"]
    # Schema-transparent: no leaked handle column.
    assert out.schema.names == expected.schema.names


def test_auto_offload_default_off_is_plain_sort(tmp_path):
    # Default: the plan is a bare Sort, no offload wrapping.
    plan = _ds().sort("k")._plan
    assert isinstance(plan, Sort)

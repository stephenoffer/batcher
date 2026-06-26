"""Blob-by-reference offload/materialize — large payloads ride as URI handles.

`offload_blobs` writes each row's payload to a content-addressed store and replaces
it with a tiny handle (nulling the bytes); `materialize_blobs` reads it back. The two
are inverses, content addressing dedupes, and the handle column is small enough to
shuffle/spill cheaply.
"""

from __future__ import annotations

import hashlib

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.integration


def _ds():
    return bt.from_arrow(
        pa.table(
            {
                "id": [1, 2, 3],
                "bytes": [b"alpha-payload", b"beta-payload", b"alpha-payload"],
            }
        )
    )


def test_offload_replaces_payload_with_handle(tmp_path):
    out = _ds().offload_blobs(root=str(tmp_path)).collect()
    # The payload column is nulled; a uri handle column is added.
    assert out.column("bytes").to_pylist() == [None, None, None]
    uris = out.column("uri").to_pylist()
    assert all(u and str(tmp_path) in u for u in uris)
    # Content-addressed: identical payloads (rows 0 and 2) get the SAME handle.
    assert uris[0] == uris[2]
    assert uris[0] != uris[1]
    # The handle is the SHA-256 of the payload.
    assert uris[0].endswith(hashlib.sha256(b"alpha-payload").hexdigest())


def test_offload_materialize_round_trip(tmp_path):
    original = _ds().collect()
    restored = _ds().offload_blobs(root=str(tmp_path)).materialize_blobs().collect()
    # Round-trip is identity on the payloads (modulo binary → large_binary).
    assert restored.column("bytes").to_pylist() == original.column("bytes").to_pylist()
    assert restored.column("id").to_pylist() == original.column("id").to_pylist()


def test_handle_column_is_small(tmp_path):
    # The point of offload: the handle column is tiny vs the inline payloads, so it
    # crosses a shuffle/spill cheaply. After offload, bytes carries no payload.
    out = _ds().offload_blobs(root=str(tmp_path)).collect()
    assert out.schema.field("uri").type == pa.string()
    assert out.column("bytes").null_count == out.num_rows


def test_dedup_writes_one_file_per_distinct_payload(tmp_path):
    _ds().offload_blobs(root=str(tmp_path)).collect()
    # Two distinct payloads → exactly two files in the content-addressed store.
    files = list(tmp_path.iterdir())
    assert len(files) == 2

"""Regression tests for object-store URI prefix mapping in `io.filesystem`.

`pyarrow.fs.FileSystem.from_uri` strips a trailing slash from the in-filesystem path
(``s3://bucket/dir/`` to in_path ``bucket/dir``). The prefix is then `base minus in_path`;
computed off the un-trimmed base it mis-slices by the slash (``s3://`` → ``s3://r``),
so every later `_p()` drops a real character and listings fail with "does not exist".
"""

from __future__ import annotations

import pytest

from batcher.io import filesystem as fsmod

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("trailing", ["", "/"])
def test_object_store_prefix_survives_trailing_slash(monkeypatch, trailing):
    bucket_path = "ray-benchmark-data/tpch/lineitem"
    uri = f"s3://{bucket_path}{trailing}"

    class _FakeS3:
        type_name = "s3"

    # `pyarrow.fs.FileSystem` is an immutable C type, so patch the module's reference to
    # it. Mirror pyarrow: `from_uri` returns the fs and a trailing-slash-stripped in_path.
    class _FakeFS:
        @staticmethod
        def from_uri(_p):
            return _FakeS3(), bucket_path

    monkeypatch.setattr(fsmod.pafs, "FileSystem", _FakeFS)
    fs = fsmod.resolve_filesystem(uri)
    # The prefix must be exactly the scheme+authority, so mapping the original URI back
    # to an in-filesystem path recovers the bucket key (modulo the trailing slash).
    assert fs._prefix == "s3://"
    assert fs._p(uri).rstrip("/") == bucket_path

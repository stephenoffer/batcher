"""Regression tests for object-store URI prefix mapping in `io.filesystem`.

`pyarrow.fs.FileSystem.from_uri` strips a trailing slash from the in-filesystem path
(``s3://bucket/dir/`` to in_path ``bucket/dir``). The prefix is then `base minus in_path`;
computed off the un-trimmed base it mis-slices by the slash (``s3://`` → ``s3://r``),
so every later `_p()` drops a real character and listings fail with "does not exist".

Azure breaks the assumption underneath all of that — that the backend's path is a *suffix*
of the URI. It is not: the container is in the authority, so
``abfs://c@a.dfs.core.windows.net/dir/k`` maps to ``c/dir/k``. Subtracting `len(in_path)`
then sliced mid-hostname and silently sent every read, list, and **write** to a wrong
container/key, which is why these tests assert the exact mapped path, not just its length.
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
    # it. Mirror pyarrow: `from_uri` returns the fs and the in_path (everything after
    # `scheme://`), trailing slash stripped — resolve_filesystem now reduces the URI to
    # `scheme://authority` first, so the object path's trailing slash never reaches the
    # prefix math (the more robust fix this test guards).
    class _FakeFS:
        @staticmethod
        def from_uri(p):
            return _FakeS3(), p.split("://", 1)[1].rstrip("/")

    monkeypatch.setattr(fsmod.pafs, "FileSystem", _FakeFS)
    fsmod._resolve_uri_fs.cache_clear()  # isolate from other resolutions / parametrize runs
    fs = fsmod.resolve_filesystem(uri)
    # The prefix must be exactly the scheme+authority, so mapping the original URI back
    # to an in-filesystem path recovers the bucket key (modulo the trailing slash).
    assert fs._prefix == "s3://"
    assert fs._p(uri).rstrip("/") == bucket_path
    # A suffix-shaped scheme prepends nothing — the mapping is a pure prefix strip.
    assert fs._root == ""


# `abfs`/`abfss` URIs and the in-filesystem path pyarrow's AzureFileSystem reports for
# each: the container moves out of the authority and in front of the key.
_AZURE_CASES = [
    ("abfs://c@a.dfs.core.windows.net/dir/k.parquet", "c/dir/k.parquet"),
    ("abfss://c@a.dfs.core.windows.net/dir/k.parquet", "c/dir/k.parquet"),
    ("abfs://c@a.dfs.core.windows.net/k.parquet", "c/k.parquet"),
]


@pytest.mark.parametrize(("uri", "expected"), _AZURE_CASES)
def test_azure_container_in_authority_maps_to_container_prefixed_path(monkeypatch, uri, expected):
    """The container lives in the authority, so `in_path` is NOT a suffix of the URI."""

    class _FakeAzure:
        type_name = "abfs"

    class _FakeFS:
        @staticmethod
        def from_uri(p):
            # Mirror pyarrow: `abfs://<container>@<account>.dfs…/<key>` → `<container>/<key>`,
            # and a bare authority → `<container>/`.
            rest = p.split("://", 1)[1].split("?", 1)[0]
            authority, _, key = rest.partition("/")
            return _FakeAzure(), f"{authority.split('@', 1)[0]}/{key}"

    monkeypatch.setattr(fsmod.pafs, "FileSystem", _FakeFS)
    fsmod._resolve_uri_fs.cache_clear()
    fs = fsmod.resolve_filesystem(uri)
    assert fs._p(uri) == expected
    # And the inverse must land back on the caller-visible URI, or listings return paths
    # no caller can re-open.
    assert fs._uri(expected) == uri


def test_unimplemented_scheme_is_reported_as_unsupported(monkeypatch):
    """ "Protocol not known" is the *only* signal that a scheme is unimplemented."""
    import fsspec

    def _unknown(protocol, *a, **k):
        raise ValueError(f"Protocol not known: {protocol}")

    monkeypatch.setattr(fsspec, "filesystem", _unknown)
    fsmod._resolve_uri_fs.cache_clear()
    with pytest.raises(fsmod.IOError, match="unsupported storage scheme"):
        fsmod.resolve_filesystem("wasb://c@a.blob.core.windows.net/k")


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("unable to connect to account for Must provide a connection_string"),
        OSError("Unable to load libjvm"),
    ],
)
def test_backend_construction_failure_keeps_its_own_message(monkeypatch, exc):
    """A backend that exists but cannot be built is NOT an unsupported scheme.

    adlfs raises ValueError when Azure credentials are missing and the HDFS driver raises
    OSError when libjvm will not load — both share the exception types of "protocol not
    known". Calling either "unsupported scheme" sends the user hunting for a missing
    feature instead of fixing their credentials or their JVM, so the backend's own message
    has to survive."""

    import fsspec

    def _boom(protocol, *a, **k):
        raise exc

    monkeypatch.setattr(fsspec, "filesystem", _boom)
    fsmod._resolve_uri_fs.cache_clear()
    with pytest.raises(fsmod.IOError) as caught:
        fsmod.resolve_filesystem("az://c@a.dfs.core.windows.net/k")
    assert "unsupported storage scheme" not in str(caught.value)
    assert str(exc) in str(caught.value)


@pytest.mark.parametrize(("uri", "expected"), _AZURE_CASES)
def test_azure_mapping_against_real_pyarrow(uri, expected):
    """The same contract against the real AzureFileSystem, so the fake above can't drift."""
    import pyarrow as pa

    try:
        pa.fs.FileSystem.from_uri(uri)
    except Exception as e:  # pragma: no cover - pyarrow built without the Azure backend
        pytest.skip(f"pyarrow has no usable Azure backend here: {e}")
    fsmod._resolve_uri_fs.cache_clear()
    fs = fsmod.resolve_filesystem(uri)
    assert fs._p(uri) == expected
    assert fs._uri(expected) == uri


@pytest.mark.parametrize(
    "query",
    [
        # Path-style addressing + a plain-HTTP endpoint: the MinIO/Ceph default shape.
        "force_virtual_addressing=false&endpoint_override=http://minio.internal:9000",
        "anonymous=true",
        "access_key=AK&secret_key=SK&region=us-east-1",
        "connect_timeout=30&request_timeout=60",
    ],
)
def test_on_prem_s3_options_are_supported_not_refused(query):
    """`from_uri` takes a narrow option set; the `S3FileSystem` constructor takes the ones
    an on-prem store actually needs. Refusing them left path-style-only and
    self-hosted-credential deployments with no in-URI escape hatch — and, before that,
    silently folded the rejected option into the object key."""
    uri = f"s3://bucket/dir/k.parquet?{query}"
    fsmod._resolve_uri_fs.cache_clear()
    fs = fsmod.resolve_filesystem(uri)
    assert fs._p(uri) == "bucket/dir/k.parquet"
    assert fs._uri("bucket/dir/k.parquet") == "s3://bucket/dir/k.parquet"


def test_an_unknown_s3_option_is_named_rather_than_ignored():
    """`S3FileSystem` would accept the call and quietly drop an unknown kwarg, leaving the
    user debugging a connection that used none of their settings."""
    fsmod._resolve_uri_fs.cache_clear()
    with pytest.raises(fsmod.IOError, match="unknown s3:// option"):
        fsmod.resolve_filesystem("s3://bucket/k.parquet?definitely_not_an_option=1")


@pytest.mark.parametrize("alias", ["s3a", "gcs"])
def test_scheme_aliases_take_the_native_backend(alias):
    """The alias used to be applied only *after* `from_uri` had already failed, so every
    Hadoop-spelled URI silently took the slower fsspec path."""
    native = {"s3a": "S3FileSystem", "gcs": "GcsFileSystem"}[alias]
    uri = f"{alias}://bucket/d/x.parquet"
    fsmod._resolve_uri_fs.cache_clear()
    fs = fsmod.resolve_filesystem(uri)
    assert type(fs._fs).__name__ == native
    assert fs._p(uri) == "bucket/d/x.parquet"
    assert fs._uri("bucket/d/x.parquet") == uri

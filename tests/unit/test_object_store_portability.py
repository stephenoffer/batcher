"""Object stores outside the three hyperscalers, and the two ways they were mis-served.

Batcher reaches Alibaba OSS, Tencent COS, Huawei OBS, Oracle OCI, OpenStack Swift, lakeFS and
any in-house backend through the fsspec fallback. Both defects here were silent on exactly
those platforms and invisible on AWS, GCS and Azure, which is why they survived:

* **Credentials were dropped.** `storage_options` was folded into the URI query, which is how
  a *native* pyarrow backend is configured. An fsspec backend takes keyword arguments, so a
  read with perfectly correct keys failed as though it had none — and the folded query became
  part of the object key it asked for.
* **Writes did a full server-side copy.** A scheme missing from the object-store set is treated
  as a filesystem, so every write went temp-then-rename. On an object store that rename is a
  copy of the whole object, and it is not atomic either, so the cost bought nothing.
"""

from __future__ import annotations

import pytest

import batcher.io.filesystem as fsmod
from batcher._internal.errors import IOError as BatcherIOError

pytestmark = pytest.mark.unit


@pytest.fixture
def fsspec_spy(monkeypatch):
    """Capture what the fsspec fallback constructs, without needing any driver installed."""
    import fsspec

    calls: list[tuple[str, dict]] = []

    def spy(protocol, **kwargs):
        calls.append((protocol, kwargs))
        raise ValueError("Protocol not known: spy")

    monkeypatch.setattr(fsspec, "filesystem", spy)
    fsmod._resolve_uri_fs.cache_clear()
    yield calls
    fsmod._resolve_uri_fs.cache_clear()


def test_storage_options_reach_an_fsspec_backend_as_keyword_arguments(fsspec_spy):
    with pytest.raises(BatcherIOError):
        fsmod.resolve_filesystem(
            "oss://bucket/data/x.parquet",
            storage_options={"key": "AK", "secret": "SK", "endpoint": "oss-cn-hangzhou.example"},
        )
    assert fsspec_spy == [
        ("oss", {"key": "AK", "secret": "SK", "endpoint": "oss-cn-hangzhou.example"})
    ]


def test_a_secret_reference_is_resolved_before_it_reaches_the_backend(fsspec_spy, monkeypatch):
    # The split that rides to a distributed worker carries the reference, never the key.
    monkeypatch.setenv("BATCHER_TEST_OSS_SECRET", "the-real-key")
    with pytest.raises(BatcherIOError):
        fsmod.resolve_filesystem(
            "cos://bucket/x", storage_options={"secret": "env:BATCHER_TEST_OSS_SECRET"}
        )
    assert fsspec_spy[0][1] == {"secret": "the-real-key"}


def test_an_option_the_backend_does_not_take_is_named_not_swallowed(monkeypatch):
    import fsspec

    def spy(protocol, **kwargs):
        raise TypeError("__init__() got an unexpected keyword argument 'nonsense'")

    monkeypatch.setattr(fsspec, "filesystem", spy)
    fsmod._resolve_uri_fs.cache_clear()
    with pytest.raises(BatcherIOError) as exc:
        fsmod.resolve_filesystem("obs://bucket/x", storage_options={"nonsense": "1"})
    assert "does not accept" in str(exc.value)
    fsmod._resolve_uri_fs.cache_clear()


def test_a_secret_containing_a_separator_survives_the_uri_fold(monkeypatch):
    # A native backend is configured through the query, and a raw `&` or `/` in a key
    # truncates it — which presents as an authentication failure with a correct key.
    built: dict[str, object] = {}

    def fake_s3(**kwargs):
        built.update(kwargs)
        raise ValueError("stop")

    monkeypatch.setattr(fsmod.pafs, "S3FileSystem", fake_s3)
    fsmod._resolve_uri_fs.cache_clear()
    secret = "abc&def/ghi+jkl"
    with pytest.raises(BatcherIOError):
        fsmod.resolve_filesystem(
            "s3://bucket/x",
            storage_options={"secret_key": secret, "endpoint_override": "http://minio:9000"},
        )
    assert built.get("secret_key") == secret
    fsmod._resolve_uri_fs.cache_clear()


@pytest.mark.parametrize(
    "scheme", ["oss", "cos", "cosn", "obs", "oci", "swift", "lakefs", "adl", "s3n"]
)
def test_every_object_store_scheme_writes_in_place_rather_than_copying(scheme):
    # `atomic_rename=False` is what stops a write becoming temp-write plus a full server-side
    # object copy — which is what a scheme missing from this set silently costs.
    canonical = fsmod._SCHEME_ALIASES.get(scheme, scheme)
    assert canonical in fsmod._OBJECT_STORE_SCHEMES or scheme in fsmod._OBJECT_STORE_SCHEMES


def test_the_legacy_hadoop_s3_spelling_takes_the_native_backend():
    # `s3n://` still appears in inherited manifests and job configs. Without the alias it took
    # the fsspec path: correct, slower, and needing a driver installed.
    fsmod._resolve_uri_fs.cache_clear()
    fs = fsmod.resolve_filesystem("s3n://bucket/d/x.parquet")
    assert type(fs._fs).__name__ == "S3FileSystem"
    assert fs._p("s3n://bucket/d/x.parquet") == "bucket/d/x.parquet"
    fsmod._resolve_uri_fs.cache_clear()


def test_a_bring_your_own_object_store_filesystem_is_not_rename_published():
    # The same rule on the hand-in path: a user's own OSS/COS handle must not be published
    # through a rename either.
    class FakeFsspec:
        def _strip_protocol(self, path):
            return path.split("://", 1)[1]

    fs = fsmod.resolve_filesystem("oss://bucket/k.parquet", filesystem=FakeFsspec())
    assert fs._atomic_rename is False

"""A native Parquet failure falls back to PyArrow *and leaves a trace*; an absent engine does not.

Every native entry point used to end in `except Exception: return None`, so a fast path that
had been broken for months looked exactly like one that never applied. The fallback is right;
the silence was not.
"""

from __future__ import annotations

import pytest

from batcher.io.formats.structured import _parquet_native as native

pytestmark = pytest.mark.unit


class _Broken:
    def __getattr__(self, name):
        def fail(*_a, **_k):
            raise RuntimeError(f"{name} exploded")

        return fail


_CALLS = {
    "read_one": lambda: native.read_one("s3://b/f.parquet", None),
    "read_many": lambda: native.read_many(["s3://b/f.parquet"], None),
    "read_row_groups_filtered": lambda: native.read_row_groups_filtered(
        "s3://b/f.parquet", [0], None, None
    ),
    "footer_stats": lambda: native.footer_stats(["s3://b/f.parquet"]),
    "file_manifest": lambda: native.file_manifest(["s3://b/f.parquet"], ["x"]),
}


@pytest.fixture
def traced(monkeypatch):
    calls: list[tuple[str, str, BaseException]] = []
    monkeypatch.setattr(native, "note_suppressed", lambda *a: calls.append(a))
    return calls


@pytest.mark.parametrize("name", sorted(_CALLS))
def test_a_native_failure_is_traced(monkeypatch, traced, name):
    monkeypatch.setattr(native, "engine_or_none", lambda: _Broken())
    assert _CALLS[name]() is None
    assert [(sub, step) for sub, step, _ in traced] == [("io", f"native parquet {name}")]
    assert isinstance(traced[0][2], RuntimeError)


@pytest.mark.parametrize("name", sorted(_CALLS))
def test_an_absent_engine_falls_back_silently(monkeypatch, traced, name):
    monkeypatch.setattr(native, "engine_or_none", lambda: None)
    assert _CALLS[name]() is None
    assert traced == []

"""The IO format registry is complete — every built-in format is registered.

`io/formats/` is the grouped-by-family + registry scaffolding: a new format is one
file that registers a `SourceFormat`/`SinkFormat`. This guards that the three
built-ins stay wired (a missing registration would silently drop a format).
"""

from __future__ import annotations

import batcher.io  # noqa: F401 -- import triggers format registration
from batcher.io.formats.base import SINKS, SOURCES

BUILTIN_FORMATS = {"parquet", "csv", "json"}


def test_builtin_sources_registered():
    assert set(SOURCES.names()) >= BUILTIN_FORMATS


def test_builtin_sinks_registered():
    assert set(SINKS.names()) >= BUILTIN_FORMATS


def test_registered_formats_resolve():
    for name in BUILTIN_FORMATS:
        assert SOURCES.get(name) is not None
        assert SINKS.get(name) is not None

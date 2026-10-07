"""Every reader and writer docstring names the extra its connector needs (AP-014).

A missing driver already raises `MissingDependencyError` naming the extra. This holds the
other half: `help(bt.read.mongo)` says ``pip install 'batcher-engine[mongo]'`` before the
first call fails. The extra is read from the connector's own module, the same string its
`require` call raises with, so the docstring cannot drift from the error.
"""

from __future__ import annotations

import inspect
import re
import sys

import pytest

from batcher.api.io_namespace.reader import Reader
from batcher.api.io_namespace.writer import Writer
from batcher.io.formats.base import SOURCES
from batcher.io.sink import SINKS

#: How a connector module names its extra: a `require(..., extra="x")` call, a module-level
#: `_EXTRA = "x"` handed to `require_module`, or the NoSQL `require_driver(module, "x")`.
_PATTERNS = (
    re.compile(r'extra\s*=\s*"([a-z0-9_-]+)"'),
    re.compile(r'_EXTRA\s*=\s*"([a-z0-9_-]+)"'),
    re.compile(r'require_driver\(\s*"[^"]+",\s*"([a-z0-9_-]+)"\)'),
)

#: ``all`` guards a pyarrow build feature (ORC, IPC), not an installable connector.
_NOT_A_CONNECTOR_EXTRA = {"all"}

#: Methods whose docstring deliberately does not advertise their module's extra.
_EXEMPT = {
    # Always raises: Hudi writes need Spark/Flink, so naming an install would mislead.
    ("write", "hudi"),
}


def _extras(cls: type) -> set[str]:
    source = inspect.getsource(sys.modules[cls.__module__])
    found = {m for pattern in _PATTERNS for m in pattern.findall(source)}
    return found - _NOT_A_CONNECTOR_EXTRA


def _cases() -> list[tuple[str, str, str]]:
    cases = []
    for label, registry, namespace in (("read", SOURCES, Reader), ("write", SINKS, Writer)):
        for name in sorted(registry.names()):
            cls = registry.get(name)
            if not isinstance(cls, type) or getattr(namespace, name, None) is None:
                continue
            if (label, name) in _EXEMPT:
                continue
            cases.extend((label, name, extra) for extra in sorted(_extras(cls)))
    return cases


_CASES = _cases()


def test_the_walk_found_connectors():
    # The positive control: a parametrize over an empty walk would pass vacuously.
    assert ("read", "mongo", "mongo") in _CASES
    assert len(_CASES) > 20


@pytest.mark.parametrize(("label", "name", "extra"), _CASES)
def test_the_docstring_names_the_extra(label, name, extra):
    namespace = Reader if label == "read" else Writer
    doc = getattr(namespace, name).__doc__ or ""
    assert f"batcher-engine[{extra}]" in doc, (
        f"{label}.{name}'s docstring should say pip install 'batcher-engine[{extra}]'"
    )

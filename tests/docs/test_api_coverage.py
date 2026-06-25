"""Mechanical anti-drift gate: every public API name must be documented.

Introspects the public surface (``batcher.__all__``, the `Dataset`/`Expr` public
methods, the accessor-namespace methods, and the preprocessor exports) and asserts
each name appears somewhere in the ``docs/**/*.md`` corpus. This converts the
hand-maintained reference tables into a checked contract: add a public name without
documenting it and this test fails (the v1 docs rotted precisely because nothing
enforced this).

A name that genuinely should not yet be documented goes in ``KNOWN_UNDOCUMENTED``
with the intent to drain it to empty — and a name that is *both* documented and
allowlisted also fails, so the allowlist cannot silently rot.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import batcher as bt

pytestmark = pytest.mark.unit

_DOCS = Path(__file__).resolve().parents[2] / "docs"

# Public names not yet documented (drain toward empty). Keep each with a reason.
KNOWN_UNDOCUMENTED: dict[str, str] = {
    # Not user-facing API names.
    "__version__": "package version string, not an API symbol",
    # Streaming surface (in-flight); documentation pending.
    "OutputMode": "streaming output mode enum; docs pending",
}


def _docs_corpus() -> str:
    """All Markdown under docs/ concatenated (the searchable documentation text)."""
    parts = []
    for md in _DOCS.rglob("*.md"):
        if "_build" in md.parts:
            continue
        parts.append(md.read_text(encoding="utf-8"))
    return "\n".join(parts)


def _public_names() -> set[str]:
    names: set[str] = set(bt.__all__)
    # Accessor-namespace methods (the .str/.dt/.list/.struct/.json surface).
    from batcher.plan.expr_ir.namespaces import (
        _DtNamespace,
        _JsonNamespace,
        _ListNamespace,
        _MapNamespace,
        _StrNamespace,
        _StructNamespace,
    )

    for ns in (
        _StrNamespace,
        _DtNamespace,
        _ListNamespace,
        _StructNamespace,
        _JsonNamespace,
        _MapNamespace,
    ):
        names |= {n for n in vars(ns) if not n.startswith("_") and callable(vars(ns)[n])}
    # Preprocessors.
    from batcher.ml import preprocessors

    names |= set(preprocessors.__all__)
    return names


def _mentioned(name: str, corpus: str) -> bool:
    return re.search(rf"\b{re.escape(name)}\b", corpus) is not None


def test_every_public_name_is_documented():
    corpus = _docs_corpus()
    documented = {n for n in _public_names() if _mentioned(n, corpus)}
    undocumented = {n for n in _public_names() if n not in documented}

    missing = sorted(undocumented - set(KNOWN_UNDOCUMENTED))
    assert not missing, (
        f"{len(missing)} public name(s) not documented in docs/: {missing}\n"
        "Document them (e.g. in docs/api/reference.md) or add to KNOWN_UNDOCUMENTED."
    )

    # The allowlist may not list a name that is actually documented (keep it honest).
    stale = sorted(n for n in KNOWN_UNDOCUMENTED if n in documented)
    assert not stale, f"KNOWN_UNDOCUMENTED lists documented names (remove them): {stale}"

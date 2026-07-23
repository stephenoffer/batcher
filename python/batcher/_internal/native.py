"""The single accessor for the compiled Rust data plane (``batcher._native``).

Every control-plane module that needs the engine goes through here. This is the one
place in the Python tree that names the compiled extension.

**Why this module exists — do not bypass it.** Writing ``import batcher._native as nat``
in a module makes the static import graph resolve the compiled extension to the *root*
``batcher`` package: `grimp` (behind `import-linter`) cannot see into a ``.so``, so it
attributes the import to the nearest package it *can* see. The root package re-exports
``api``, and ``api`` imports ``kyber``, ``carbonite``, ``core``, and ``governance`` — so
every hand-rolled ``import batcher._native`` forges a cycle
(``core -> batcher -> api -> kyber``) that silently breaks the layer-independence
contract in `.claude/rules/architecture.md`. Loading the extension through
:func:`importlib.import_module` here keeps that phantom edge out of the graph entirely,
so the contract measures real layering instead of an artefact of the FFI.

The accessors are lazy: the extension is imported on first use, not at module import, so
a control-plane-only process (docs build, optimizer unit tests) never pays for it and
never fails on a tree that has not been built yet.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from batcher._internal.errors import BackendError

__all__ = ["engine", "engine_or_none", "has_engine"]


def engine_or_none() -> ModuleType | None:
    """Return the compiled engine module, or ``None`` when it is not built.

    Use this on *best-effort* paths — statistics collection, native fast paths with a
    pure-Python fallback — where an unbuilt engine must degrade rather than raise.

    Deliberately **not** memoized. ``importlib.import_module`` already resolves an imported
    module with a ``sys.modules`` dict lookup, so a cache would buy nothing measurable — and it
    would cost the ability to substitute the engine: the unit tests that exercise the
    distributed reducers install a stub ``batcher._native`` in ``sys.modules`` and drive the
    real orchestration against it. A module-level cache captured the first (real) engine and
    silently ignored the stub. An accessor that cannot be stubbed makes the code it fronts
    untestable, which is a worse cost than a dict lookup.

    Returns:
        The ``batcher._native`` module, or ``None`` if it cannot be imported.
    """
    try:
        return importlib.import_module("batcher._native")
    except ImportError:
        return None


def engine() -> ModuleType:
    """Return the compiled engine module, raising ``BackendError`` if it is missing.

    Use this on paths that *require* the data plane — executing a plan, driving a
    shuffle — where there is no meaningful fallback.

    Returns:
        The ``batcher._native`` module.

    Raises:
        BackendError: The compiled engine is not importable.
    """
    mod = engine_or_none()
    if mod is None:
        raise BackendError(
            "the compiled Batcher engine (`batcher._native`) is not available; "
            "build it into the environment with `just build`"
        )
    return mod


def has_engine() -> bool:
    """Return whether the compiled engine is importable.

    Returns:
        ``True`` when ``batcher._native`` can be imported, ``False`` otherwise.
    """
    return engine_or_none() is not None

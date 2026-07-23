"""The one optional-dependency import guard.

Batcher keeps its base install small and puts every connector's driver behind an
extra. Each of those call sites needs the same three things: import the driver
lazily (so importing the format module never drags the driver in), and if it is
absent raise a typed error naming both the feature and the extra that installs it —
never a bare `ImportError`, which tells a user nothing actionable.

That guard had been written out by hand in twenty-odd modules, all producing the
same sentence. `require` is that sentence, once. It raises
`MissingDependencyError`, which reads

    "{feature} requires {provides}, which is not installed. Install it with:
         pip install 'batcher-engine[{extra}]'"

and carries that command in an `install` field so a caller can surface it its own
way. It remains a `BackendError`, so the handlers written against the older shape
keep working, and it is now also an `ImportError`, so the spelling a user reaches
for around an optional import works too.

This lives in `_internal` (layer 0) rather than `io/` because `io/` is at its
directory-size limit and because the guard is not IO-specific — anything holding
an optional dependency can use it.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

from batcher._internal.errors import MissingDependencyError

__all__ = ["require"]


def require(
    module: str,
    attr: str | None = None,
    *,
    feature: str,
    provides: str,
    extra: str,
) -> Any | ModuleType:
    """Import an optional dependency, or raise a typed install hint.

    Args:
        module: Importable module name (e.g. ``"pyarrow.orc"``, ``"hudi"``).
        attr: When given, the attribute to pull off the module (e.g. ``"HudiTable"``)
            instead of returning the module itself.
        feature: What the user was trying to do, as it should read at the start of
            the error (e.g. ``"Hudi read support"``).
        provides: The distribution or package that supplies it, named the way a user
            would recognize it (e.g. ``"hudi-rs"`` — which is not the module name).
        extra: The Batcher extra that installs it (e.g. ``"hudi"``).

    Returns:
        The imported module, or `attr` from it when `attr` is given.

    Raises:
        MissingDependencyError: If the dependency is not installed. It is a
            `BackendError` (so existing handlers keep working) and an `ImportError`
            (so ``except ImportError`` around an optional feature works too), and it
            carries the exact install command in its `install` field.
    """
    try:
        mod = importlib.import_module(module)
        if attr is None:
            return mod
        try:
            return getattr(mod, attr)
        except AttributeError:
            # `attr` is a submodule that the parent does not import eagerly (e.g.
            # `pyiceberg.catalog`). This is the fallback `from module import attr`
            # performs, and without it the guard raises AttributeError instead of
            # returning the module the caller asked for.
            return importlib.import_module(f"{module}.{attr}")
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        # `MissingDependencyError` is itself an `ImportError`, so it must be raised from
        # the handler (as here) and never from inside the guarded `try` body — moving
        # this line up would make the guard catch its own failure.
        raise MissingDependencyError.of(feature=feature, provides=provides, extra=extra) from exc

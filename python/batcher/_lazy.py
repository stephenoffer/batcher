"""PEP 562 lazy re-export façades, shared by every package that is one.

`batcher`, `batcher.api`, and `batcher.api.session` are all pure re-export façades:
they bind several hundred names imported from leaf modules and declare an `__all__`.
Doing that eagerly means importing the *entire* control plane to reach any one name —
545 ms and 609 modules for a script that wanted `bt.col`, and a fixed cost paid by
every process in a distributed run.

This module turns such a façade into one that imports a name's defining module on
first touch and nothing else. The routing tables live in `batcher._exports`, generated
by `just gen-exports` and gated by `tests/unit/test_lazy_exports.py`.

Python imports a package's ancestors before the package itself, so laziness is only
worth anything if *every* façade on the path has it: routing `from_pydict` straight to
`api.session.frames` saves nothing while importing `batcher.api` still executes an
eager `from batcher.api.functions import *`.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Mapping, Sequence
from types import ModuleType

__all__ = ["install"]


class _Facade(ModuleType):
    """A façade module whose routed names outrank same-named submodules.

    Importing `batcher.api.security` binds `security` onto `batcher.api` as a side
    effect, and `batcher.api.security` is a *package* while `security` is the function
    the façade exports. The eager façade won that race by construction — its star
    imports ran after the submodule imports, so the function was bound last. Laziness
    removes that ordering, and the loser is whichever the importing script happened to
    touch first: `from batcher.api import security` would hand back a module.

    Five names collide this way (`security`, and `read`/`sql`/`versions`/`accelerators`
    under `api.session`), so those are resolved through the table on every access and
    cached outside `__dict__`, where a later submodule import cannot overwrite them.
    Every other name keeps the plain `__getattr__` path and is bound into `__dict__`
    once, so this costs nothing for the other 674.
    """

    __slots__ = ()

    def __getattribute__(self, name: str) -> object:
        namespace = object.__getattribute__(self, "__dict__")
        shadowed = namespace.get("__lazy_shadowed__")
        if shadowed is not None and name in shadowed:
            cache = namespace["__lazy_cache__"]
            if name not in cache:
                cache[name] = namespace["__lazy_resolve__"](name)
            return cache[name]
        # `ModuleType`'s, not `object`'s: PEP 562's fallback to a module-level
        # `__getattr__` lives in the module type's own lookup, so going straight to
        # `object` would silently disable laziness for every non-shadowed name.
        return ModuleType.__getattribute__(self, name)


def install(
    module_name: str,
    exports: Mapping[str, str],
    *,
    subpackages: Sequence[str] = (),
    shadowed: Sequence[str] = (),
    on_missing: Callable[[str, list[str]], Exception] | None = None,
) -> tuple[Callable[[str], object], Callable[[], list[str]]]:
    """Build the `__getattr__`/`__dir__` pair implementing a lazy façade.

    Args:
        module_name: The façade's own `__name__`, used to bind resolved names back
            into its namespace so each is imported at most once.
        exports: Public name -> ``"module"`` or ``"module:attr"``. The second form is a
            name renamed on re-export, such as `concat_str` from `string.building.concat`.
        subpackages: Public subpackages reachable as an attribute of the façade and
            imported on first touch, such as `batcher.ml`.
        shadowed: Exported names that are *also* submodule names of this package, and
            so would be overwritten by the import machinery. Resolved through the table
            on every access rather than bound into the module namespace.
        on_missing: Builds the error raised for an unknown name, given the name and the
            façade's `__all__`. Defaults to a plain `AttributeError`.

    Returns:
        The `__getattr__` and `__dir__` functions to assign in the façade module.

    Examples:
        .. doctest::

            >>> from batcher._lazy import install
            >>> getter, lister = install("batcher", {"col": "batcher.plan.expr_ir"})
            >>> lister()
            ['col']
    """
    exports = dict(exports)
    subpackages = tuple(subpackages)

    def _resolve(name: str) -> object:
        """Import the module defining `name` and read it out of there."""
        module_path, _, attr = exports[name].partition(":")
        return getattr(importlib.import_module(module_path), attr or name)

    if shadowed:
        module = sys.modules[module_name]
        module.__dict__["__lazy_shadowed__"] = frozenset(shadowed)
        module.__dict__["__lazy_cache__"] = {}
        module.__dict__["__lazy_resolve__"] = _resolve
        module.__class__ = _Facade

    def __getattr__(name: str) -> object:
        # Dunder and private probes (import machinery, IPython, copy) must fail plainly,
        # and must never be routed: a façade that answers `__path__` or `__all__` from a
        # table breaks the import system in ways that surface far from the cause.
        if name.startswith("_"):
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        namespace = sys.modules[module_name].__dict__
        if name in exports:
            value = _resolve(name)
            namespace[name] = value  # bind it, so the next lookup never reaches here
            return value
        if name in subpackages:
            module = importlib.import_module(f"{module_name}.{name}")
            namespace[name] = module
            return module
        if on_missing is not None:
            raise on_missing(name, list(namespace.get("__all__", ())))
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")

    def __dir__() -> list[str]:
        # Spelled out because laziness means most of these names are not in the module's
        # `globals()` until something touches them, and `dir()` is how a user discovers
        # the surface in a REPL.
        return sorted({*exports, *subpackages})

    return __getattr__, __dir__

"""Attach the compatibility aliases onto `Expr`.

The alias functions live in `names` and `operators` as plain module-level functions;
this module binds each onto the `Expr` class at import time, the same way the typed
accessors are bound in `namespaces/_bind.py`. Kept out of the package ``__init__`` so
that façade stays a re-export only.
"""

from __future__ import annotations

from batcher.plan.expr_ir.compat import names, operators

__all__ = ["bind_compat_methods"]

_MODULES = (names, operators)


def bind_compat_methods(cls: type) -> None:
    """Attach every compatibility alias in this package onto `cls`.

    Args:
        cls: The `Expr` class to bind the aliases onto.

    Returns:
        None. `cls` gains one method per name exported by the alias modules.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.col("x").isna().to_ir() == bt.col("x").is_null().to_ir()
            True
    """
    for module in _MODULES:
        for name in module.__all__:
            func = getattr(module, name)
            func.__qualname__ = f"{cls.__name__}.{name}"
            # `__module__` must move with `__qualname__`. Sphinx resolves a method's
            # owning class by looking `Expr` up in `sys.modules[func.__module__]`, and
            # these functions are defined in `compat.*` where `Expr` exists only under
            # `TYPE_CHECKING`. Leaving the pair inconsistent made autodoc raise
            # "module ... has no attribute 'Expr'" for every alias, which fails the
            # docs build under `-W`.
            func.__module__ = cls.__module__
            setattr(cls, name, func)

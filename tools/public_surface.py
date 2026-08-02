"""The definition of Batcher's public API surface, in one place.

Two independent gates need the same answer to "what is public?": the documentation
coverage test (``tests/docs/test_api_coverage.py``, which asserts every public name
is *mentioned* in ``docs/``) and the docstring style linter
(``tools/lint_docstrings.py``, which asserts every public callable is *documented*
to the project's Google style). Deriving both from this module keeps them from
drifting apart, and makes "we added a public name" a single-edit event.

The surface is what a user can reach from ``import batcher as bt``: the curated
``batcher.__all__``, the fluent builder classes it exposes, the typed accessor
namespaces those classes hand out (``col("x").str``, ``ds.ml``, ``bt.read``), and
the ML preprocessor exports. Private helpers (``_``-prefixed) are excluded — with
the exception of the accessor-namespace *classes*, which are private by name
(``_StrNamespace``) but public by reach, since ``col("x").str.upper()`` is the
documented spelling.

The root ``__all__`` is not the whole surface, and assuming it was is how 73 public
names went undocumented while every gate stayed green: a user also reaches
``batcher.ml`` (engines, loaders, serving), ``batcher.governance`` (row filters,
column masks, lineage), ``batcher.config`` (the tunable dataclasses), and
``batcher.io`` (the ``Source``/``Sink`` protocols and formats a custom connector
implements). Those four packages curate their own ``__all__``, so they are part of
the surface too — enumerated in ``_SUBPACKAGES`` below. A gate that derives the
surface from one ``__all__`` only proves that one ``__all__`` is documented.
"""

from __future__ import annotations

import inspect
from typing import Any

# Dunder methods that are part of the fluent surface (documented in complete.md's
# `:special-members:` list) rather than Python plumbing.
PUBLIC_DUNDERS = frozenset(
    {
        "__getitem__",
        "__len__",
        "__iter__",
        "__contains__",
    }
)


# The subpackages that curate their own public ``__all__``. A user imports from these
# directly (``from batcher.governance import RowFilter``), so every name they export is
# public and owes the same documentation as a root export.
_SUBPACKAGES = (
    "batcher.ml",
    "batcher.graph",
    "batcher.io",
    "batcher.config",
    "batcher.governance",
)


def _subpackage_exports() -> list[tuple[str, Any]]:
    """Every name the public subpackages export, as ``(qualified_name, obj)`` pairs."""
    import importlib

    out: list[tuple[str, Any]] = []
    for mod_name in _SUBPACKAGES:
        module = importlib.import_module(mod_name)
        for name in getattr(module, "__all__", ()):
            out.append((f"{mod_name}.{name}", getattr(module, name)))
    return out


def _accessor_namespaces() -> list[type]:
    """The typed accessor classes reached as attributes of Expr/Dataset."""
    from batcher.api.dataset.dq import DatasetDQ, ValidationReport
    from batcher.api.dataset.meta import (
        ApproxMeta,
        ColumnChecks,
        ColumnMeta,
        DatasetMeta,
        NullsMeta,
        PairMeta,
        SchemaMeta,
        StorageMeta,
    )
    from batcher.api.dataset.ml import DatasetML
    from batcher.api.dataset.scd import DatasetSCD
    from batcher.api.io_namespace.reader import Reader
    from batcher.api.io_namespace.writer import Writer
    from batcher.plan.expr_ir.audio import _AudioNamespace
    from batcher.plan.expr_ir.image import _ImageNamespace
    from batcher.plan.expr_ir.namespaces.collections import (
        _JsonNamespace,
        _ListNamespace,
        _MapNamespace,
        _StructNamespace,
    )
    from batcher.plan.expr_ir.namespaces.strings import _StrNamespace
    from batcher.plan.expr_ir.namespaces.temporal import _DtNamespace
    from batcher.plan.expr_ir.selectors.core import _SelectorNameNamespace
    from batcher.plan.expr_ir.video import _VideoNamespace

    return [
        _StrNamespace,
        _DtNamespace,
        _ListNamespace,
        _StructNamespace,
        _JsonNamespace,
        _MapNamespace,
        _ImageNamespace,
        _AudioNamespace,
        _VideoNamespace,
        _SelectorNameNamespace,
        Reader,
        Writer,
        DatasetML,
        DatasetDQ,
        ValidationReport,
        DatasetSCD,
        # The `ds.meta` accessor tree — metadata shortcuts, reached as `ds.meta.col("x")`,
        # `ds.meta.col("x").check`, `ds.meta.schema`, `.nulls`, `.approx`, `.storage`,
        # and `ds.meta.against(other)`.
        DatasetMeta,
        SchemaMeta,
        NullsMeta,
        ColumnMeta,
        ColumnChecks,
        ApproxMeta,
        StorageMeta,
        PairMeta,
    ]


def public_classes() -> list[tuple[str, type]]:
    """Every public class, as ``(qualified_name, cls)`` pairs."""
    import batcher as bt

    out: list[tuple[str, type]] = []
    for name in bt.__all__:
        obj = getattr(bt, name, None)
        if inspect.isclass(obj):
            out.append((f"batcher.{name}", obj))

    # `Expr` is reached as the return of `bt.col(...)`, not exported by name.
    from batcher.plan.expr_ir.core import Expr

    out.append(("batcher.plan.expr_ir.core.Expr", Expr))

    for cls in _accessor_namespaces():
        out.append((f"{cls.__module__}.{cls.__name__}", cls))

    from batcher.ml import preprocessors

    for name in preprocessors.__all__:
        obj = getattr(preprocessors, name)
        if inspect.isclass(obj):
            out.append((f"batcher.ml.preprocessors.{name}", obj))

    for qual, obj in _subpackage_exports():
        if inspect.isclass(obj):
            out.append((qual, obj))
    return out


def public_functions() -> list[tuple[str, Any]]:
    """Every public module-level function, as ``(qualified_name, fn)`` pairs."""
    import batcher as bt

    out: list[tuple[str, Any]] = []
    for name in bt.__all__:
        obj = getattr(bt, name, None)
        if inspect.isfunction(obj):
            out.append((f"batcher.{name}", obj))

    from batcher.ml import preprocessors

    for name in preprocessors.__all__:
        obj = getattr(preprocessors, name)
        if inspect.isfunction(obj):
            out.append((f"batcher.ml.preprocessors.{name}", obj))

    for qual, obj in _subpackage_exports():
        if inspect.isfunction(obj):
            out.append((qual, obj))
    return out


def _class_members(qual: str, cls: type) -> list[tuple[str, Any]]:
    """The documented callables a class contributes: its methods and properties."""
    out: list[tuple[str, Any]] = []
    for name, member in vars(cls).items():
        if name.startswith("_") and name not in PUBLIC_DUNDERS:
            continue
        if isinstance(member, property):
            if member.fget is not None:
                out.append((f"{qual}.{name}", member.fget))
        elif inspect.isfunction(member):
            out.append((f"{qual}.{name}", member))
        elif isinstance(member, (staticmethod, classmethod)):
            out.append((f"{qual}.{name}", member.__func__))
    return out


def public_callables() -> list[tuple[str, Any]]:
    """Every public callable users can invoke, as ``(qualified_name, obj)`` pairs.

    Classes appear both in their own right (the class docstring is documentation)
    and via each of their public methods and properties. Names are unique.
    """
    seen: set[str] = set()
    out: list[tuple[str, Any]] = []

    def add(qual: str, obj: Any) -> None:
        if qual not in seen:
            seen.add(qual)
            out.append((qual, obj))

    for qual, fn in public_functions():
        add(qual, fn)
    for qual, cls in public_classes():
        add(qual, cls)
        for mqual, member in _class_members(qual, cls):
            add(mqual, member)
    return out


def public_names() -> set[str]:
    """The bare names (no module prefix) that the documentation must mention."""
    import batcher as bt

    names: set[str] = set(bt.__all__)
    for cls in _accessor_namespaces():
        names |= {
            n
            for n, m in vars(cls).items()
            if not n.startswith("_") and (inspect.isfunction(m) or isinstance(m, property))
        }

    from batcher.ml import preprocessors

    names |= set(preprocessors.__all__)
    names |= {qual.rsplit(".", 1)[-1] for qual, _ in _subpackage_exports()}
    return names


# The accessor namespaces reachable through an `Expr` (as `col("x").str`, `.dt`, …).
# The IO/dataset/preprocessor accessors in `_accessor_namespaces()` are not part of the
# *expression* surface, so the expression reference is scoped to just these.
_EXPR_ACCESSORS = (
    ("str", "batcher.plan.expr_ir.namespaces.strings", "_StrNamespace"),
    ("dt", "batcher.plan.expr_ir.namespaces.temporal", "_DtNamespace"),
    ("list", "batcher.plan.expr_ir.namespaces.collections", "_ListNamespace"),
    ("struct", "batcher.plan.expr_ir.namespaces.collections", "_StructNamespace"),
    ("json", "batcher.plan.expr_ir.namespaces.collections", "_JsonNamespace"),
    ("map", "batcher.plan.expr_ir.namespaces.collections", "_MapNamespace"),
    ("image", "batcher.plan.expr_ir.image", "_ImageNamespace"),
    ("audio", "batcher.plan.expr_ir.audio", "_AudioNamespace"),
    ("video", "batcher.plan.expr_ir.video", "_VideoNamespace"),
)


def _public_method_names(cls: type) -> set[str]:
    return {
        n
        for n, m in vars(cls).items()
        if not n.startswith("_")
        and n != "to_ir"
        and (inspect.isfunction(m) or isinstance(m, (property, staticmethod, classmethod)))
    }


def expression_names() -> set[str]:
    """Every method a user can call on an `Expr` — the fluent builder plus accessors.

    This is what the expression *reference* page must enumerate exhaustively: the
    public methods of `Expr` itself and of each typed accessor namespace it hands out.
    The accessor-property names (``str``, ``dt``, …) are excluded — they are entry
    points documented as the namespaces they open, not methods in their own right.
    """
    import importlib

    from batcher.plan.expr_ir.core import Expr

    accessor_attrs = {name for name, _, _ in _EXPR_ACCESSORS}
    names = _public_method_names(Expr) - accessor_attrs
    for _, module, cls_name in _EXPR_ACCESSORS:
        cls = getattr(importlib.import_module(module), cls_name)
        names |= _public_method_names(cls)
    return names

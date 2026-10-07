"""What one UDF needs on a remote worker, read from the pickle that would ship it.

A remote worker receives a stage's `fn` by serializing it on the driver, so the honest answer
to "what does the worker need" is in that serialization. It is taken with the same
cloudpickle Ray uses (Ray's vendored copy when Ray is installed), through a pickler that
records, as it goes, every module the payload refers to:

* **by reference** — a module-level function or class is pickled as ``module.name``, so the
  worker must be able to import that module. A function from the user's own ``models.py``
  is the case that fails in production: it pickles in a dozen bytes and then the worker
  raises ``ModuleNotFoundError: models`` after the cluster has started and the model loaded.
* **at call time** — an ``import torch`` inside a function body or a class ``__init__`` is
  not part of the payload at all, but it runs on the worker. Every code object the payload
  carries is scanned for its imports, which is how a model engine's ``import vllm`` (made in
  the engine factory, on purpose, so the driver never needs vLLM) is still reported.

The same dump gives the closure size and the picklability verdict, so one serialization
answers all three questions. Nothing here runs the `fn`.
"""

from __future__ import annotations

import dis
import importlib.util
import io
import sys
import types
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ModuleNeed", "StageNeeds", "scan_callable"]

#: Modules the worker always has: the engine itself and the serializers that ship the `fn`.
_PROVIDED = frozenset({"batcher", "cloudpickle", "ray", "__main__", "builtins", "pyarrow"})


@dataclass(frozen=True)
class ModuleNeed:
    """One top-level module a worker must be able to import.

    Attributes:
        module: The top-level import name, such as ``"torch"``.
        kind: ``"package"`` (an installed distribution), ``"local"`` (a file importable
            here that no distribution installs, so a worker has it only if it is shipped),
            or ``"missing"`` (not importable even here).
        distribution: The distribution that provides it, for a package.
        version: That distribution's installed version, for a package.
        path: Where the module was found, for a local module.
        how: ``"reference"`` when the payload refers to it, ``"import"`` when only a
            function body imports it.
    """

    module: str
    kind: str
    distribution: str | None = None
    version: str | None = None
    path: str | None = None
    how: str = "reference"


@dataclass
class StageNeeds:
    """Everything one UDF stage asks of a remote worker.

    Attributes:
        label: A readable name for the stage's `fn`.
        picklable: Whether the `fn` serializes at all.
        error: Why it does not, naming captured variables where they can be found.
        closure_bytes: The serialized size, when it serializes.
        modules: The non-standard-library modules it needs, sorted by name.
    """

    label: str
    picklable: bool
    error: str | None = None
    closure_bytes: int | None = None
    modules: list[ModuleNeed] = field(default_factory=list)


def _pickler_base() -> tuple[Any, str]:
    """The pickler class Ray would serialize with, and what it is called in a message."""
    try:
        from ray import cloudpickle

        return cloudpickle.Pickler, "ray.cloudpickle"
    except ImportError:
        pass
    try:
        import cloudpickle

        return cloudpickle.Pickler, "cloudpickle"
    except ImportError:
        import pickle

        return pickle.Pickler, "pickle"


def _recording_pickler(base: Any) -> Any:
    """A subclass of `base` that records referenced modules and code-object imports."""

    class _Recording(base):  # type: ignore[misc, valid-type]
        def __init__(self, file: io.BytesIO) -> None:
            super().__init__(file)
            self.referenced: set[str] = set()
            self.imported: set[str] = set()

        def reducer_override(self, obj: Any) -> Any:
            if isinstance(obj, types.ModuleType):
                self.referenced.add(obj.__name__)
            elif isinstance(obj, types.CodeType):
                self.imported.update(
                    str(ins.argval)
                    for ins in dis.get_instructions(obj)
                    if ins.opname == "IMPORT_NAME" and ins.argval
                )
            elif isinstance(obj, types.FunctionType | type | types.BuiltinFunctionType):
                module = getattr(obj, "__module__", None)
                if isinstance(module, str):
                    self.referenced.add(module)
            parent = getattr(super(), "reducer_override", None)
            return NotImplemented if parent is None else parent(obj)

    return _Recording


def _top(name: str) -> str:
    return name.split(".", 1)[0]


def _classify(top: str, how: str, distributions: dict[str, list[str]]) -> ModuleNeed | None:
    """What kind of need `top` is, or `None` for one every worker already has."""
    if top in _PROVIDED or top in sys.stdlib_module_names:
        return None
    try:
        spec = importlib.util.find_spec(top)
    except (ImportError, ValueError):
        spec = None
    if spec is None:
        return ModuleNeed(top, "missing", how=how)
    if spec.origin in (None, "built-in", "frozen") and not spec.submodule_search_locations:
        return None
    dists = distributions.get(top)
    if dists:
        from importlib.metadata import PackageNotFoundError, version

        try:
            found = version(dists[0])
        except PackageNotFoundError:
            found = None
        return ModuleNeed(top, "package", distribution=dists[0], version=found, how=how)
    path = spec.origin or next(iter(spec.submodule_search_locations or ()), None)
    return ModuleNeed(top, "local", path=path, how=how)


def scan_callable(fn: object, label: str, distributions: dict[str, list[str]]) -> StageNeeds:
    """Serialize `fn` as a worker would receive it, and report what the worker needs.

    Args:
        fn: The stage's callable (a function, a class UDF, or a callable instance).
        label: A readable name for it.
        distributions: ``importlib.metadata.packages_distributions()``, computed once by
            the caller because it walks every installed distribution.

    Returns:
        The stage's needs. A `fn` that does not serialize reports why and nothing else,
        because the modules of a payload that cannot be built are not knowable.
    """
    base, via = _pickler_base()
    buffer = io.BytesIO()
    pickler = _recording_pickler(base)(buffer)
    try:
        pickler.dump(fn)
    except Exception as exc:
        reason = str(exc).splitlines()[0][:200] if str(exc) else type(exc).__name__
        names = _captured(fn) if via == "ray.cloudpickle" else []
        captured = f"; it captures {names}" if names else ""
        return StageNeeds(label, False, error=f"{via} cannot serialize it: {reason}{captured}")
    seen: dict[str, ModuleNeed] = {}
    for how, names in (("reference", pickler.referenced), ("import", pickler.imported)):
        for top in sorted({_top(n) for n in names}):
            if top in seen:
                continue
            need = _classify(top, how, distributions)
            if need is not None:
                seen[top] = need
    return StageNeeds(
        label,
        True,
        closure_bytes=len(buffer.getvalue()),
        modules=sorted(seen.values(), key=lambda m: m.module),
    )


def _captured(fn: object) -> list[str]:
    """The captured variables Ray's inspector blames, reusing the submit-time diagnostic."""
    from batcher.api.dataset._udf.cluster import _captured_names

    return _captured_names(fn)

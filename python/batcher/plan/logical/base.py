"""`LogicalPlan` — the base class for declarative plan nodes.

Immutable node tree. Each fluent `Dataset` operation returns a new `LogicalPlan`
wrapping the previous one. Validation (column references resolve against the
input's available columns) happens at build time so mistakes fail fast, before
the optimizer or engine ever runs. Logical plans lower to the relational IR JSON
via `to_ir()`; types of derived columns are resolved by the engine.
"""

from __future__ import annotations

import functools
import hashlib
import json
from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir import Expr
from batcher.plan.expr_ir import referenced_columns as _referenced_columns

if TYPE_CHECKING:
    from batcher.plan.schema import SchemaRef

__all__ = ["LogicalPlan"]

# Sentinel distinguishing "not yet cached" from a cached `None` (an `available_schema`
# that legitimately returns "unknown").
_UNSET: Any = object()


def _memoize_noarg(fn, slot: str):
    """Wrap a no-argument pure method so its result is cached per node instance.

    Plan nodes are immutable, so `to_ir`/`available_schema` are pure functions of the
    node — but the optimizer calls them repeatedly (canonical keys, fixpoint change
    detection, type inference) and each recurses over the whole subtree. Caching in the
    instance `__dict__` (present because the `LogicalPlan` base sets no `__slots__`, even
    though frozen subclasses do) collapses that to one computation per node. Writing to
    `__dict__` bypasses the frozen `__setattr__`; the cached value is treated as read-only
    (verified: nothing mutates a `to_ir()` dict or a `SchemaRef` in place).
    """

    @functools.wraps(fn)
    def wrapper(self):
        cache = self.__dict__
        val = cache.get(slot, _UNSET)
        if val is not _UNSET:
            return val
        val = fn(self)
        cache[slot] = val
        return val

    wrapper._memoized = True  # type: ignore[attr-defined]
    return wrapper


def _validate_refs(expr: Expr, available: set[str], *, what: str) -> None:
    """Raise `PlanError` if `expr` references a column not in `available`.

    The single source of the "unknown column(s)" validation message; `what`
    labels the site (e.g. ``"filter"``, ``f"projection {alias!r}"``).
    """
    missing = _referenced_columns(expr) - available
    if missing:
        raise PlanError(
            f"{what} references unknown column(s) {sorted(missing)}; available: {sorted(available)}"
        )


class LogicalPlan:
    """Base class for logical plan nodes."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # Memoize the pure, no-arg structural methods each concrete node defines. Runs
        # once when the subclass is created (and again, harmlessly, when
        # `@dataclass(slots=True)` re-creates it — the `_memoized` guard makes that a
        # no-op). Only wraps a method the subclass actually overrides.
        super().__init_subclass__(**kwargs)
        for name in ("to_ir", "available_schema"):
            fn = cls.__dict__.get(name)
            if fn is not None and not getattr(fn, "_memoized", False):
                setattr(cls, name, _memoize_noarg(fn, f"_c_{name}"))

    def to_ir(self) -> dict[str, Any]:  # pragma: no cover - overridden
        raise NotImplementedError

    def content_key(self) -> str:
        """A stable content fingerprint of this plan's lowered IR (memoized per node).

        Two plans with byte-identical `to_ir()` share a key; any node change changes it.
        `kyber.plan_cache` keys its optimizer memo on this so a re-issued identical query
        reuses its plan. Building the hash (serialize the IR + `blake2b`) is essentially
        the whole cost of a plan-cache lookup, so it is cached in the instance `__dict__`
        the way `to_ir` is: a plan keyed repeatedly — a `collect` loop, an adaptive
        re-optimization of the same subtree — then pays it once, not once per lookup.

        `sort_keys` is unnecessary — `to_ir()` builds its dicts in a fixed, deterministic
        order, so the serialization already canonicalizes an identical plan — which is why
        the compact, unsorted dump is a safe key (never a wrong hit; at worst a missed one).

        A plan carrying an **opaque** node (`map_batches` runs in Python and deliberately has
        no engine IR, so its `to_ir()` raises) cannot be fingerprinted by content. Such a plan
        is keyed by *instance identity* instead: the memo still hits when the same plan object
        is optimized twice (the adaptive loop, a `collect` loop), and two different UDFs can
        never collide onto one key — a collision would hand one query the other's optimized
        plan, i.e. a wrong answer. A rebuilt-from-scratch UDF plan simply misses and
        re-optimizes, which is only ever a cost, never an error.
        """
        cache = self.__dict__
        val = cache.get("_c_content_key", _UNSET)
        if val is _UNSET:
            try:
                payload = json.dumps(self.to_ir(), separators=(",", ":"), default=str)
            except NotImplementedError:
                payload = f"opaque:{id(self):x}"
            val = hashlib.blake2b(payload.encode(), digest_size=16).hexdigest()
            cache["_c_content_key"] = val
        return val

    def available_columns(self) -> list[str]:  # pragma: no cover - overridden
        raise NotImplementedError

    def available_schema(self) -> SchemaRef | None:
        """The output schema (names **and** Arrow types), or ``None`` if not inferable.

        A type-carrying companion to `available_columns()`: nodes that can compute
        their output types from the scan-leaf schema plus per-expression inference
        override this; anything uncertain returns ``None`` so callers fall back to a
        zero-row execution. Pure analysis — it never touches the IR or runs the
        engine. The default is ``None`` (unknown).
        """
        return None

    def _check(self, expr: Expr) -> None:
        """Raise `PlanError` if `expr` references a column not produced by input."""
        _validate_refs(expr, set(self.available_columns()), what="expression")

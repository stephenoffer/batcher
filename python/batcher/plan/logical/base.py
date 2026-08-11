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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from batcher._internal.errors import ColumnNotFoundError, PlanError
from batcher.plan.expr_ir import Col, Expr
from batcher.plan.expr_ir import referenced_columns as _referenced_columns

if TYPE_CHECKING:
    from batcher.plan.schema import SchemaRef

__all__ = ["LogicalPlan", "SortKeySpec"]

# Sentinel distinguishing "not yet cached" from a cached `None` (an `available_schema`
# that legitimately returns "unknown").
_UNSET: Any = object()

# Instance-`__dict__` slot for `available_column_set`, named like the `_memoize_noarg`
# slots (`_c_to_ir`, `_c_available_columns`) it sits beside.
_COLUMN_SET_SLOT = "_c_available_column_set"


def _memoize_noarg(fn, slot: str):
    """Wrap a no-argument pure method so its result is cached per node instance.

    Plan nodes are immutable, so `to_ir`/`available_schema`/`available_columns` are pure
    functions of the node — but the optimizer calls them repeatedly (canonical keys,
    fixpoint change detection, type inference, reference validation) and each recurses over
    the whole subtree. Caching in the instance `__dict__` (present because the `LogicalPlan`
    base sets no `__slots__`, even though frozen subclasses do) collapses that to one
    computation per node. Writing to `__dict__` bypasses the frozen `__setattr__`; the
    cached value is treated as read-only (verified across all call sites: nothing mutates a
    `to_ir()` dict, a `SchemaRef`, or an `available_columns()` list in place).
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


def _reject_duplicate_aliases(aliases: list[str], *, what: str) -> None:
    """Raise `PlanError` if `aliases` names a column more than once.

    A relation's output column names must be unique: a duplicate silently collapses
    when the result is materialized to a name-keyed structure (``to_pydict``), losing
    one of the columns. Rejecting at build time (as Polars does) turns that silent data
    loss into an actionable error. `what` labels the site (e.g. ``"select"``).
    """
    # The common answer — no duplicates — is one C-level `set` build and a length
    # comparison. This runs on every projection, aggregate, and join a plan builds, so
    # the interpreted loop that used to decide it cost a Python iteration per output
    # column of every relation, to conclude "nothing wrong" every time.
    if len(set(aliases)) == len(aliases):
        return
    # A duplicate does exist: name them, in first-seen order, for the message. Membership
    # is tested against a set rather than the growing `dups` list, so reporting stays
    # linear even when a projection repeats one name across thousands of columns.
    seen: set[str] = set()
    dups: list[str] = []
    dup_set: set[str] = set()
    for name in aliases:
        if name in seen:
            if name not in dup_set:
                dup_set.add(name)
                dups.append(name)
        else:
            seen.add(name)
    if dups:
        raise PlanError(
            f"{what} would produce duplicate output column(s) {dups}; each output column "
            "needs a distinct name — rename one with .alias('...') or a keyword"
        )


def available_column_set(plan: LogicalPlan) -> set[str]:
    """`plan.available_columns()` as a set, memoized on the node.

    Every node validates its expressions against the *set* of its input's output columns,
    so building an N-node plan built N sets — each one a fresh O(width) pass over a list
    that is itself memoized. Caching the set beside the list makes a deep plan over a wide
    relation linear in its size rather than in size times width. Read-only by contract,
    like the other memos on `LogicalPlan` (the callers only test membership and subset).

    Args:
        plan: The node whose output column names are wanted.

    Returns:
        The node's output column names.
    """
    cache = plan.__dict__
    columns = cache.get(_COLUMN_SET_SLOT)
    if columns is None:
        columns = set(plan.available_columns())
        cache[_COLUMN_SET_SLOT] = columns
    return columns


def _validate_refs(expr: Expr, available: set[str], *, what: str) -> None:
    """Raise `ColumnNotFoundError` if `expr` references a column not in `available`.

    The single source of the "unknown column(s)" validation message; `what`
    labels the site (e.g. ``"filter"``, ``f"projection {alias!r}"``).

    The error is the narrow `ColumnNotFoundError` rather than a bare `PlanError`, so
    `except ColumnNotFoundError` and `except KeyError` both catch the most common plan
    mistake there is. It subclasses `PlanError`, so handlers catching that still work, and
    the message is unchanged — only the type is more specific.
    """
    missing = _referenced_columns(expr) - available
    if missing:
        raise ColumnNotFoundError(
            f"{what} references unknown column(s) {sorted(missing)}; "
            f"available: {sorted(available)}",
            column=sorted(missing)[0],
            available=sorted(available),
        )


def _validate_projection_refs(expr: Expr, available: set[str], alias: str) -> None:
    """`_validate_refs` for one `Project` item, labelling the site only when it fails.

    `with_columns` on a wide relation emits one projection per column, nearly all of them
    bare pass-through ``Col``s, so this is the per-column inner loop of the whole
    projection path. Two things it used to pay unconditionally now happen only on the
    error path: formatting a ``f"projection {alias!r}"`` label for a failure that is not
    going to occur, and — for a bare column — building a one-element reference set and a
    set difference to answer a question a single membership test answers.
    """
    if type(expr) is Col:
        if expr.name in available:
            return
    elif _referenced_columns(expr) <= available:
        return
    _validate_refs(expr, available, what=f"projection {alias!r}")


class LogicalPlan:
    """Base class for logical plan nodes."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # Memoize the pure, no-arg structural methods each concrete node defines. Runs
        # once when the subclass is created (and again, harmlessly, when
        # `@dataclass(slots=True)` re-creates it — the `_memoized` guard makes that a
        # no-op). Only wraps a method the subclass actually overrides.
        super().__init_subclass__(**kwargs)
        # `available_columns` joins the memoized set for the same reason: it is pure and
        # no-arg, and it is the single most-called structural method in the control plane.
        # Every node's `__post_init__` asks its *input* for it to validate column
        # references, so building an N-node plan asked N times and each answer rebuilt a
        # list of the node's output names — O(width) per call on a wide relation, which
        # is exactly where `with_columns` already does the most work.
        for name in ("to_ir", "available_schema", "available_columns"):
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

        The IR alone is **not** the whole identity, because a `Scan`'s IR is only its
        `source_id`: the engine reads column types off the Arrow batches it is handed, so
        the schema is deliberately not on the wire. That made two runs of one query text
        over sources with the same column *names* and different column *types* collide,
        and handed the second run the plan optimized for the first one's types — silently,
        because every schema-dependent rewrite (key-type validation, cast folding, a range
        join's row encoding) had already made its decision. Each node therefore contributes
        an `identity_suffix()` alongside its IR; `Scan` returns its schema there.
        """
        cache = self.__dict__
        val = cache.get("_c_content_key", _UNSET)
        if val is _UNSET:
            try:
                payload = json.dumps(self.to_ir(), separators=(",", ":"), default=str)
            except NotImplementedError:
                payload = f"opaque:{id(self):x}"
            payload += "|" + self._identity_suffixes()
            val = hashlib.blake2b(payload.encode(), digest_size=16).hexdigest()
            cache["_c_content_key"] = val
        return val

    def identity_suffix(self) -> str:
        """Any part of this node's identity its `to_ir()` does not carry.

        Empty for every node whose IR is its whole identity, which is all of them but
        `Scan`. Overriding this is how a node keeps `content_key` honest without widening
        the wire contract.
        """
        return ""

    def _identity_suffixes(self) -> str:
        """This subtree's identity suffixes, in a fixed pre-order."""
        from batcher.plan.visitor import walk

        return ";".join(f"{i}:{n.identity_suffix()}" for i, n in enumerate(walk(self)))

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
        _validate_refs(expr, available_column_set(self), what="expression")


@dataclass(frozen=True, slots=True)
class SortKeySpec:
    """One ordering term: an expression and how it orders.

    Lives here, in the neutral base, because three nodes in two sibling modules order rows
    by it — `Sort` and `Distinct` in one, `Window` in the other — and both modules already
    depend on this one. Its previous home next to `Sort` made `relational` import
    `aggregate` while `aggregate` imports `relational`, which is a cycle.

    `nulls_first` defaults to `False` to match SQL's `ORDER BY`, not arrow's `SortOptions`,
    whose default is the opposite. The engine reads this flag rather than defaulting.
    """

    expr: Expr
    descending: bool = False
    nulls_first: bool = False

"""Manifest-driven file skipping — turn a pushed predicate into a surviving-file set.

A lakehouse transaction log already records, per data file, the exact record count,
the partition values, and per-column min/max/null-count. That is a zone map over the
*file* dimension, and it is the difference between opening one Parquet footer and
opening a hundred thousand of them. This module is the layer that consumes it: given
the predicate Kyber pushed to a source and a manifest of per-file statistics, it
reports which files can still contain a matching row — so `Source.splits()` emits
work only for those, and the rest are never opened, listed, or shipped to a worker.

This is the file-granularity sibling of `pruning` (which zone-maps *row groups*
inside a file the reader already opened). Pruning here happens at **plan time**, on
the control plane, before any data file is touched.

The manifest is an Arrow table in the **add-action layout** — the shape delta-rs
returns from ``get_add_actions(flatten=True)``, and the shape the Iceberg and Delta
Sharing connectors normalize into (`ADD_ACTION_LAYOUT`):

    path | num_records | partition.<col> | min.<col> | max.<col> | null_count.<col>

Everything is evaluated **vectorized over the file dimension** with pyarrow compute —
one pass over the manifest, no Python loop over files, so a table with a million data
files prunes in the control plane without an ``O(files)`` interpreter cost.

## Soundness

Pruning is a three-valued decision and the only unsafe answer is a false *prune*. So
the rule throughout is: **a file is dropped only when the manifest proves it cannot
contain a matching row; anything unknown keeps it.** A missing statistic, an
unrecorded column, an unsupported predicate node, or a type that will not compare all
resolve to *keep*. Every mask is therefore `fill_null(True)` — a null bound means "no
recorded stat", never "no match". The engine re-checks the predicate with its own
`Filter` regardless, so an over-broad survivor set is always correct and only costs
I/O; an over-narrow one would silently lose rows.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "MAX_PREFIX",
    "MIN_PREFIX",
    "NULL_PREFIX",
    "PARTITION_PREFIX",
    "file_prune_mask",
    "surviving_files",
]

# The add-action manifest layout. `lakehouse_manifest` aggregates the same columns for
# whole-source statistics; both read this one definition so the contract cannot drift.
PARTITION_PREFIX = "partition."
MIN_PREFIX = "min."
MAX_PREFIX = "max."
NULL_PREFIX = "null_count."

_CMP = frozenset({"eq", "ne", "lt", "le", "gt", "ge"})
# `lit OP col` is normalized to `col OP' lit` by flipping the operator.
_FLIP = {"lt": "gt", "le": "ge", "gt": "lt", "ge": "le", "eq": "eq", "ne": "ne"}


def surviving_files(predicate: dict[str, Any] | None, manifest: Any) -> list[str] | None:
    """Paths of the files in `manifest` that can still contain a row matching `predicate`.

    The entry point a connector's `splits()` calls. Returns `None` when the manifest
    proves nothing — no predicate, an unusable manifest, or a predicate no column
    statistic can decide — which the caller reads as "keep every file" and plans an
    unpruned scan. Never raises: a manifest it cannot interpret yields `None`, so
    pruning degrades to the status quo rather than failing a query.

    Args:
        predicate: The pushed predicate IR, or None.
        manifest: Per-file statistics in the add-action layout.

    Returns:
        The surviving file paths, or None if no file-level pruning could be decided.
    """
    mask = file_prune_mask(predicate, manifest)
    if mask is None:
        return None
    try:
        return manifest.column("path").filter(mask).to_pylist()
    except Exception:
        return None


def file_prune_mask(predicate: dict[str, Any] | None, manifest: Any) -> Any | None:
    """Boolean mask over `manifest`'s rows: True where the file may contain a match.

    Vectorized over the file dimension. `None` means "undecidable — keep everything",
    which is distinct from an all-True mask only in that it lets the caller skip the
    filtering work entirely.

    Args:
        predicate: The pushed predicate IR, or None.
        manifest: Per-file statistics in the add-action layout.

    Returns:
        A pyarrow BooleanArray, or None if the predicate could not be evaluated.
    """
    if predicate is None or manifest is None:
        return None
    try:
        import pyarrow.compute as pc

        if manifest.num_rows == 0 or "path" not in manifest.column_names:
            return None
        mask = _eval(predicate, manifest, pc)
        if mask is None:
            return None
        # A null cell is an *absent statistic*, never a proof of non-match.
        return pc.fill_null(mask, True)
    except Exception:
        return None  # an uninterpretable manifest prunes nothing, it does not fail


def _eval(ir: dict[str, Any], manifest: Any, pc: Any) -> Any | None:
    """Recursively evaluate `ir` into a per-file keep-mask, or None if undecidable."""
    kind = ir.get("e")
    if kind == "binary":
        op = ir.get("op")
        if op == "and":
            return _conjunction(ir, manifest, pc)
        if op == "or":
            return _disjunction(ir, manifest, pc)
        if op in _CMP:
            return _comparison(op, ir.get("left"), ir.get("right"), manifest, pc)
        return None
    if kind in ("is_null", "is_not_null"):
        return _nullness(kind, ir.get("input"), manifest, pc)
    return None


def _conjunction(ir: dict[str, Any], manifest: Any, pc: Any) -> Any | None:
    """`a AND b` — a file survives only if it survives both sides.

    An undecidable side contributes no pruning (it keeps everything), so it is simply
    dropped from the conjunction rather than poisoning it: ``decidable AND unknown``
    still prunes on the decidable half. That is the whole point of pushing a compound
    predicate — the partition column prunes even when the payload column cannot.
    """
    left = _eval(ir["left"], manifest, pc)
    right = _eval(ir["right"], manifest, pc)
    if left is None:
        return right
    if right is None:
        return left
    return pc.and_(pc.fill_null(left, True), pc.fill_null(right, True))


def _disjunction(ir: dict[str, Any], manifest: Any, pc: Any) -> Any | None:
    """`a OR b` — a file survives if it survives *either* side.

    Unlike a conjunction, one undecidable side makes the whole disjunction
    undecidable: if `b` might match in any file, no file can be pruned on `a` alone.
    """
    left = _eval(ir["left"], manifest, pc)
    right = _eval(ir["right"], manifest, pc)
    if left is None or right is None:
        return None
    return pc.or_(pc.fill_null(left, True), pc.fill_null(right, True))


def _column_of(ir: dict[str, Any] | None) -> str | None:
    return ir["name"] if isinstance(ir, dict) and ir.get("e") == "col" else None


def _literal_of(ir: dict[str, Any] | None) -> tuple[Any, bool]:
    """``(value, is_literal)`` for a literal IR node, unwrapping its typed kind.

    Temporal kinds stay as their raw epoch offsets (date=days, timestamp/time=micros),
    matching how a manifest records them; a mismatch merely fails the comparison cast
    below and keeps the file.
    """
    if not isinstance(ir, dict) or ir.get("e") != "lit":
        return None, False
    ((_kind, value),) = ir["value"].items()
    return value, True


def _comparison(
    op: str, left: dict | None, right: dict | None, manifest: Any, pc: Any
) -> Any | None:
    """Keep-mask for one ``col OP literal`` term, from the column's per-file bounds."""
    column, value = _column_of(left), None
    literal, is_lit = _literal_of(right)
    if column is not None and is_lit:
        value = literal
    else:  # try the flipped form: `literal OP col`
        column = _column_of(right)
        literal, is_lit = _literal_of(left)
        if column is None or not is_lit:
            return None
        value, op = literal, _FLIP[op]

    lo, hi = _bounds(column, manifest)
    if lo is None or hi is None:
        return None  # no recorded bound for this column → cannot prune on it

    try:
        # A file whose values are all NULL matches no comparison (NULL OP x is never
        # true), so it is prunable even though its min/max are absent.
        keep = _compare(op, lo, hi, value, pc)
        if keep is None:
            return None
        all_null = _all_null_mask(column, manifest, pc)
        if all_null is not None:
            keep = pc.and_(pc.fill_null(keep, True), pc.invert(all_null))
        return keep
    except Exception:
        return None  # a type that will not compare prunes nothing


def _compare(op: str, lo: Any, hi: Any, value: Any, pc: Any) -> Any | None:
    """The zone-map decision for `op` against a column's per-file ``[lo, hi]`` bounds.

    Each mask answers "could a value in ``[lo, hi]`` satisfy ``x OP value``?" — the
    negation of the interval being provably disjoint from the predicate's solution set.
    """
    if op == "eq":  # value must lie inside the file's range
        return pc.and_(pc.less_equal(lo, value), pc.greater_equal(hi, value))
    if op == "lt":  # some value below `value` requires the minimum to be below it
        return pc.less(lo, value)
    if op == "le":
        return pc.less_equal(lo, value)
    if op == "gt":  # some value above `value` requires the maximum to be above it
        return pc.greater(hi, value)
    if op == "ge":
        return pc.greater_equal(hi, value)
    if op == "ne":  # only a file that is *constant* at `value` can be ruled out
        return pc.invert(pc.and_(pc.equal(lo, value), pc.equal(hi, value)))
    return None


def _nullness(kind: str, input_ir: dict | None, manifest: Any, pc: Any) -> Any | None:
    """Keep-mask for ``IS NULL`` / ``IS NOT NULL`` from per-file null counts."""
    column = _column_of(input_ir)
    if column is None:
        return None
    name = f"{NULL_PREFIX}{column}"
    names = manifest.column_names
    if name not in names or "num_records" not in names:
        return None
    nulls = manifest.column(name)
    rows = manifest.column("num_records")
    try:
        if kind == "is_null":  # needs at least one null
            return pc.greater(nulls, 0)
        return pc.less(nulls, rows)  # IS NOT NULL needs at least one non-null
    except Exception:
        return None


def _bounds(column: str, manifest: Any) -> tuple[Any, Any]:
    """The per-file ``(min, max)`` arrays for `column`, or ``(None, None)``.

    A **partition** column is constant within each file and recorded untruncated, so
    its literal value is both bounds — which is what lets a partition predicate prune
    a file exactly. A data column uses its recorded ``min.``/``max.`` statistics; a
    truncated string bound only widens the interval, so it stays sound to prune on.
    """
    names = manifest.column_names
    part = f"{PARTITION_PREFIX}{column}"
    if part in names:
        value = manifest.column(part)
        return value, value
    lo, hi = f"{MIN_PREFIX}{column}", f"{MAX_PREFIX}{column}"
    if lo in names and hi in names:
        return manifest.column(lo), manifest.column(hi)
    return None, None


def _all_null_mask(column: str, manifest: Any, pc: Any) -> Any | None:
    """Mask of files where `column` is entirely NULL (so no comparison can match)."""
    names = manifest.column_names
    name = f"{NULL_PREFIX}{column}"
    if name not in names or "num_records" not in names:
        return None
    try:
        return pc.fill_null(pc.equal(manifest.column(name), manifest.column("num_records")), False)
    except Exception:
        return None

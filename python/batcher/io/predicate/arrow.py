"""IR to a `pyarrow.dataset.Expression`, for every file-format and lakehouse reader.

The most-used translator: parquet, ORC, Lance, Delta, Hudi and the multimodal readers all
bind their scan filter through it. It is also the only one handed the scanned relation's
schema, which is what lets a literal be typed to its column instead of guessed at.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from batcher.io.predicate._literals import (
    _col_and_pa_literal,
    _comparable,
    _field_type,
    _literal,
)
from batcher.io.predicate._shapes import _const_bool, _in_list, _str_predicate
from batcher.plan.ir_tags import COMPARISON_FLIP, COMPARISON_OPS

__all__ = ["to_pyarrow_expression"]


def to_pyarrow_expression(ir: dict[str, Any] | None, schema: Any | None = None) -> Any | None:
    """Translate the pushable subset of `ir` to a `pyarrow.dataset.Expression`.

    `schema` is the scanned table's Arrow schema, when the caller has it. It lets a
    temporal literal be typed to its column — the tz-aware timestamp columns common in
    lakehouse tables cannot be compared against a tz-naive literal, so without it a pushed
    filter on such a column raises rather than prunes. Omitting it keeps the prior
    behavior (a tz-naive ``timestamp[us]`` literal).

    **`None` in gives `None` out**, the same answer an unpushable predicate gets, because
    every caller treats the two identically: no filter to bind, so read what you would have
    read. Accepting it here retired five near-identical wrappers — `orc._pa_filter`,
    `parquet.source._pa_filter`, `parquet.dataset._pa_filter`, `hudi._expression` and
    `delta.source`'s guard — that existed for no other reason. The signature was the only
    thing making them necessary, and a wrapper whose whole body is an argument check belongs in
    the callee.

    Args:
        ir: The predicate IR Kyber pushed to this scan, or `None` when it pushed nothing.
        schema: The scanned relation's Arrow schema, when the caller has it.

    Returns:
        A `pyarrow.dataset.Expression`, or `None` when there is nothing pushable to bind.
    """
    if ir is None:
        return None
    return _to_pa(ir, _dataset_module(), schema)


# `pyarrow.dataset` is a heavy submodule and is *not* loaded by importing pyarrow, so a
# module-level import here would add ~50 ms to `import batcher` for a module only the
# predicate-pushdown path needs. Bound on first use instead of re-imported per pushdown.
_DATASET = None


def _dataset_module() -> Any:
    """The `pyarrow.dataset` module, imported at most once."""
    global _DATASET
    if _DATASET is None:
        import pyarrow.dataset

        _DATASET = pyarrow.dataset
    return _DATASET


_COMPUTE = None


def _compute_module() -> Any:
    """The `pyarrow.compute` module, imported at most once.

    The string predicates are spelled as compute kernels bound to a field expression
    (``pc.starts_with(ds.field("c"), "US")``) because `Expression` itself has no
    ``starts_with``. Loaded like `pyarrow.dataset` above and for the same reason.
    """
    global _COMPUTE
    if _COMPUTE is None:
        import pyarrow.compute

        _COMPUTE = pyarrow.compute
    return _COMPUTE


def _pa_in_list(ir: dict[str, Any], ds: Any, schema: Any | None) -> Any | None:
    """A pyarrow ``is_in`` expression for a pushable `IN` list, or None.

    Every member is type-checked against the column with the same `_comparable` gate the
    comparisons use, because `is_in` has the identical failure mode: a ``date32`` column
    against string members raises `ArrowNotImplementedError` from inside whichever task
    built the scanner rather than declining.

    Members are the plain Python values, not `_pa_literal` scalars: pyarrow builds the
    value set with `pa.array`, which types a list of `date`/`datetime` objects correctly
    but rejects a list of `pa.Scalar`. The one shape that cannot be expressed that way is
    a **timezone-aware** column, whose members would have to carry the column's own zone,
    so that declines rather than risk comparing across zones.
    """
    parsed = _in_list(ir)
    if parsed is None:
        return None
    column, members = parsed
    col_type = _field_type(schema, column)
    if any(not _comparable(col_type, {"value": m}) for m in members):
        return None
    if col_type is not None and pa.types.is_timestamp(col_type) and col_type.tz is not None:
        return None
    return ds.field(column).isin([_literal({"value": m}) for m in members])


def _pa_str(ir: dict[str, Any], ds: Any, schema: Any | None) -> Any | None:
    """A pyarrow expression for a pushable string predicate, or None.

    Exact rather than widening, unlike the SQL `LIKE` spelling: these kernels are
    byte-wise and take an explicit ``ignore_case`` that defaults off, so there is no
    collation to change the answer underneath. That is why the caller offers them in
    exact mode too.
    """
    parsed = _str_predicate(ir)
    if parsed is None:
        return None
    column, fn, pattern = parsed
    col_type = _field_type(schema, column)
    if col_type is not None and not (
        pa.types.is_string(col_type) or pa.types.is_large_string(col_type)
    ):
        return None
    pc = _compute_module()
    field = ds.field(column)
    if fn == "starts_with":
        return pc.starts_with(field, pattern)
    if fn == "ends_with":
        return pc.ends_with(field, pattern)
    return pc.match_substring(field, pattern)


def _to_pa(
    ir: dict[str, Any], ds: Any, schema: Any | None = None, exact: bool = False
) -> Any | None:
    e = ir.get("e")
    const = _const_bool(ir)
    if const is not None:
        return ds.scalar(const)
    if e == "is_null":
        inner = ir["input"]
        return ds.field(inner["name"]).is_null() if inner.get("e") == "col" else None
    if e == "is_not_null":
        inner = ir["input"]
        return ds.field(inner["name"]).is_valid() if inner.get("e") == "col" else None
    if e == "not":
        # Exact: negating a widened operand narrows the filter and would lose rows.
        inner_expr = _to_pa(ir["input"], ds, schema, True)
        return None if inner_expr is None else ~inner_expr
    if e == "in_list":
        return _pa_in_list(ir, ds, schema)
    if e == "str":
        return _pa_str(ir, ds, schema)
    if e != "binary":
        return None
    op = ir["op"]
    if op in ("and", "or"):
        left = _to_pa(ir["left"], ds, schema, exact)
        right = _to_pa(ir["right"], ds, schema, exact)
        if op == "or":
            # Dropping a disjunct narrows the filter and would lose rows. All or nothing.
            return None if left is None or right is None else (left | right)
        if exact:
            return None if left is None or right is None else (left & right)
        # An AND may keep whichever side is pushable: a conjunct only ever *widens* the
        # rows read, and the engine's `Filter` re-checks all of them. Dropping the whole
        # filter because one term was not pushable is what turned an unpushable date
        # comparison into a full scan of an eight-column, six-predicate ClickBench query.
        if left is None:
            return right
        return left if right is None else (left & right)
    if op in COMPARISON_OPS:
        parsed = _col_and_pa_literal(ir["left"], ir["right"], schema)
        if parsed is None:
            return None
        col, value, flipped = parsed
        effective = COMPARISON_FLIP[op] if flipped else op
        field = ds.field(col)
        return {
            "eq": field == value,
            "ne": field != value,
            "lt": field < value,
            "le": field <= value,
            "gt": field > value,
            "ge": field >= value,
        }[effective]
    return None

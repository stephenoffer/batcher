"""Output types for the container accessors: `list`, `struct` and `map`.

Every rule here answers from the operand's *already-inferred* Arrow type rather than from
the expression, so nothing in this module recurses back into the dispatcher. That is what
keeps it a leaf of the `infer` package: `dispatch` resolves the operand once and asks these
functions what the container op does to it.

Inferring these (rather than returning ``None``) matters for more than a tidy schema: an
uninferable projection sends `Dataset.schema` down the zero-row execution fallback, and the
engine collapses a zero-row projection's whole schema to `Null` -- so a single uninferred
`list.sum` would make *every* output column, its passthrough neighbours included, report
`null`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

if TYPE_CHECKING:
    from batcher.plan.expr_ir import Expr

__all__ = [
    "as_list_type",
    "list_element_type",
    "list_operand",
    "listfunc_type",
    "mapfunc_type",
    "struct_field_type",
]

# `list` accessor (`ListFunc`) output types. `len`/`n_unique`/`arg_max`/`arg_min`
# count or index -> Int64.
_LIST_INT = frozenset({"len", "n_unique", "arg_max", "arg_min"})
# `sort_desc` is `sort`'s twin and preserves the element type just as it does (verified:
# an Int list sorts to List<Int64>, a Float list to List<Float64>).
_LIST_SAME = frozenset({"reverse", "sort", "sort_desc", "unique"})
# Positions, not values: `arg_sort` returns the permutation that would sort the list, so
# it is List<Int64> whatever the elements are (the plural of `arg_min`/`arg_max` above).
_LIST_INT_LIST = frozenset({"arg_sort"})
# Element-wise transforms the engine computes in floating point whatever the input's
# element width (verified: an Int list's cum_sum/diff/softmax all come back List<Double>).
# `cum_sum` is float here even though the scalar `sum` is element-typed, because the
# engine's running total is accumulated in f64.
_LIST_FLOAT_LIST = frozenset({"cum_sum", "diff", "softmax"})
# Genuinely float, whatever the element width (verified against the engine: an Int
# list's mean/median/product/std/var/l2_norm all come back as `double`). `sum` is NOT
# here: it preserves the element type (Int list -> Int64, like `min`/`max`), and
# classifying it as float made `Dataset.schema` disagree with execution.
_LIST_FLOAT_REDUCE = frozenset(
    {
        "mean",
        "median",
        "product",
        "std",
        "var",
        "l2_norm",
        "entropy",
        # The remaining norms, verified the same way: `l1_norm` and `max_abs` reduce an
        # Int list to `double` exactly as `l2_norm` does.
        "l1_norm",
        "max_abs",
    }
)
# Reductions that preserve the (numeric) element type: `sum` alongside `min`/`max`.
_LIST_ELEMENT_REDUCE = frozenset({"sum", "min", "max"})


def list_operand(expr: object) -> Expr:
    """The list-typed operand of a slice (`input`) or set-op (`left`) node."""
    inp = getattr(expr, "input", None)
    return inp if inp is not None else expr.left  # type: ignore[attr-defined]


def as_list_type(t: pa.DataType | None) -> pa.DataType | None:
    """Return `t` only if it is a List type (an unchanged list output)."""
    return t if t is not None and pa.types.is_list(t) else None


def list_element_type(t: pa.DataType | None) -> pa.DataType | None:
    """The element type of a List type, or ``None`` if `t` is not a list."""
    return t.value_type if t is not None and pa.types.is_list(t) else None


def listfunc_type(fn: str, input_t: pa.DataType | None) -> pa.DataType | None:
    """The Arrow type a `list` accessor function produces over `input_t`."""
    if fn in _LIST_INT:
        return pa.int64()
    if fn in _LIST_SAME:
        return as_list_type(input_t)
    if fn in _LIST_INT_LIST:
        return pa.list_(pa.int64()) if list_element_type(input_t) is not None else None
    if fn in _LIST_FLOAT_LIST:
        return pa.list_(pa.float64()) if list_element_type(input_t) is not None else None
    if fn in _LIST_FLOAT_REDUCE:
        return pa.float64()  # always double, whatever the element width
    if fn in _LIST_ELEMENT_REDUCE:
        # `sum`/`min`/`max` preserve the element type (already widened at the scan leaf):
        # summing/minning an Int list yields Int64, a Float list yields Float64.
        return list_element_type(input_t)
    if fn in ("normalize", "log_softmax"):
        # Rescale each element (unit L2 norm, or the log-domain distribution) -> List<Float64>.
        return pa.list_(pa.float64()) if list_element_type(input_t) is not None else None
    if fn == "flatten":
        # `List<List<T>>` -> `List<T>`: the flattened output IS the (list) element type.
        return as_list_type(list_element_type(input_t))
    return None  # any remaining reduction the engine decides -> fall back


def struct_field_type(struct_t: pa.DataType | None, field: str) -> pa.DataType | None:
    """The type of one named field of a Struct, or ``None`` if absent or not a struct."""
    if struct_t is None or not pa.types.is_struct(struct_t):
        return None
    idx = struct_t.get_field_index(field)
    return struct_t.field(idx).type if idx >= 0 else None


def mapfunc_type(fn: str, map_t: pa.DataType | None) -> pa.DataType | None:
    """The Arrow type a `map` accessor function produces over `map_t`."""
    if map_t is not None and pa.types.is_struct(map_t) and fn == "map_keys":
        # `.struct.keys()` is the same node as `.map.keys()` — a struct is a keyed
        # container and the kernel answers both — but its keys come from the *type*, so
        # they are always text. Without this arm the whole `.struct.keys()` column
        # declared `null` while producing `List<Utf8>`.
        return pa.list_(pa.string())
    if map_t is None or not pa.types.is_map(map_t):
        return None
    if fn == "map_keys":
        return pa.list_(map_t.key_type)
    if fn == "map_values":
        return pa.list_(map_t.item_type)
    if fn == "map_entries":
        # The entries child of an Arrow Map is `Struct<key, value>`, and the field names
        # are part of the type a caller then subscripts (`e.struct.get("key")`), so they
        # are spelled here rather than left to the engine. `key` is non-nullable because
        # a map entry cannot have one; `value` can be null.
        return pa.list_(
            pa.struct(
                [
                    pa.field("key", map_t.key_type, nullable=False),
                    pa.field("value", map_t.item_type),
                ]
            )
        )
    if fn == "element_at":
        return map_t.item_type
    return None

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
    "widened_element_out",
]

# `list` accessor (`ListFunc`) output types. `len`/`n_unique`/`arg_max`/`arg_min`
# count or index -> Int64.
_LIST_INT = frozenset({"len", "n_unique", "n_unique_with_nulls", "arg_max", "arg_min"})
# `sort_desc` is `sort`'s twin and preserves the element type just as it does (verified:
# an Int list sorts to List<Int64>, a Float list to List<Float64>).
# The null-placing and null-keeping variants are the same kernels with one flag moved.
_LIST_SAME = frozenset(
    {
        "reverse",
        "sort",
        "sort_desc",
        "sort_nulls_first",
        "sort_desc_nulls_first",
        "unique",
        "unique_with_nulls",
    }
)
# Positions, not values: `arg_sort` returns the permutation that would sort the list, so
# it is List<Int64> whatever the elements are (the plural of `arg_min`/`arg_max` above).
_LIST_INT_LIST = frozenset({"arg_sort"})
# Element-wise transforms the engine computes in floating point whatever the input's
# element width (verified: an Int list's cum_sum/softmax come back List<Double>).
# `cum_sum` is float here even though the scalar `sum` is element-typed, because the
# engine's running total is accumulated in f64. `diff` is not here: an integer list
# differences exactly into List<Int64> (see `listfunc_type`).
_LIST_FLOAT_LIST = frozenset({"cum_sum", "softmax"})
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
# Reductions that preserve the element type, whatever it is. An ordering comparison is
# defined for every element type the engine carries, so `min`/`max` over a String list is a
# String and over a Date list a Date (verified against the engine for int, float, string,
# bool, date32 and timestamp elements).
_LIST_ORDER_REDUCE = frozenset({"min", "max"})

# `sum` preserves the element type only while that type is *numeric*. It reads as `min`'s
# and `max`'s sibling and is not one: there is no such thing as adding two strings here, so
# the engine coerces the elements and returns Double. Classifying it with them declared
# `string` for a `List<String>` sum that the engine returns as `double` -- worse than an
# uncertain answer, because a confident wrong one is what a caller plans against. It also
# made the *same query* return two different types: `Project.available_schema` types an
# empty result, so a filter that matched nothing produced `v: string` where a filter that
# matched produced `v: double`. Measured: String, Boolean, Timestamp and Null elements all
# sum to Double; Date raises; Int64 and Float64 are preserved.
_LIST_NUMERIC_SUM = frozenset({"sum"})


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
    if fn == "diff":
        # `bc_expr`'s `list_reduce::diff` keeps an integer element (other than UInt64) in
        # Int64 and computes every other numeric element in Float64.
        element = list_element_type(input_t)
        if element is None:
            return None
        exact = pa.types.is_integer(element) and not pa.types.is_uint64(element)
        return pa.list_(pa.int64() if exact else pa.float64())
    if fn in _LIST_FLOAT_REDUCE:
        return pa.float64()  # always double, whatever the element width
    if fn in _LIST_ORDER_REDUCE:
        # `min`/`max` preserve the element type, except that a narrow *integer* element widens
        # on the way out — see `widened_element_out`.
        return widened_element_out(list_element_type(input_t))
    if fn in _LIST_NUMERIC_SUM:
        element = list_element_type(input_t)
        if element is None:
            return None
        numeric = pa.types.is_integer(element) or pa.types.is_floating(element)
        # `sum` accumulates a narrow integer child in `i64` and returns Int64 (the engine's
        # exact-integer arm), so the same widening applies here for the same reason.
        return widened_element_out(element) if numeric else pa.float64()
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


def mapfunc_type(fn: str, map_t: pa.DataType | None, key: object = None) -> pa.DataType | None:
    """The Arrow type a `map` accessor function produces over `map_t`.

    `key` is the literal lookup an `element_at` carries. It is only consulted for a
    **struct** input, where the answer is a named field's type rather than the container's
    uniform value type.
    """
    if map_t is not None and pa.types.is_struct(map_t):
        # A struct is a keyed container and the same kernel answers both namespaces, so
        # `.struct.keys()`/`.struct.get()` arrive here as `.map` nodes.
        if fn == "map_keys":
            # A struct's keys come from the *type*, so they are always text. Without this
            # the whole `.struct.keys()` column declared `null` while producing
            # `List<Utf8>`.
            return pa.list_(pa.string())
        if fn == "element_at" and isinstance(key, str):
            # `.struct.get(name)` is documented as the subscript spelling of
            # `.struct.field(name)` -- it is what ``s["x"]`` lowers to -- and the two built
            # different nodes, of which only `StructField` was typed. So the *same* field
            # projection declared `string` written one way and nothing at all written the
            # other, which cost every column in the projection its type. Answered by the
            # helper `StructField` already uses, so the two spellings cannot drift again.
            return struct_field_type(map_t, key)
        return None
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


def widened_element_out(element: pa.DataType | None) -> pa.DataType | None:
    """A list element's type once an op has made it a *top-level column*.

    The FFI boundary leaves a narrow numeric leaf inside a list at its own width, because a
    tensor column's child is a component of one value rather than a column (see
    ``plan.types.lattice._widen_element``). The ops that hand an element back as a column widen
    a narrow **integer** there, so `Dataset.schema` has to predict the same thing or it lies
    about a query the engine answers with `int64`. Floats keep their width on both sides, and
    ``uint64`` was never widened at the boundary either.
    """
    if element is None:
        return None
    from batcher.plan.types.lattice import widen

    return widen(element) if pa.types.is_integer(element) else element

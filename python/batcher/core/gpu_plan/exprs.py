"""Scalar `Expr` IR → dataframe column, for the GPU (cuDF) and verification (pandas) backends.

This is the translator's vocabulary: every expression the GPU path can evaluate is a case
here, and everything else raises `Unsupported` so the caller drops the whole stage to the
native CPU engine. Coverage is what decides how much of a real query reaches the device — a
plan is GPU-eligible only if *every* expression in it is, so one missing case sends an
otherwise perfect chain back to the host.

Two rules govern what may be added. A case must be **result-identical to the CPU engine**
including its null behavior, which is not the same as its NaN behavior: Batcher follows Arrow,
where `null` and `NaN` are different values, and both backends here are loaded so they agree
(see `backend.DfBackend`). And a case must be *exact* — never an approximation that happens to
match on the common input, because a fallback costs time while a wrong answer costs trust.
"""

from __future__ import annotations

import datetime as _dt
from typing import TYPE_CHECKING, Any

from batcher.core.gpu_plan.backend import Unsupported
from batcher.core.gpu_plan.scalar_fns import (
    apply_ufunc,
    eval_date,
    eval_date_trunc,
    eval_math,
    eval_math2,
    eval_strftime,
)
from batcher.core.gpu_plan.vocab.operators import compare, eval_binary
from batcher.core.gpu_plan.vocab.strings import eval_str

if TYPE_CHECKING:
    from batcher.core.gpu_plan.backend import DfBackend

__all__ = ["DECLINED_EXPRS", "eval_expr", "literal_value"]


def literal_value(tagged: dict) -> Any:
    """A tagged IR literal as the Python scalar the dataframe libraries compare against.

    The tag carries the *type*, and dropping it is a silent-wrong-answer bug rather than a
    cosmetic one: a `date` literal rides the wire as days-since-epoch and a `timestamp` as
    microseconds, so handing the raw integer to a comparison against a datetime column either
    raises (pandas) or coerces to something meaningless (cuDF). Non-finite floats ride as
    names because JSON has no `NaN`/`Infinity` token.

    Args:
        tagged: The single-entry ``{"<kind>": <value>}`` dict from a `lit` node's ``value``.

    Returns:
        The Python scalar the literal denotes.

    Raises:
        Unsupported: For a literal kind the translator does not model.
    """
    if len(tagged) != 1:
        raise Unsupported(f"literal shape {sorted(tagged)}")
    kind, raw = next(iter(tagged.items()))
    if kind in ("int", "str", "bool"):
        return raw
    if kind == "float":
        return float(raw) if isinstance(raw, (int, float)) else _NON_FINITE[raw]
    if kind == "date":
        return _dt.date(1970, 1, 1) + _dt.timedelta(days=int(raw))
    if kind == "timestamp":
        return _dt.datetime(1970, 1, 1) + _dt.timedelta(microseconds=int(raw))
    raise Unsupported(f"literal kind {kind!r}")


_NON_FINITE = {
    "NaN": float("nan"),
    "inf": float("inf"),
    "+inf": float("inf"),
    "Infinity": float("inf"),
    "-inf": float("-inf"),
    "-Infinity": float("-inf"),
}


def eval_expr(ir: dict, df, be: DfBackend):
    """Evaluate one `Expr` IR node against dataframe `df`, returning a column or a scalar.

    A scalar comes back only for a bare literal; every caller that needs a column passes the
    result through `DfBackend.column`, so a literal is broadcast in the library's own layer
    rather than one Python object per row.

    Args:
        ir: The expression's JSON IR node.
        df: The dataframe the column names resolve against.
        be: The dataframe backend to compute on.

    Returns:
        A Series of `df`'s length, or a Python scalar for a bare literal.

    Raises:
        Unsupported: For any node outside the translated subset.
    """
    handler = _HANDLERS.get(ir.get("e"))
    if handler is None:
        raise Unsupported(f"expr {ir.get('e')}")
    return handler(ir, df, be)


def _col(ir, df, _be):
    name = ir["name"]
    if name not in df.columns:
        raise Unsupported(f"column {name!r} absent from the GPU frame")
    return df[name]


def _not(ir, df, be):
    return ~be.column(eval_expr(ir["input"], df, be), df)


def _is_null(ir, df, be):
    return be.column(eval_expr(ir["input"], df, be), df).isna()


def _is_not_null(ir, df, be):
    return be.column(eval_expr(ir["input"], df, be), df).notna()


def _is_nan(ir, df, be):
    # `x != x` is True exactly for NaN and null for null, which is what the engine returns —
    # `.isna()` would fold the two together and report NaN and null alike.
    x = be.column(eval_expr(ir["input"], df, be), df)
    return x != x


def _is_inf(ir, df, be):
    x = be.column(eval_expr(ir["input"], df, be), df)
    return (x == float("inf")) | (x == float("-inf"))


def _cast(ir, df, be):
    from batcher.plan.types.registry import DTYPE_REGISTRY

    target = DTYPE_REGISTRY.get(ir["dtype"])
    if target is None:
        raise Unsupported(f"cast to {ir['dtype']!r}")
    if ir.get("try_cast"):
        # A `try_cast` nulls the rows that fail rather than raising, and neither backend has
        # that mode — approximating it with a strict cast would raise on data the CPU engine
        # accepts.
        raise Unsupported("try_cast")
    x = be.column(eval_expr(ir["input"], df, be), df)
    import pyarrow as pa

    if pa.types.is_string(target) and be.is_float(x):
        # Float → string is a *formatting* decision, and the two implementations make three
        # different ones: whether an integral value keeps its `.0` (`"4"` against `"4.0"`),
        # what the sign of zero prints as, and whether `NaN` becomes the string `"nan"` or a
        # null. None of those is more correct than the others, which is exactly why this
        # cannot be reconciled — it has to be declined, or a column of numbers becomes a
        # column of subtly different text.
        raise Unsupported("cast float to string")
    if pa.types.is_integer(target) and be.is_float(x):
        # Float → integer **rounds**, half to even, where a direct `astype` instead raises on
        # any value with a fractional part. That is not a cast this path can decline: it is
        # the ordinary spelling of bucketing a measure, so refusing it would send every such
        # query to the host.
        x = apply_ufunc("rint", x, be)
    return x.astype(be.dtype(target))


def _case(ir, df, be):
    """`CASE WHEN … THEN … ELSE …` as a fold of `where` from the last branch backwards.

    A null predicate must select the *else* arm, matching the engine (a `WHEN` that is
    unknown is not taken), so each branch's condition is filled with False before use.
    """
    otherwise = ir.get("otherwise")
    if otherwise is None:
        raise Unsupported("CASE without ELSE")  # the null-typed default has no column dtype
    out = be.column(eval_expr(otherwise, df, be), df)
    for branch in reversed(ir["branches"]):
        cond = be.column(eval_expr(branch["when"], df, be), df).fillna(False)
        out = be.column(eval_expr(branch["then"], df, be), df).where(cond, out)
    return out


def _coalesce(ir, df, be):
    inputs = [be.column(eval_expr(i, df, be), df) for i in ir["inputs"]]
    out = inputs[0]
    for nxt in inputs[1:]:
        out = out.where(out.notna(), nxt)
    return out


def _nullif(ir, df, be):
    left = be.column(eval_expr(ir["left"], df, be), df)
    right = be.column(eval_expr(ir["right"], df, be), df)
    return left.where((left != right).fillna(True), None)


def _extreme(ir, df, be, *, want_max: bool):
    """`GREATEST`/`LEAST` — the extreme of the *non-null* arguments, null only if all are.

    This is the engine's semantics (verified against it), and it is not SQL's: DuckDB's
    `GREATEST` returns NULL when any argument is NULL. Folding pairwise keeps it exact on
    both backends without an `axis=1` reduction cuDF only partly supports.
    """
    inputs = [be.column(eval_expr(i, df, be), df) for i in ir["inputs"]]
    out = inputs[0]
    for nxt in inputs[1:]:
        # Through `_compare`, so a float `NaN` is the largest value here too — the engine's
        # `greatest` over a NaN returns NaN, and an IEEE comparison would return the other
        # argument instead.
        op = "ge" if want_max else "le"
        picked = (
            compare(op, out, nxt)
            if be.is_float(out) or be.is_float(nxt)
            else ((out >= nxt) if want_max else (out <= nxt))
        )
        out = out.where(picked.fillna(False), nxt).where(out.notna(), nxt).where(nxt.notna(), out)
    return out


def _in_list(ir, df, be):
    values = [literal_value(v) for v in ir["set"]]
    return be.column(eval_expr(ir["input"], df, be), df).isin(values)


def _list(ir, df, be: DfBackend):
    """A scalar reduction over each row's list.

    Neither library exposes these as a call both of them have, so they are built from `explode`
    plus `groupby` — see `vocab.lists`, which states why that construction is exact rather than
    close. The list→list functions (`sort`, `normalize`, `softmax`) are still declined: their
    result has to be reassembled into a list, and the only portable way to do that materializes
    a Python object per row, which is a hot-path tuple touch rather than a translation.
    """
    from batcher.core.gpu_plan.vocab import eval_list_fn

    return eval_list_fn(ir["fn"], be.column(eval_expr(ir["input"], df, be), df), be)


def _list_binary(ir, df, be: DfBackend):
    """A pairwise reduction over two list columns — the vector distances and the dot product."""
    from batcher.core.gpu_plan.vocab import eval_list_binary

    left = be.column(eval_expr(ir["left"], df, be), df)
    right = be.column(eval_expr(ir["right"], df, be), df)
    return eval_list_binary(ir["fn"], left, right, be)


def _list_get(ir, df, be: DfBackend):
    """`list[index]` — the index is a constant in every plan the engine builds."""
    from batcher.core.gpu_plan.vocab import eval_list_get

    return eval_list_get(be.column(eval_expr(ir["input"], df, be), df), int(ir["index"]), be)


def _list_contains(ir, df, be: DfBackend):
    from batcher.core.gpu_plan.vocab import eval_list_contains

    x = be.column(eval_expr(ir["input"], df, be), df)
    return eval_list_contains(x, literal_value(ir["value"]), be)


def _list_position(ir, df, be: DfBackend):
    from batcher.core.gpu_plan.vocab import eval_list_position

    x = be.column(eval_expr(ir["input"], df, be), df)
    return eval_list_position(x, literal_value(ir["value"]), be)


def _struct_field(ir, df, be: DfBackend):
    """`struct.field(name)` — one field of a struct column.

    The one struct operation both libraries spell the same way, and the one worth having: a
    struct column is how every semi-structured source arrives, so a plan that reads one field
    of one used to send its whole chain to the host.
    """
    from batcher.core.gpu_plan.backend import call_or_decline

    x = be.column(eval_expr(ir["input"], df, be), df)
    return call_or_decline(x.struct, "field", ir["field"])


def _make_temporal(ir, df, be: DfBackend):
    """An epoch or calendar constructor, whose arguments are sub-expressions."""
    from batcher.core.gpu_plan.vocab.dates import eval_make_temporal

    args = [be.column(eval_expr(a, df, be), df) for a in ir["args"]]
    return eval_make_temporal(ir["fn"], args, be)


def _window_start(ir, df, be: DfBackend):
    """`window_start` — the tumbling-window bucket key, whose width and origin are constants."""
    from batcher.core.gpu_plan.vocab.dates import eval_window_start

    x = be.column(eval_expr(ir["input"], df, be), df)
    return eval_window_start(x, int(ir["width_micros"]), int(ir.get("origin_micros", 0)), be)


def _date_offset(ir, df, be: DfBackend):
    """`offset_by` — a shift by months, days and micros, of which the zero ones are omitted.

    Not routed through `_named`: the offset components are fields of the node rather than a
    sub-expression, so the evaluator takes them directly and never needs `eval_expr`.
    """
    from batcher.core.gpu_plan.temporal import eval_date_offset

    x = be.column(eval_expr(ir["input"], df, be), df)
    return eval_date_offset(
        x, int(ir.get("months", 0)), int(ir.get("days", 0)), int(ir.get("micros", 0)), be
    )


def _named(handler):
    """Adapt a function-family evaluator to the handler signature.

    The families take `eval_expr` as an argument rather than importing it, so the vocabulary
    module does not import back into the dispatcher that dispatches to it.
    """
    return lambda ir, df, be: handler(ir, df, be, eval_expr)


_HANDLERS = {
    "col": _col,
    "lit": lambda ir, _df, _be: literal_value(ir["value"]),
    "binary": _named(eval_binary),
    "not": _not,
    "is_null": _is_null,
    "is_not_null": _is_not_null,
    "is_nan": _is_nan,
    "is_inf": _is_inf,
    "cast": _cast,
    "case": _case,
    "coalesce": _coalesce,
    "nullif": _nullif,
    "greatest": lambda ir, df, be: _extreme(ir, df, be, want_max=True),
    "least": lambda ir, df, be: _extreme(ir, df, be, want_max=False),
    "math": _named(eval_math),
    "math2": _named(eval_math2),
    "in_list": _in_list,
    "str": _named(eval_str),
    "date": _named(eval_date),
    "date_trunc": _named(eval_date_trunc),
    "date_offset": _date_offset,
    "list": _list,
    "list_binary": _list_binary,
    "list_get": _list_get,
    "list_contains": _list_contains,
    "list_position": _list_position,
    "struct_field": _struct_field,
    "window_start": _window_start,
    "make_temporal": _make_temporal,
    "strftime": _named(eval_strftime),
}


#: Every `bc_expr::Expr` tag this tier does not translate, and why.
#:
#: Exhaustive against `plan.ir_tags.ExprTag` by test. `_HANDLERS` is keyed by those tag
#: strings, so a tag renamed on the Rust and Python sides leaves its handler keyed on a dead
#: string — the expression stops being translated, the plan silently falls back to the CPU
#: engine, and nothing fails. Pinning both directions turns that into a red test.
DECLINED_EXPRS: dict[str, str] = {
    # Rust-only kernels. Decoding an image, resampling audio or running a geometry predicate
    # happens in `bc-expr::eval::{media,geo}`; there is no dataframe-library equivalent to
    # translate onto, and approximating one is exactly what this package refuses to do.
    "image": "media decode is a Rust kernel (`bc-expr::eval::media::image`)",
    "audio": "media decode is a Rust kernel (`bc-expr::eval::media::audio`)",
    "video": "media decode is a Rust kernel (`bc-expr::eval::media`)",
    "geo": "geometry is a Rust kernel (`bc-geo` + `bc-expr::eval::geo`)",
    # Not translated. Listed rather than absent so adding one is a decision with a date on it.
    "array": "not translated",
    "convert_timezone": "not translated",
    "hash": "not translated",
    "list_filter": "not translated",
    "list_join": "not translated",
    "list_set": "not translated",
    "list_simhash": "not translated",
    "list_slice": "not translated",
    "list_transform": "not translated",
    "list_zip": "not translated",
    "make_struct": "not translated",
    "map": "not translated",
    "sequence": "not translated",
    "strptime": "not translated",
    "window_buckets": "not translated",
}

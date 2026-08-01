"""The JSON IR vocabulary — the single Python home for the wire-contract tags.

Python's `to_ir()` and Rust's `serde` enums are two statements of one contract; they
must agree byte-for-byte (`CLAUDE.md` invariant #8). Keeping the Python side's tag
*strings* here — rather than scattered as literals across ~30 `to_ir()` methods —
gives the contract one documented home and turns a typo into an `AttributeError`
(`Op.SCNA`) instead of a silently-wrong tag that only a differential test would catch.

This module is pure constants: it imports nothing from the plan/subsystem layers, so
it stays in the neutral `plan` package without risking an import cycle. Rust remains
the authority for its own serde tags; the two are reconciled by the round-trip /
differential tests, never by code generation across the boundary.
"""

from __future__ import annotations

from typing import Final


class Op:
    """`RelOp` discriminator tags — the ``"op"`` field of a node's `to_ir()`.

    Values mirror `bc_ir::RelOp` serde tags exactly; changing one requires changing
    the Rust side in the same commit plus a round-trip test.
    """

    SCAN: Final = "scan"
    FILTER: Final = "filter"
    PROJECT: Final = "project"
    AGGREGATE: Final = "aggregate"
    SORT: Final = "sort"
    HASH_JOIN: Final = "hash_join"
    DISTINCT: Final = "distinct"
    UNION: Final = "union"
    WINDOW: Final = "window"
    LIMIT: Final = "limit"
    UNNEST: Final = "unnest"
    UNPIVOT: Final = "unpivot"
    ROW_ID: Final = "row_id"
    SAMPLE: Final = "sample"
    ASOF_JOIN: Final = "asof_join"
    RANGE_JOIN: Final = "range_join"


class ExprTag:
    """Scalar `Expr` discriminator tags — the ``"e"`` field of an expression's
    `to_ir()`. Values mirror `bc_expr::Expr` serde tags exactly.
    """

    COL: Final = "col"
    LIT: Final = "lit"
    BINARY: Final = "binary"
    NOT: Final = "not"
    CAST: Final = "cast"
    IS_NULL: Final = "is_null"
    IS_NOT_NULL: Final = "is_not_null"
    IS_NAN: Final = "is_nan"
    IS_INF: Final = "is_inf"
    CASE: Final = "case"
    STR: Final = "str"
    MATH: Final = "math"
    MATH2: Final = "math2"
    COALESCE: Final = "coalesce"
    IN_LIST: Final = "in_list"
    NULLIF: Final = "nullif"
    GREATEST: Final = "greatest"
    LEAST: Final = "least"
    ARRAY: Final = "array"
    HASH: Final = "hash"
    SEQUENCE: Final = "sequence"
    DATE: Final = "date"
    DATE_TRUNC: Final = "date_trunc"
    DATE_OFFSET: Final = "date_offset"
    WINDOW_START: Final = "window_start"
    WINDOW_BUCKETS: Final = "window_buckets"
    STRFTIME: Final = "strftime"
    STRPTIME: Final = "strptime"
    CONVERT_TIMEZONE: Final = "convert_timezone"
    LIST: Final = "list"
    LIST_BINARY: Final = "list_binary"
    LIST_JOIN: Final = "list_join"
    LIST_GET: Final = "list_get"
    LIST_SIMHASH: Final = "list_simhash"
    LIST_CONTAINS: Final = "list_contains"
    LIST_POSITION: Final = "list_position"
    LIST_SET: Final = "list_set"
    LIST_ZIP: Final = "list_zip"
    LIST_TRANSFORM: Final = "list_transform"
    LIST_FILTER: Final = "list_filter"
    LIST_SLICE: Final = "list_slice"
    STRUCT_FIELD: Final = "struct_field"
    MAKE_STRUCT: Final = "make_struct"
    MAKE_TEMPORAL: Final = "make_temporal"
    MAP: Final = "map"
    GEO: Final = "geo"
    IMAGE: Final = "image"
    AUDIO: Final = "audio"
    VIDEO: Final = "video"


# The `Binary` comparison operators, mirroring the Rust `BinaryOp` serde tags.
#
# Fourteen modules across Kyber were spelling this set out for themselves — as a frozenset, a
# tuple, a dict keyed by it, and (twice, adjacently, in one file) the same dict literal — and
# `ruff` reports none of that: `F811` does not fire on a module-level constant reassignment.
# The spellings had already drifted apart, which is the cost: one of them omitted `eq`/`ne`
# while carrying the same name as the ones that did not, so whether a rule saw an equality
# predicate depended on which module it happened to be written in.
#
# A rule that genuinely wants a *subset* takes `ORDERING_COMPARISONS` or names its own for
# what it is; what it must not do is redefine "the comparisons" to mean something narrower.
COMPARISON_OPS: Final = frozenset({"eq", "ne", "lt", "le", "gt", "ge"})
# The order-only comparisons. Separate because range/zonemap reasoning is defined by an
# interval endpoint moving, which `eq`/`ne` do not do: `eq` is a degenerate interval and `ne`
# is not an interval at all, so admitting them to a bounds walk widens it wrongly.
ORDERING_COMPARISONS: Final = frozenset({"lt", "le", "gt", "ge"})

# Window-function names, mirroring the Rust `WindowFn` enum (serde snake_case).
# Ranking functions take no input; "value" functions select a row's value by offset
# (input required); the aggregates run as windowed/running aggregates.
WINDOW_RANKING: Final = frozenset(
    {"row_number", "rank", "dense_rank", "percent_rank", "cume_dist", "ntile"}
)
WINDOW_AGGREGATES: Final = frozenset(
    {
        "sum", "avg", "min", "max", "count",
        # DuckDB, Spark and Polars all allow any aggregate over a window. These are the
        # ones whose *running* form costs O(1) per row, which is what lets them share the
        # existing whole-partition and running machinery. Order statistics
        # (`median`/`quantile`/`mode`) need a sorted structure and are deliberately
        # absent — see `bc_runtime::window_agg`.
        "var", "stddev", "product",
        "bool_and", "bool_or", "bit_and", "bit_or", "bit_xor", "count_distinct",
    }
)  # fmt: skip
# The fills select by *nullness* rather than by offset, but share the value functions'
# contract: input required, output type = input type, no explicit frame (theirs is
# implied). Unlike the other value functions they are meaningless without an order.
WINDOW_FILL: Final = frozenset({"forward_fill", "backward_fill"})
WINDOW_VALUE: Final = (
    frozenset({"first_value", "last_value", "lag", "lead", "nth_value"}) | WINDOW_FILL
)
WINDOW_FUNCS: Final = WINDOW_RANKING | WINDOW_AGGREGATES | WINDOW_VALUE
# Functions that honour an explicit frame: the reducing aggregates, plus the
# positional value functions that pick the frame's first/last/nth row. `lag`/`lead`
# and the fills carry no frame (theirs is fixed by their own offset / nullness).
# Functions that honour an explicit `ROWS`/`GROUPS` frame.
#
# The six folds joined the original five once the framed path's two-stack slide was
# generalized from `+` to any associative, commutative operator (`bc_runtime::window_agg`).
# That structure exists because the naive O(1) slide needs an *inverse* to un-apply the
# leaving value, and `product` cannot divide out a zero while `bit_and`/`bool_and` cannot
# un-AND at all.
#
# `var`/`stddev` and `count_distinct` are still absent, and for reasons the slide cannot
# fix: the moment pair keeps a Welford state whose combine is Chan's parallel formula
# rather than an operator, and a distinct count needs a multiset rather than a fold.
# Listing either here would send a frame to a kernel that cannot honour it.
WINDOW_FRAMEABLE: Final = frozenset(
    {
        "sum", "avg", "min", "max", "count", "first_value", "last_value", "nth_value",
        "product", "bool_and", "bool_or", "bit_and", "bit_or", "bit_xor",
    }
)  # fmt: skip

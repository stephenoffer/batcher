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
    SPATIAL: Final = "spatial"
    IMAGE: Final = "image"
    IMAGE_CROP: Final = "image_crop"
    AUDIO: Final = "audio"
    VIDEO: Final = "video"
    SEQ: Final = "seq"


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
#: Microseconds in a day — the conversion between the IR's *day* offsets and its
#: *timestamp* unit. The engine's timestamps are microseconds (`timestamp[us]`), while
#: `DateOffset` and a `datetime.timedelta` both count whole days separately, so every
#: place that turns one into the other multiplies by this.
#:
#: It lives here because it was written out six times under four names — `_DAY_MICROS`,
#: `_MICROS_PER_DAY`, `_DAY_US`, and three bare literals — across `plan`, `kyber` and
#: `core`. Two of those are subsystems that MUST NOT import each other
#: (`.claude/rules/architecture.md`), so copy-paste was the only way they could share it,
#: and copy-paste is exactly the wrong way: the neutral `plan` layer is the one both may
#: read from. Same reason `_median` should not have been pasted into three subsystems.
MICROS_PER_DAY: Final = 86_400_000_000

COMPARISON_OPS: Final = frozenset({"eq", "ne", "lt", "le", "gt", "ge"})
#: The same six in a FIXED order, for the callers that *generate* something per operator —
#: Kyber registers one rule per comparison, and registration order is run order. Iterating the
#: frozenset instead reorders those rules on every interpreter run, which no correctness test
#: can see (each rule is individually semantics-preserving) and `just lint-rule-order` fails
#: on. Use `COMPARISON_OPS` to ask "is this a comparison?"; use this to build one thing each.
COMPARISON_ORDER: Final = ("eq", "ne", "lt", "le", "gt", "ge")
# The order-only comparisons. Separate because range/zonemap reasoning is defined by an
# interval endpoint moving, which `eq`/`ne` do not do: `eq` is a degenerate interval and `ne`
# is not an interval at all, so admitting them to a bounds walk widens it wrongly.
ORDERING_COMPARISONS: Final = frozenset({"lt", "le", "gt", "ge"})
#: The ordering comparisons in a FIXED order, for per-operator *generation*. Same reason as
#: `COMPARISON_ORDER`: a rule built per operator inherits the iteration order as its run order.
ORDERING_ORDER: Final = ("lt", "le", "gt", "ge")

#: The comparison a predicate becomes when its operands swap: `lit < col` == `col > lit`.
#: Twenty-three call sites spelled this map out — optimizer rules, the IO predicate pushdown,
#: and four lakehouse/NoSQL connectors each with their own name for it — which is the same
#: fact written twenty-three times.
COMPARISON_FLIP: Final = {
    "lt": "gt",
    "gt": "lt",
    "le": "ge",
    "ge": "le",
    "eq": "eq",
    "ne": "ne",
}
#: The flip restricted to the ordering comparisons, for a caller doing interval reasoning
#: where `eq`/`ne` are not intervals and admitting them widens the walk wrongly.
ORDERING_FLIP: Final = {"lt": "gt", "gt": "lt", "le": "ge", "ge": "le"}

#: Join types whose output rows are bounded by the LEFT side — so an empty left is empty, the
#: left may drive a broadcast, and a tree-shard may split on it. `right`/`full` are absent
#: because they pad the other side's rows through.
LEFT_DRIVEN_JOINS: Final = frozenset({"inner", "left", "semi", "anti"})

#: Aggregates with an O(1) running form and a metadata-answerable value — the set a rolling
#: window, a pivot, a statistics read and the device translator all independently support.
#: Spelled `mean` (the public API's name), not `avg` (the IR's).
RUNNING_AGGREGATES: Final = frozenset({"sum", "min", "max", "count", "mean"})

#: Binary operators safe to hand a compiled/JIT path and to reason about structurally: total
#: on their input types, no null-propagation surprises, no division.
SAFE_BINARY_OPS: Final = frozenset(
    {"add", "sub", "mul", "and", "or", "eq", "ne", "lt", "le", "gt", "ge"}
)

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
        # absent — see `bc_runtime::window::agg`.
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
# The whole-prefix recurrences (`bc_runtime::window::series`). Each row's answer is a
# function of the entire ordered prefix carried in a running state, which is why none of
# them takes a frame — there is no subset of rows to aggregate — and why all of them
# require an ORDER BY, exactly as the fills do.
WINDOW_EWM: Final = frozenset({"ewm_mean", "ewm_var", "ewm_std"})
WINDOW_SERIES: Final = WINDOW_EWM | frozenset({"interpolate", "rle_id"})
WINDOW_FUNCS: Final = WINDOW_RANKING | WINDOW_AGGREGATES | WINDOW_VALUE | WINDOW_SERIES
# Functions that honour an explicit frame: the reducing aggregates, plus the
# positional value functions that pick the frame's first/last/nth row. `lag`/`lead`
# and the fills carry no frame (theirs is fixed by their own offset / nullness).
# Functions that honour an explicit `ROWS`/`GROUPS` frame.
#
# The six folds joined the original five once the framed path's two-stack slide was
# generalized from `+` to any associative, commutative operator (`bc_runtime::window::agg`).
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

# The GROUP BY aggregate vocabulary, mirroring the Rust `AggFunc` enum (serde snake_case).
#
# This is *not* `WINDOW_FUNCS`, and the difference is not cosmetic: a grouped average is
# `mean` while a windowed one is `avg`, so one `AggExpr` carries two vocabularies depending
# on whether `.over()` is called. `Aggregate` validates against this set, which is why
# `AggExpr` itself cannot — `AggExpr("avg", x).over(...)` is correct and must stay legal.
AGG_FNS: Final = frozenset(
    {
        "any_value", "approx_count_distinct", "approx_quantile", "approx_top_k",
        "arg_max", "arg_min", "bit_and", "bit_or", "bit_xor", "bool_and", "bool_or",
        "corr", "count", "count_distinct", "count_star", "covar_pop", "covar_samp",
        "entropy", "histogram", "kahan_sum", "kurtosis", "kurtosis_pop", "list_agg",
        "mad", "max", "mean", "median", "min", "mode", "product", "quantile",
        "quantile_disc", "skewness", "stddev", "sum", "var",
        # Assembly contiguity. `n_length`/`l_count` carry their fraction in `param`, as
        # `quantile` does; `n50`/`n90`/`l50` are the public spellings over them.
        "n_length", "l_count", "aun",
    }
)  # fmt: skip

#: How a window frame counts its offsets, mirroring the Rust `FrameUnits` enum.
FRAME_UNITS: Final = frozenset({"rows", "range", "groups"})

#: The `kind` discriminator of one frame edge, mirroring the Rust `FrameBound` enum, which
#: is `#[serde(tag = "kind")]`. `preceding`/`following` additionally carry `n`.
FRAME_BOUND_KINDS: Final = frozenset(
    {"unbounded_preceding", "preceding", "current_row", "following", "unbounded_following"}
)

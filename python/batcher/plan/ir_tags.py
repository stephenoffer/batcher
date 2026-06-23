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
    SAMPLE: Final = "sample"
    ASOF_JOIN: Final = "asof_join"


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
    CASE: Final = "case"
    STR: Final = "str"
    MATH: Final = "math"
    MATH2: Final = "math2"
    COALESCE: Final = "coalesce"
    NULLIF: Final = "nullif"
    GREATEST: Final = "greatest"
    LEAST: Final = "least"
    ARRAY: Final = "array"
    DATE: Final = "date"
    DATE_TRUNC: Final = "date_trunc"
    DATE_OFFSET: Final = "date_offset"
    STRFTIME: Final = "strftime"
    STRPTIME: Final = "strptime"
    LIST: Final = "list"
    LIST_BINARY: Final = "list_binary"
    LIST_JOIN: Final = "list_join"
    LIST_GET: Final = "list_get"
    LIST_CONTAINS: Final = "list_contains"
    LIST_SLICE: Final = "list_slice"
    STRUCT_FIELD: Final = "struct_field"
    IMAGE: Final = "image"


# Window-function names, mirroring the Rust `WindowFn` enum (serde snake_case).
# Ranking functions take no input; "value" functions select a row's value by offset
# (input required); the aggregates run as windowed/running aggregates.
WINDOW_RANKING: Final = frozenset(
    {"row_number", "rank", "dense_rank", "percent_rank", "cume_dist", "ntile"}
)
WINDOW_AGGREGATES: Final = frozenset({"sum", "avg", "min", "max", "count"})
WINDOW_VALUE: Final = frozenset({"first_value", "last_value", "lag", "lead"})
WINDOW_FUNCS: Final = WINDOW_RANKING | WINDOW_AGGREGATES | WINDOW_VALUE

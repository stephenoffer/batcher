"""Golden values for the IR wire-contract tags.

`plan/ir_tags.py` is the Python side of a contract with Rust's serde enums. These
tests pin every tag *value* so an accidental edit (a typo, a rename) fails here
loudly instead of silently shipping a wrong tag that only a differential test would
catch. If a value legitimately changes, the Rust serde tag changes in the same
commit and this golden updates with it.

The goldens are compared against the *whole* public surface of each class rather
than iterated over their own keys. Iterating the golden's keys is what let it drift
to covering 10 of 16 `Op` tags and 25 of 49 `ExprTag` tags while its docstring
claimed to pin every one: the six operators added later — `unnest`, `unpivot`,
`row_id`, `sample`, `asof_join`, `range_join` — could be renamed with this file
green. Comparing the dicts whole means a new tag fails here until it is pinned.

This checks the Python side only. `test_ir_contract.py` is the cross-language half,
deriving what Rust will actually accept from the enum definitions.
"""

from __future__ import annotations

import batcher as bt
from batcher import col, lit
from batcher.plan.ir_tags import ExprTag, Op

OP_GOLDEN = {
    "AGGREGATE": "aggregate",
    "ASOF_JOIN": "asof_join",
    "DISTINCT": "distinct",
    "FILTER": "filter",
    "HASH_JOIN": "hash_join",
    "LIMIT": "limit",
    "PROJECT": "project",
    "RANGE_JOIN": "range_join",
    "ROW_ID": "row_id",
    "SAMPLE": "sample",
    "SCAN": "scan",
    "SORT": "sort",
    "UNION": "union",
    "UNNEST": "unnest",
    "UNPIVOT": "unpivot",
    "WINDOW": "window",
}

EXPR_GOLDEN = {
    "ARRAY": "array",
    "AUDIO": "audio",
    "BINARY": "binary",
    "CASE": "case",
    "CAST": "cast",
    "COALESCE": "coalesce",
    "COL": "col",
    "CONVERT_TIMEZONE": "convert_timezone",
    "DATE": "date",
    "DATE_OFFSET": "date_offset",
    "DATE_TRUNC": "date_trunc",
    "GEO": "geo",
    "GREATEST": "greatest",
    "HASH": "hash",
    "IMAGE": "image",
    "IN_LIST": "in_list",
    "IS_INF": "is_inf",
    "IS_NAN": "is_nan",
    "IS_NOT_NULL": "is_not_null",
    "IS_NULL": "is_null",
    "LEAST": "least",
    "LIST": "list",
    "LIST_BINARY": "list_binary",
    "LIST_CONTAINS": "list_contains",
    "LIST_FILTER": "list_filter",
    "LIST_GET": "list_get",
    "LIST_JOIN": "list_join",
    "LIST_POSITION": "list_position",
    "LIST_SET": "list_set",
    "LIST_SIMHASH": "list_simhash",
    "LIST_SLICE": "list_slice",
    "LIST_TRANSFORM": "list_transform",
    "LIST_ZIP": "list_zip",
    "LIT": "lit",
    "MAKE_STRUCT": "make_struct",
    "MAKE_TEMPORAL": "make_temporal",
    "MAP": "map",
    "MATH": "math",
    "MATH2": "math2",
    "NOT": "not",
    "NULLIF": "nullif",
    "SEQUENCE": "sequence",
    "STR": "str",
    "STRFTIME": "strftime",
    "STRPTIME": "strptime",
    "STRUCT_FIELD": "struct_field",
    "VIDEO": "video",
    "WINDOW_BUCKETS": "window_buckets",
    "WINDOW_START": "window_start",
}


def _public_tags(cls: type) -> dict[str, str]:
    """Every tag the class actually exposes, so a golden cannot silently under-cover."""
    return {
        name: value
        for name, value in vars(cls).items()
        if not name.startswith("_") and isinstance(value, str)
    }


def test_op_tag_values():
    assert _public_tags(Op) == OP_GOLDEN


def test_expr_tag_values():
    assert _public_tags(ExprTag) == EXPR_GOLDEN


def test_to_ir_uses_the_centralized_tags():
    """A real plan's lowered IR carries exactly the centralized tag strings."""
    plan = (
        bt.from_pydict({"k": [1, 2], "x": [10, 20]})
        .filter(col("x") > lit(5))
        .group_by("k")
        .agg(s=col("x").sum())
        ._plan
    )
    ir = plan.to_ir()
    assert ir["op"] == Op.AGGREGATE
    assert ir["input"]["op"] == Op.FILTER
    assert ir["input"]["predicate"]["e"] == ExprTag.BINARY
    assert ir["input"]["input"]["op"] == Op.SCAN

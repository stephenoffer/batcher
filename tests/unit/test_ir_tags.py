"""Golden values for the IR wire-contract tags.

`plan/ir_tags.py` is the Python side of a contract with Rust's serde enums. These
tests pin every tag *value* so an accidental edit (a typo, a rename) fails here
loudly instead of silently shipping a wrong tag that only a differential test would
catch. If a value legitimately changes, the Rust serde tag changes in the same
commit and this golden updates with it.
"""

from __future__ import annotations

import batcher as bt
from batcher import col, lit
from batcher.plan.ir_tags import ExprTag, Op

OP_GOLDEN = {
    "SCAN": "scan",
    "FILTER": "filter",
    "PROJECT": "project",
    "AGGREGATE": "aggregate",
    "SORT": "sort",
    "HASH_JOIN": "hash_join",
    "DISTINCT": "distinct",
    "UNION": "union",
    "WINDOW": "window",
    "LIMIT": "limit",
}

EXPR_GOLDEN = {
    "COL": "col",
    "LIT": "lit",
    "BINARY": "binary",
    "NOT": "not",
    "CAST": "cast",
    "IS_NULL": "is_null",
    "IS_NOT_NULL": "is_not_null",
    "CASE": "case",
    "STR": "str",
    "MATH": "math",
    "MATH2": "math2",
    "COALESCE": "coalesce",
    "NULLIF": "nullif",
    "GREATEST": "greatest",
    "LEAST": "least",
    "DATE": "date",
    "DATE_TRUNC": "date_trunc",
    "LIST": "list",
    "LIST_GET": "list_get",
    "LIST_CONTAINS": "list_contains",
    "LIST_SLICE": "list_slice",
    "STRUCT_FIELD": "struct_field",
}


def test_op_tag_values():
    assert {k: getattr(Op, k) for k in OP_GOLDEN} == OP_GOLDEN


def test_expr_tag_values():
    assert {k: getattr(ExprTag, k) for k in EXPR_GOLDEN} == EXPR_GOLDEN


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

"""Operator-mix: string execution over TPC-H ``lineitem``.

The suite had no string-expression family at all, which is the one place the scorecard
already records a structural loss: ``docs/architecture/internals/competitive_architecture.md``
ceiling 2 ("no string-optimized representation -- no ``StringView``, dictionary decoded at
the leaf"). A gap with no case in the suite cannot be tracked, and cannot be shown to have
closed.

Each case isolates a different cost a string representation decides:

* **substring search** (``op-str-like-contains``) -- the shape ``StringView`` and a
  Volnitsky/two-way search win, because a non-matching row is rejected on its inline prefix.
* **prefix match** (``op-str-like-prefix``) -- decided by the first four bytes, so an inline
  prefix answers it without dereferencing the heap at all.
* **decode cost** (``op-str-length``) -- no comparison, just walking every offset.
* **transform then group** (``op-str-upper-group``) -- a produced string feeding a key.
* **a computed key with many groups** (``op-str-substring-group``) -- where the group-by pays
  for a string it has to build first.
* **concatenation into a distinct** (``op-str-concat-distinct``) -- allocation-bound.
* **replacement** (``op-str-replace-length``) -- a per-row rewrite of the whole column.

Every case returns an aggregate, so the correctness gate compares a handful of rows rather
than six million strings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

from registry import suite

from .base import sql_fanout, with_native

if TYPE_CHECKING:
    from context import Context

strings = suite("ops-strings", dataset="operators")


@strings.case("op-str-like-contains")
def like_contains(ctx: Context):
    """COUNT WHERE l_comment LIKE '%requests%' -- unanchored substring search over 6M strings."""
    sql = "SELECT COUNT(*) AS n FROM lineitem WHERE l_comment LIKE '%requests%'"

    def pyarrow(t: pa.Table) -> pa.Table:
        mask = pc.match_substring(t["l_comment"], "requests")
        return pa.table({"n": pa.array([pc.sum(pc.cast(mask, pa.int64())).as_py()])})

    return with_native(ctx, sql_fanout(ctx, sql), pyarrow=pyarrow)


@strings.case("op-str-like-prefix")
def like_prefix(ctx: Context):
    """COUNT WHERE l_comment LIKE 'the%' -- decided by the leading bytes alone."""
    sql = "SELECT COUNT(*) AS n FROM lineitem WHERE l_comment LIKE 'the%'"

    def pyarrow(t: pa.Table) -> pa.Table:
        mask = pc.starts_with(t["l_comment"], pattern="the")
        return pa.table({"n": pa.array([pc.sum(pc.cast(mask, pa.int64())).as_py()])})

    return with_native(ctx, sql_fanout(ctx, sql), pyarrow=pyarrow)


@strings.case("op-str-length")
def str_length(ctx: Context):
    """SUM(length(l_comment)) -- offset walking with no comparison and no allocation."""
    sql = "SELECT SUM(LENGTH(l_comment)) AS s FROM lineitem"

    def pyarrow(t: pa.Table) -> pa.Table:
        return pa.table({"s": pa.array([pc.sum(pc.utf8_length(t["l_comment"])).as_py()])})

    return with_native(ctx, sql_fanout(ctx, sql), pyarrow=pyarrow)


@strings.case("op-str-upper-group")
def upper_group(ctx: Context):
    """GROUP BY upper(l_shipmode) -- a produced string used as a low-cardinality key."""
    sql = "SELECT UPPER(l_shipmode) AS m, COUNT(*) AS n FROM lineitem GROUP BY UPPER(l_shipmode)"

    def pyarrow(t: pa.Table) -> pa.Table:
        up = pc.utf8_upper(t["l_shipmode"])
        a = pa.table({"m": up}).group_by("m").aggregate([([], "count_all")])
        return pa.table({"m": a["m"], "n": a["count_all"]})

    return with_native(ctx, sql_fanout(ctx, sql), pyarrow=pyarrow)


@strings.case("op-str-substring-group")
def substring_group(ctx: Context):
    """GROUP BY the first four characters of l_comment -- a built key with many groups."""
    sql = (
        "SELECT SUBSTRING(l_comment, 1, 4) AS p, COUNT(*) AS n "
        "FROM lineitem GROUP BY SUBSTRING(l_comment, 1, 4)"
    )

    def pyarrow(t: pa.Table) -> pa.Table:
        p = pc.utf8_slice_codeunits(t["l_comment"], 0, 4)
        a = pa.table({"p": p}).group_by("p").aggregate([([], "count_all")])
        return pa.table({"p": a["p"], "n": a["count_all"]})

    return with_native(ctx, sql_fanout(ctx, sql), pyarrow=pyarrow)


@strings.case("op-str-concat-distinct")
def concat_distinct(ctx: Context):
    """COUNT(DISTINCT l_shipmode || '|' || l_shipinstruct) -- allocate, then key on the result."""
    sql = "SELECT COUNT(DISTINCT l_shipmode || '|' || l_shipinstruct) AS n FROM lineitem"

    def pyarrow(t: pa.Table) -> pa.Table:
        joined = pc.binary_join_element_wise(t["l_shipmode"], t["l_shipinstruct"], "|")
        return pa.table({"n": pa.array([len(pc.unique(joined))], type=pa.int64())})

    return with_native(ctx, sql_fanout(ctx, sql), pyarrow=pyarrow)


@strings.case("op-str-replace-length")
def replace_length(ctx: Context):
    """SUM(length(replace(l_comment,'e',''))) -- a per-row rewrite of the whole column."""
    sql = "SELECT SUM(LENGTH(REPLACE(l_comment, 'e', ''))) AS s FROM lineitem"

    def pyarrow(t: pa.Table) -> pa.Table:
        rep = pc.replace_substring(t["l_comment"], pattern="e", replacement="")
        return pa.table({"s": pa.array([pc.sum(pc.utf8_length(rep)).as_py()])})

    return with_native(ctx, sql_fanout(ctx, sql), pyarrow=pyarrow)


@strings.case("op-str-regexp")
def regexp(ctx: Context):
    """COUNT WHERE l_comment matches `^[a-z]+ ly` -- a real regex automaton, not a LIKE."""
    sql = "SELECT COUNT(*) AS n FROM lineitem WHERE REGEXP_MATCHES(l_comment, '^[a-z]+ ly')"

    def pyarrow(t: pa.Table) -> pa.Table:
        n = pc.sum(pc.match_substring_regex(t["l_comment"], "^[a-z]+ ly")).as_py()
        return pa.table({"n": pa.array([n], type=pa.int64())})

    return with_native(ctx, sql_fanout(ctx, sql), pyarrow=pyarrow)

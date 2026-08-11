"""A partly-translatable `AND` pushes the part that translated.

Dropping a conjunct only widens the rows a source returns, and the engine's `Filter`
re-checks all of them, so the SQL/Mongo/Iceberg read translators keep the pushable half
instead of declining outright. They used to decline, which turned one unpushable term in
a six-predicate warehouse query into a full table extract over the network.

An `OR` is the opposite case and stays all-or-nothing: dropping a disjunct *narrows* the
filter, which loses rows. So does a predicate that picks rows to replace rather than rows
to skip, which is why `to_iceberg_expression` only widens when asked.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.io.predicate import to_mongo_filter, to_pyarrow_expression, to_sql_where

pytestmark = pytest.mark.unit

# `x > 5` translates; `x > y` (column vs column) does not.
_PUSHABLE = bt.col("x") > 5
_UNPUSHABLE = bt.col("x") > bt.col("y")


def test_sql_keeps_the_translatable_conjunct():
    assert to_sql_where((_PUSHABLE & _UNPUSHABLE).to_ir()) == "x > 5"
    assert to_sql_where((_UNPUSHABLE & _PUSHABLE).to_ir()) == "x > 5"


def test_sql_keeps_both_conjuncts_when_both_translate():
    where = to_sql_where((_PUSHABLE & (bt.col("y") == 3)).to_ir())
    assert where == "(x > 5 AND y = 3)"


def test_sql_declines_an_or_with_an_untranslatable_side():
    # Keeping only `x > 5` would drop every row matched by the other disjunct.
    assert to_sql_where((_PUSHABLE | _UNPUSHABLE).to_ir()) is None
    assert to_sql_where((_UNPUSHABLE | _PUSHABLE).to_ir()) is None


def test_sql_declines_when_neither_side_translates():
    assert to_sql_where((_UNPUSHABLE & _UNPUSHABLE).to_ir()) is None


def test_sql_keeps_the_translatable_conjunct_of_a_nested_and():
    ir = (_PUSHABLE & (bt.col("y") == 3) & _UNPUSHABLE).to_ir()
    assert to_sql_where(ir) == "(x > 5 AND y = 3)"


def test_mongo_keeps_the_translatable_conjunct():
    assert to_mongo_filter((_PUSHABLE & _UNPUSHABLE).to_ir()) == {"x": {"$gt": 5}}


def test_mongo_declines_a_partial_or():
    assert to_mongo_filter((_PUSHABLE | _UNPUSHABLE).to_ir()) is None


def test_mongo_combines_two_translatable_conjuncts():
    got = to_mongo_filter((_PUSHABLE & (bt.col("y") == 3)).to_ir())
    assert got == {"$and": [{"x": {"$gt": 5}}, {"y": {"$eq": 3}}]}


def test_pyarrow_still_keeps_the_translatable_conjunct():
    # The behavior the other backends are being brought in line with.
    assert to_pyarrow_expression((_PUSHABLE & _UNPUSHABLE).to_ir()) is not None
    assert to_pyarrow_expression((_PUSHABLE | _UNPUSHABLE).to_ir()) is None


def test_iceberg_widens_only_when_asked():
    ie = pytest.importorskip("pyiceberg.expressions")
    from batcher.io.predicate import to_iceberg_expression

    ir = (_PUSHABLE & _UNPUSHABLE).to_ir()
    # `replace_where` must never widen: it chooses the rows to overwrite.
    assert to_iceberg_expression(ir) is None
    # A scan may: the filter only prunes what is read.
    assert to_iceberg_expression(ir, allow_partial=True) == ie.GreaterThan("x", 5)

"""Disjunction implied-predicate inference preserves results vs DuckDB.

The `disjunction_infer` rule adds per-column ``IN`` predicates implied by a
multi-column DNF (``(a=1 AND b=2) OR (a=3 AND b=4)``) so pushdown can sink them onto
each table's scan. The derived predicates are provable supersets, so the query
result must be byte-for-byte identical to DuckDB — including the join-through-rename
shape (a self-join on the same table, the TPC-H Q7 pattern) that requires the pushed
``IN``'s inner column to be renamed as it crosses the alias projection.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt

# Registers the @rule on import so the full optimizer applies it under `.collect()`.
import batcher.kyber.rules.extra.disjunction_infer
from _harness import assert_same
from batcher import col


def _dim(duck):
    t = pa.table(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["FRANCE", "GERMANY", "SPAIN", "ITALY", None],
        }
    )
    duck.register("dim", t)
    return bt.from_arrow(t)


def _facts(duck):
    t = pa.table(
        {
            "a": [1, 2, 3, 1, 2, 4, None],
            "b": [2, 3, 4, 4, 1, 5, 2],
            "v": [10, 20, 30, 40, 50, 60, 70],
        }
    )
    duck.register("facts", t)
    return bt.from_arrow(t)


def test_two_column_dnf(duck):
    ds = _facts(duck).filter(
        ((col("a") == 1) & (col("b") == 2)) | ((col("a") == 3) & (col("b") == 4))
    )
    assert_same(
        ds.collect(),
        duck.sql("SELECT * FROM facts WHERE (a=1 AND b=2) OR (a=3 AND b=4)"),
    )


def test_dnf_with_nulls_and_no_match(duck):
    # A pair that matches nothing plus a null-bearing column — the derived IN must not
    # admit a null row the disjunction rejects.
    ds = _facts(duck).filter(
        ((col("a") == 9) & (col("b") == 9)) | ((col("a") == 2) & (col("b") == 1))
    )
    assert_same(
        ds.collect(),
        duck.sql("SELECT * FROM facts WHERE (a=9 AND b=9) OR (a=2 AND b=1)"),
    )


def test_self_join_nation_pair_q7_shape(duck):
    # TPC-H Q7's core: the same dimension joined twice under aliases, with a
    # cross-alias disjunction. The derived `name IN (FRANCE, GERMANY)` on each side must
    # push through the alias rename onto that scan — the case that requires InList's
    # inner column to be renamed by `substitute_columns`.
    dim = _dim(duck)
    facts = _facts(duck)
    joined = (
        facts.join(dim.rename({"name": "a_name", "id": "a_id"}), left_on="a", right_on="a_id")
        .join(dim.rename({"name": "b_name", "id": "b_id"}), left_on="b", right_on="b_id")
        .filter(
            ((col("a_name") == "FRANCE") & (col("b_name") == "GERMANY"))
            | ((col("a_name") == "GERMANY") & (col("b_name") == "FRANCE"))
        )
        .select(col("v"), col("a_name"), col("b_name"))
    )
    sql = """
        SELECT f.v, d1.name AS a_name, d2.name AS b_name
        FROM facts f
        JOIN dim d1 ON f.a = d1.id
        JOIN dim d2 ON f.b = d2.id
        WHERE (d1.name = 'FRANCE' AND d2.name = 'GERMANY')
           OR (d1.name = 'GERMANY' AND d2.name = 'FRANCE')
    """
    assert_same(joined.collect(), duck.sql(sql))

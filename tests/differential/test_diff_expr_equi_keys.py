"""Differential tests vs DuckDB for equi-join keys over expressions (`a.w = b.w - 52`).

`derive_join_keys` keys a join on an equality with an expression operand by computing the
expression as a hidden `__eqk_` column below the join (`kyber.rules.equi_expr_keys`). The
rewrite moves the expression from the joined rows to the input rows, so these cases carry
what could make that visible: NULL keys, keys that match nothing, many-to-many duplicates,
an empty side, the literal on either side of the `+`, and the operands written in either
order across the join.

The refused shapes (a float operand, a multiplication, an oversized literal) must still match
DuckDB, and must not carry the hidden key. Each accepted shape asserts the key *is* in the
plan, which is the positive control that keeps the refusals from passing vacuously.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

_HIDDEN = "__eqk_"


def _tables(*, rows: int = 4_000, empty_right: bool = False) -> dict[str, pa.Table]:
    rng = np.random.default_rng(11)

    def side(prefix: str, n: int) -> pa.Table:
        return pa.table(
            {
                f"{prefix}_k": pa.array(rng.integers(0, 7, n), mask=rng.random(n) < 0.03),
                f"{prefix}_w": pa.array(rng.integers(0, 120, n), mask=rng.random(n) < 0.03),
                f"{prefix}_s": [f"s{v}" for v in rng.integers(0, 5, n)],
                f"{prefix}_f": np.where(rng.random(n) < 0.02, -0.0, rng.integers(0, 60, n) * 1.0),
                f"{prefix}_v": rng.random(n),
            }
        )

    return {"a": side("a", rows), "b": side("b", 0 if empty_right else rows // 2)}


def _run(duck, query: str, tables: dict[str, pa.Table]) -> tuple[pa.Table, str]:
    session = bt.Session()
    for name, table in tables.items():
        session.register(name, table)
        duck.register(name, table)
    return session.sql(query).collect(), session.sql(query).explain()


_ACCEPTED = {
    "col_key_plus_offset": (
        "SELECT a_k, a_w, b_w, a_v, b_v FROM a, b WHERE a_k = b_k AND a_w = b_w - 52"
    ),
    "offset_only": "SELECT a_w, b_w, a_v FROM a, b WHERE a_w = b_w + 1",
    "literal_first": "SELECT a_w, b_w, a_v FROM a, b WHERE a_k = b_k AND a_w = 3 + b_w",
    "swapped_sides": "SELECT a_w, b_w, a_s FROM a, b WHERE b_w - 7 = a_w AND b_s = a_s",
    "offset_both_sides": "SELECT a_w, b_w FROM a, b WHERE a_k = b_k AND a_w + 1 = b_w - 1",
    "aggregated": (
        "SELECT a_k, count(*) AS n, sum(a_v * b_v) AS s FROM a, b "
        "WHERE a_k = b_k AND a_w = b_w - 12 GROUP BY a_k"
    ),
}

_REFUSED = {
    "float_operand": "SELECT a_f, b_f, a_v FROM a, b WHERE a_k = b_k AND a_f = b_f + 1",
    "multiplication": "SELECT a_w, b_w FROM a, b WHERE a_k = b_k AND a_w = b_w * 2",
    "oversized_literal": "SELECT a_w, b_w FROM a, b WHERE a_k = b_k AND a_w = b_w + 9000000000",
}


@pytest.mark.parametrize("name", sorted(_ACCEPTED))
def test_an_expression_equality_becomes_a_key_and_matches_duckdb(duck, name):
    query = _ACCEPTED[name]
    result, explained = _run(duck, query, _tables())
    assert _HIDDEN in explained, explained  # positive control: the key was derived
    assert result.num_rows > 0
    assert_same(result, duck.sql(query))


@pytest.mark.parametrize("name", sorted(_REFUSED))
def test_a_refused_expression_stays_a_filter_and_matches_duckdb(duck, name):
    query = _REFUSED[name]
    result, explained = _run(duck, query, _tables())
    assert _HIDDEN not in explained, explained
    assert_same(result, duck.sql(query))


def test_an_empty_side_matches_duckdb(duck):
    query = _ACCEPTED["col_key_plus_offset"]
    result, _ = _run(duck, query, _tables(empty_right=True))
    assert result.num_rows == 0
    assert_same(result, duck.sql(query))

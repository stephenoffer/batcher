"""`IN` lists, `NOT`, and the string predicates translate on every backend.

These three shapes used to translate to `None` in all five translators, so
``col.is_in([...])``, ``~expr`` and ``col.str.starts_with(...)`` reached no source: a
warehouse extracted the whole relation over the network and a parquet scan pruned no row
group, while the engine's `Filter` discarded the rows afterwards.

The load-bearing case here is `NOT` over a *partly* translatable operand. Widening is what
makes a partial `AND` safe, and it is exactly what makes a negation unsafe — ``NOT`` of a
superset is a subset, so a filter that only meant to prune I/O would drop rows that match.
Every translator therefore declines rather than widening under a negation, and the tests
below pin that per backend.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.io.predicate import (
    to_mongo_filter,
    to_native_predicate,
    to_pyarrow_expression,
    to_sql_where,
)

pytestmark = pytest.mark.unit

# `x > 5` translates; `x > y` (column vs column) does not.
_PUSHABLE = bt.col("x") > 5
_UNPUSHABLE = bt.col("x") > bt.col("y")


def test_sql_pushes_an_in_list():
    assert to_sql_where(bt.col("a").is_in([1, 2, 3]).to_ir()) == "a IN (1, 2, 3)"


def test_sql_quotes_string_members_of_an_in_list():
    got = to_sql_where(bt.col("c").is_in(["US", "O'Brien"]).to_ir())
    assert got == "c IN ('US', 'O''Brien')"


def test_sql_declines_an_in_list_past_the_length_cap():
    from batcher.io.predicate.sql import _SQL_IN_MAX

    assert to_sql_where(bt.col("a").is_in(list(range(_SQL_IN_MAX))).to_ir()) is not None
    assert to_sql_where(bt.col("a").is_in(list(range(_SQL_IN_MAX + 1))).to_ir()) is None


def test_sql_pushes_a_negation():
    assert to_sql_where((~(bt.col("a") == 1)).to_ir()) == "NOT (a = 1)"


def test_sql_pushes_string_predicates_as_like():
    assert to_sql_where(bt.col("s").str.starts_with("US").to_ir()) == "s LIKE 'US%'"
    assert to_sql_where(bt.col("s").str.ends_with("US").to_ir()) == "s LIKE '%US'"
    assert to_sql_where(bt.col("s").str.contains("US").to_ir()) == "s LIKE '%US%'"


@pytest.mark.parametrize("pattern", ["50%", "a_b", "back\\slash", "bang!"])
def test_sql_declines_a_like_pattern_holding_a_wildcard(pattern):
    # Quoting these needs an ESCAPE clause BigQuery does not accept and whose backslash
    # form ClickHouse re-reads inside the string literal. Declining is always correct.
    assert to_sql_where(bt.col("s").str.starts_with(pattern).to_ir()) is None


def test_a_constant_false_predicate_prunes_everything():
    # `is_in([])` folds to a literal False in the builder; pushed, it reads nothing.
    assert to_sql_where(bt.col("a").is_in([]).to_ir()) == "1 = 0"


def test_sql_declines_a_negated_partial_and():
    # `NOT (a = 1 AND unpushable)` must not become `NOT (a = 1)`: that drops every row
    # with `a = 1`, including the ones the unpushable conjunct would have excluded.
    ir = (~((bt.col("a") == 1) & _UNPUSHABLE)).to_ir()
    assert to_sql_where(ir) is None


def test_sql_declines_a_negated_like():
    # LIKE's case sensitivity is the column's collation, so it may widen; negated, it
    # would then narrow. Only the negation declines — the bare form still pushes.
    assert to_sql_where(bt.col("s").str.starts_with("US").to_ir()) is not None
    assert to_sql_where((~bt.col("s").str.starts_with("US")).to_ir()) is None


def test_sql_still_widens_a_partial_and_outside_a_negation():
    assert to_sql_where(((bt.col("a") == 1) & _UNPUSHABLE).to_ir()) == "a = 1"


def test_pyarrow_pushes_the_new_shapes():
    assert to_pyarrow_expression(bt.col("a").is_in([1, 2]).to_ir()) is not None
    assert to_pyarrow_expression((~(bt.col("a") == 1)).to_ir()) is not None
    assert to_pyarrow_expression(bt.col("s").str.starts_with("US").to_ir()) is not None


def test_pyarrow_declines_a_negated_partial_and():
    ir = (~((bt.col("a") == 1) & _UNPUSHABLE)).to_ir()
    assert to_pyarrow_expression(ir) is None


def test_pyarrow_declines_an_in_list_the_column_cannot_compare_against():
    import pyarrow as pa

    schema = pa.schema([("d", pa.date32())])
    # A date column against string members raises from inside the scanner rather than
    # declining, which is why the members are type-checked here first.
    assert to_pyarrow_expression(bt.col("d").is_in(["2024-01-01"]).to_ir(), schema) is None


def test_native_expands_an_in_list_into_the_existing_vocabulary():
    # The reader's `Pred` enum has no set node, so `IN` becomes an OR of equalities
    # rather than a wire change across the FFI.
    got = to_native_predicate(bt.col("a").is_in([1, 2]).to_ir())
    assert got == {
        "node": "or",
        "left": {"node": "cmp", "col": "a", "op": "eq", "lit": 1},
        "right": {"node": "cmp", "col": "a", "op": "eq", "lit": 2},
    }


def test_native_carries_a_negation_to_the_leaves_by_de_morgan():
    got = to_native_predicate((~((bt.col("a") == 1) & (bt.col("b") < 2))).to_ir())
    assert got == {
        "node": "or",
        "left": {"node": "cmp", "col": "a", "op": "ne", "lit": 1},
        "right": {"node": "cmp", "col": "b", "op": "ge", "lit": 2},
    }


def test_native_negates_is_null():
    assert to_native_predicate((~bt.col("a").is_null()).to_ir()) == {
        "node": "is_null",
        "col": "a",
        "negated": True,
    }


def test_native_declines_an_in_list_past_the_tree_cap():
    from batcher.io.predicate.native import _NATIVE_IN_MAX

    assert to_native_predicate(bt.col("a").is_in(list(range(_NATIVE_IN_MAX + 1))).to_ir()) is None


def test_mongo_pushes_an_in_list_and_an_anchored_regex():
    assert to_mongo_filter(bt.col("a").is_in([1, 2]).to_ir()) == {"a": {"$in": [1, 2]}}
    assert to_mongo_filter(bt.col("s").str.starts_with("US").to_ir()) == {"s": {"$regex": "^US"}}


def test_mongo_escapes_regex_metacharacters_in_the_pattern():
    # Unescaped, `a.b` would match `axb` — more documents than the predicate names.
    got = to_mongo_filter(bt.col("s").str.starts_with("a.b").to_ir())
    assert got == {"s": {"$regex": "^a\\.b"}}


def test_mongo_declines_a_negated_partial_and():
    ir = (~((bt.col("a") == 1) & _UNPUSHABLE)).to_ir()
    assert to_mongo_filter(ir) is None


def test_iceberg_pushes_a_set_a_prefix_and_a_negation():
    ie = pytest.importorskip("pyiceberg.expressions")
    from batcher.io.predicate import to_iceberg_expression

    assert to_iceberg_expression(bt.col("a").is_in([1, 2]).to_ir()) == ie.In("a", [1, 2])
    assert to_iceberg_expression(bt.col("s").str.starts_with("US").to_ir()) == ie.StartsWith(
        "s", "US"
    )
    assert to_iceberg_expression((~(bt.col("a") == 1)).to_ir()) == ie.Not(ie.EqualTo("a", 1))


def test_iceberg_declines_a_suffix_it_cannot_prune_with():
    pytest.importorskip("pyiceberg.expressions")
    from batcher.io.predicate import to_iceberg_expression

    # Manifest lower/upper bounds can answer a prefix; nothing can answer a suffix.
    assert to_iceberg_expression(bt.col("s").str.ends_with("US").to_ir()) is None


def test_iceberg_declines_a_negated_partial_and_even_when_widening_is_allowed():
    pytest.importorskip("pyiceberg.expressions")
    from batcher.io.predicate import to_iceberg_expression

    ir = (~((bt.col("a") == 1) & _UNPUSHABLE)).to_ir()
    assert to_iceberg_expression(ir, allow_partial=True) is None

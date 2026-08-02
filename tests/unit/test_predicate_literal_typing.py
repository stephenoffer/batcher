"""A pushed filter must not hand arrow a comparison it has no kernel for.

Arrow compares within a type family. It has no ``greater_equal(date32, string)``, and the
dataset scanner does not decline such a filter — it raises ``ArrowNotImplementedError``
from inside whatever task built the scan. SQL writes that comparison constantly:
``WHERE EventDate >= '2013-07-01'`` against a ``date32`` column is the ClickBench spelling,
and it failed six of the 43 queries on the distributed path (q36-q39, q41, q42) while the
same query ran single-node, where the filter is the engine's and the engine coerces.

Two rules keep it honest, and both are about *rows*, not speed:

* A mismatched literal is declined rather than pushed. Pushdown is an optimization and the
  engine's own ``Filter`` re-checks every row, so declining costs pruning and never a row.
* An unpushable conjunct drops only itself. An ``AND`` term only ever widens what is read,
  so the rest of the filter still prunes — dropping the whole thing turned a six-predicate
  query into a full scan. An ``OR`` is all-or-nothing, because dropping a disjunct
  *narrows* the filter and would lose rows.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.io.predicate import to_pyarrow_expression

pytestmark = pytest.mark.unit

_SCHEMA = pa.schema(
    [
        ("d", pa.date32()),
        ("ts", pa.timestamp("us")),
        ("n", pa.int32()),
        ("s", pa.string()),
    ]
)


def _cmp(col: str, op: str, lit: dict) -> dict:
    return {
        "e": "binary",
        "op": op,
        "left": {"e": "col", "name": col},
        "right": {"e": "lit", "value": lit},
    }


def _and(left: dict, right: dict) -> dict:
    return {"e": "binary", "op": "and", "left": left, "right": right}


def test_a_string_literal_against_a_date_column_is_declined():
    assert to_pyarrow_expression(_cmp("d", "ge", {"str": "2013-07-01"}), _SCHEMA) is None


def test_a_string_literal_against_a_timestamp_column_is_declined():
    assert to_pyarrow_expression(_cmp("ts", "lt", {"str": "2013-07-01"}), _SCHEMA) is None


def test_a_typed_temporal_literal_still_pushes():
    """The declining must not cost the pruning that already worked — the TPC-H shipdate
    and orderdate filters are exactly this shape."""
    assert to_pyarrow_expression(_cmp("d", "ge", {"date": 15887}), _SCHEMA) is not None
    assert to_pyarrow_expression(_cmp("ts", "ge", {"timestamp": 1_000_000}), _SCHEMA) is not None


def test_numeric_and_string_columns_push_their_own_kinds():
    assert to_pyarrow_expression(_cmp("n", "eq", {"int": 62}), _SCHEMA) is not None
    assert to_pyarrow_expression(_cmp("n", "lt", {"float": 2.5}), _SCHEMA) is not None
    assert to_pyarrow_expression(_cmp("s", "ne", {"str": ""}), _SCHEMA) is not None
    assert to_pyarrow_expression(_cmp("n", "eq", {"str": "62"}), _SCHEMA) is None
    assert to_pyarrow_expression(_cmp("s", "eq", {"int": 1}), _SCHEMA) is None


def test_an_unpushable_conjunct_drops_only_itself():
    """The six-predicate ClickBench shape: one bad date term must not cost the other five."""
    expr = to_pyarrow_expression(
        _and(_cmp("d", "ge", {"str": "2013-07-01"}), _cmp("n", "eq", {"int": 62})), _SCHEMA
    )
    assert expr is not None
    assert "62" in str(expr) and "2013-07-01" not in str(expr)


def test_an_unpushable_disjunct_drops_the_whole_filter():
    """Dropping a branch of an OR narrows the filter, which loses rows — never do it."""
    ir = {
        "e": "binary",
        "op": "or",
        "left": _cmp("d", "ge", {"str": "2013-07-01"}),
        "right": _cmp("n", "eq", {"int": 62}),
    }
    assert to_pyarrow_expression(ir, _SCHEMA) is None


def test_without_a_schema_nothing_changes():
    """Every caller that has never passed a schema keeps exactly its previous behavior."""
    expr = to_pyarrow_expression(_cmp("d", "ge", {"str": "2013-07-01"}))
    assert expr is not None and "2013-07-01" in str(expr)

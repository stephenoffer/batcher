"""A refused SQL aggregate must name the spelling that works, not just say no.

`first(x)` and `last(x)` are ordinary DuckDB, and Batcher deliberately does not carry them:
DuckDB defines them as the first/last row in an *unspecified* order, so two runs may disagree
— and a morsel-parallel plan that may span nodes makes that far more visible than a
single-threaded scan does. `Expr.first(order_by=...)` requires the ordering that makes the
answer mean something.

That is a design decision, so the error has to carry it. A bare "unsupported aggregate:
first" leaves a migrant guessing whether it is missing or refused, and guessing wrong costs
them a rewrite; naming `any_value` and `arg_min`/`arg_max` answers both questions at once.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.differential


def _table():
    return pa.table(
        {
            "g": pa.array(["a", "b", "a"]),
            "x": pa.array([1.0, 2.0, 3.0]),
        }
    )


@pytest.mark.parametrize("fname", ["first", "last"])
def test_a_scan_order_aggregate_is_refused_with_the_alternative(fname):
    with pytest.raises(NotImplementedError) as exc:
        bt.sql(f"SELECT {fname}(g) AS r FROM t", t=_table()).collect()
    message = str(exc.value)
    assert "any_value" in message, message
    assert "arg_min" in message or "arg_max" in message, message
    assert "not defined" in message, "the message must say *why*, not only what to type"


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT any_value(g) AS r FROM t GROUP BY g ORDER BY r", ["a", "b"]),
        ("SELECT arg_max(g, x) AS r FROM t", ["a"]),
        ("SELECT arg_min(g, x) AS r FROM t", ["a"]),
    ],
    ids=["any_value", "arg_max", "arg_min"],
)
def test_every_alternative_the_message_names_actually_works(duck, sql, expected):
    """The guard that keeps the advice honest: run what the message tells the user to type."""
    table = _table()
    duck.register("t", table)
    got = bt.sql(sql, t=table).to_pydict()["r"]
    assert got == expected
    assert got == [r[0] for r in duck.sql(sql).fetchall()]

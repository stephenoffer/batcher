"""`chain_predicate` — which rows a single-scan GPU chain can skip, and when it must read them all.

The GPU fan-out narrowed its read to the columns a chain names but never to the rows it keeps,
so a selective query moved its whole relation onto the devices and filtered there. TPC-H q6 is
the extreme case in the suite: the CPU engine prunes from the Parquet footer and answers in
0.3 s, while the GPU read all 11.6 GB of its shard set and took 23.4 s for the same answer.

Pruning is the one place in this path where being wrong loses rows rather than time, so the
cases below pin the direction of caution rather than the speed. A predicate that cannot be
traced back to a source column must not be pushed at all; one that can must come back rebased
onto the names the *file* holds, because the chain's filter is written against a projection's
aliases and the footer knows nothing about those.
"""

from __future__ import annotations

import pytest

from batcher.core.gpu_plan.pruning import chain_predicate

pytestmark = pytest.mark.unit


def _col(name: str) -> dict:
    return {"e": "col", "name": name}


def _lit(value: int) -> dict:
    return {"e": "lit", "value": {"int": value}}


def _lt(name: str, value: int) -> dict:
    return {"e": "binary", "op": "lt", "left": _col(name), "right": _lit(value)}


def _filter(pred: dict) -> dict:
    return {"op": "filter", "predicate": pred}


def _rename(pairs: dict[str, str]) -> dict:
    """A projection that renames source columns, as every benchmark session's scan opens with."""
    return {"op": "project", "exprs": [{"alias": a, "expr": _col(s)} for a, s in pairs.items()]}


def _aggregate() -> dict:
    return {"op": "aggregate", "group_keys": [], "aggregates": []}


def test_no_filter_pushes_nothing() -> None:
    assert chain_predicate([_aggregate()]) is None
    assert chain_predicate([]) is None


def test_filter_directly_on_the_scan_is_pushed_unchanged() -> None:
    pred = _lt("l_quantity", 24)
    assert chain_predicate([_filter(pred)]) == pred


def test_filter_above_a_rename_is_rebased_onto_the_source_columns() -> None:
    """The whole point: the footer knows `column04`, the chain's filter says `l_quantity`."""
    ops = [_rename({"l_quantity": "column04"}), _filter(_lt("l_quantity", 24))]
    assert chain_predicate(ops) == _lt("column04", 24)


def test_a_column_the_projection_dropped_is_not_pushed() -> None:
    """Unresolvable means unpushable — pruning on a name the file lacks could drop live rows."""
    ops = [_rename({"kept": "column00"}), _filter(_lt("dropped", 3))]
    assert chain_predicate(ops) is None


def test_a_computed_column_is_not_traceable_and_is_not_pushed() -> None:
    """`a + b` aliased to `c` maps to no single source column, so a filter on `c` cannot prune."""
    computed = {
        "op": "project",
        "exprs": [
            {
                "alias": "c",
                "expr": {"e": "binary", "op": "add", "left": _col("a"), "right": _col("b")},
            }
        ],
    }
    assert chain_predicate([computed, _filter(_lt("c", 5))]) is None


def test_a_swapping_projection_resolves_both_names_against_the_way_in() -> None:
    """Both aliases resolve against the incoming mapping, not against each other's new meaning."""
    ops = [_rename({"a": "b", "b": "a"}), _filter(_lt("a", 1))]
    assert chain_predicate(ops) == _lt("b", 1)


def test_filters_are_conjoined_in_the_engines_own_binary_and() -> None:
    ops = [_filter(_lt("x", 1)), _filter(_lt("y", 2))]
    assert chain_predicate(ops) == {
        "e": "binary",
        "op": "and",
        "left": _lt("x", 1),
        "right": _lt("y", 2),
    }


def test_a_filter_above_an_aggregate_is_never_pushed() -> None:
    """It constrains groups, not rows: pushing it would delete inputs the aggregate needed."""
    ops = [_aggregate(), _filter(_lt("total", 100))]
    assert chain_predicate(ops) is None


def test_the_filter_below_an_aggregate_is_still_pushed() -> None:
    """Stopping at the aggregate must not throw away what sat underneath it."""
    ops = [_filter(_lt("x", 1)), _aggregate(), _filter(_lt("total", 100))]
    assert chain_predicate(ops) == _lt("x", 1)


def test_rebasing_leaves_the_original_chain_untouched() -> None:
    """The ops are a Ray task argument and are read again after this; rewriting one in place
    would change what the devices are asked to compute, not merely what they read."""
    ops = [_rename({"q": "column04"}), _filter(_lt("q", 24))]
    before = _lt("q", 24)
    assert chain_predicate(ops) == _lt("column04", 24)
    assert ops[1]["predicate"] == before

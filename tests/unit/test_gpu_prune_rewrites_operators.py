"""Narrowing what an operator *declares* and not what it *does* is a missing-column failure.

`pruning` is a rewrite, not a report, and its own docstring says so — for joins. A join names
the columns it emits, so narrowing an input's read without narrowing the join's output list
leaves the join asking for a column that is no longer there.

The same is true of every operator that declares what it emits, and it was applied to only one
of them. A **projection** above a join carrying an expression nothing upstream reads had that
expression correctly excluded from the want-set, the join's output correctly narrowed to match
— and then the projection evaluated the expression anyway, against a frame the column had just
been pruned out of. An **aggregate** whose reduction alias is unread has the same shape.

Measured on a six-T4 fleet at TPC-H sf10, it cost two queries their device: q7 died on
`column 'c_nationkey' absent from the GPU frame` and q17 on `'p_partkey'`, each after a full
round trip to a worker plus a second to the single-device retry — 1.9 s and 3.2 s against CPU
answers of 0.46 s and 0.26 s.

A chain over a single scan cannot reach it, which is why it went unnoticed for so long: nothing
sits above the chain to narrow it, so its top operator is always asked for everything.
"""

from __future__ import annotations

import pytest

from batcher.core.gpu_plan.pruning import prune_tree

pytestmark = pytest.mark.unit


def _col(name: str) -> dict:
    return {"e": "col", "name": name}


def _join_tree(above: list[dict]) -> dict:
    """`above` over `L ⋈ R`, with the join declaring one output column from each side."""
    return {
        "kind": "join",
        "left": {"kind": "scan", "leaf": 0, "source_id": 0, "ops": []},
        "right": {"kind": "scan", "leaf": 1, "source_id": 1, "ops": []},
        "join": {
            "op": "hash_join",
            "join_type": "inner",
            "left_keys": ["k"],
            "right_keys": ["k"],
            "output": [
                {"side": "left", "name": "a", "alias": "a"},
                {"side": "right", "name": "b", "alias": "b"},
                {"side": "right", "name": "dead", "alias": "dead"},
            ],
        },
        "ops": above,
    }


def _sum_of(*names: str, alias: str) -> dict:
    """A keyless aggregate summing `names` — a node that answers with a concrete want-set."""
    return {
        "op": "aggregate",
        "group_keys": [],
        "aggregates": [{"func": "sum", "alias": f"{alias}_{n}", "input": _col(n)} for n in names],
    }


def _aliases(op: dict) -> list[str]:
    key = "exprs" if op["op"] == "project" else "aggregates"
    return [entry["alias"] for entry in op[key]]


def test_a_projection_loses_the_expressions_its_join_stopped_emitting():
    """The defect, stated directly: what the projection computes must be what the join emits."""
    project = {
        "op": "project",
        "exprs": [
            {"alias": "a", "expr": _col("a")},
            {"alias": "dead", "expr": _col("dead")},
        ],
    }
    # An aggregate above it, because that is what narrows: it replaces its input's columns, so
    # it answers with a concrete want-set whatever its own parent asked for. A `sort` or a
    # `filter` merely adds to what it was given, so under a root asking for everything they
    # pass "everything" down and nothing is narrowed at all.
    total = _sum_of("a", alias="total")
    pruned, projections = prune_tree(_join_tree([project, total]))

    emitted = {o["alias"] for o in pruned["join"]["output"]}
    computed = set(_aliases(pruned["ops"][0]))
    assert computed <= emitted, f"{computed - emitted} computed but no longer emitted"
    assert "dead" not in computed
    assert "dead" not in projections[1]


def test_a_kept_expression_survives():
    """Narrowing must not become deleting: what is read upstream stays."""
    project = {
        "op": "project",
        "exprs": [
            {"alias": "a", "expr": _col("a")},
            {"alias": "b", "expr": _col("b")},
        ],
    }
    pruned, _ = prune_tree(_join_tree([project, _sum_of("a", "b", alias="total")]))
    assert _aliases(pruned["ops"][0]) == ["a", "b"]


def test_an_aggregate_loses_the_reductions_nothing_reads():
    """An aggregate declares its output the same way a projection does, and needs the same rule."""
    aggregate = {
        "op": "aggregate",
        "group_keys": [{"expr": _col("a"), "alias": "a"}],
        "aggregates": [
            {"func": "sum", "alias": "kept", "input": _col("b")},
            {"func": "sum", "alias": "dead", "input": _col("dead")},
        ],
    }
    # A projection above, so the aggregate is asked for a concrete subset of its own output.
    above = {"op": "project", "exprs": [{"alias": "kept", "expr": _col("kept")}]}
    pruned, projections = prune_tree(_join_tree([aggregate, above]))

    emitted = {o["alias"] for o in pruned["join"]["output"]}
    assert _aliases(pruned["ops"][0]) == ["kept"]
    assert "dead" not in emitted
    assert "dead" not in projections[1]


def test_the_root_keeps_everything():
    """Nothing sits above the root, so it is asked for everything and nothing may be dropped."""
    project = {
        "op": "project",
        "exprs": [
            {"alias": "a", "expr": _col("a")},
            {"alias": "dead", "expr": _col("dead")},
        ],
    }
    pruned, _ = prune_tree(_join_tree([project]))
    assert _aliases(pruned["ops"][0]) == ["a", "dead"]


def test_an_operator_nothing_reads_keeps_its_declaration():
    """A `COUNT(*)` reads no column of its input, and an operator narrowed to no column at all
    emits no *rows* in both dataframe libraries — losing exactly the count being taken."""
    project = {
        "op": "project",
        "exprs": [
            {"alias": "a", "expr": _col("a")},
            {"alias": "b", "expr": _col("b")},
        ],
    }
    count = {
        "op": "aggregate",
        "group_keys": [],
        "aggregates": [{"func": "count_star", "alias": "n"}],
    }
    pruned, _ = prune_tree(_join_tree([project, count]))
    assert _aliases(pruned["ops"][0]) == ["a", "b"]


def test_pruning_is_idempotent():
    """A second pass over an already-narrowed tree must not narrow it further."""
    project = {
        "op": "project",
        "exprs": [
            {"alias": "a", "expr": _col("a")},
            {"alias": "dead", "expr": _col("dead")},
        ],
    }
    once, first = prune_tree(_join_tree([project, _sum_of("a", alias="total")]))
    twice, second = prune_tree(once)
    assert _aliases(twice["ops"][0]) == _aliases(once["ops"][0])
    assert second == first

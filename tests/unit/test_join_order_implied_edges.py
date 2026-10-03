"""Join-order regions see the equalities their key classes imply, not only the ones written.

TPC-H q9 writes `p_partkey = l_partkey` and `ps_partkey = l_partkey` but never
`p_partkey = ps_partkey`, so the join-order search never treated `part` and `partsupp` as
neighbours and never priced joining the few filtered parts to their `partsupp` rows first.
`order._with_implied_edges` closes each class of equal key columns with one edge per pair of
leaves it spans; an inner equi-join region makes that exact.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher import kyber
from batcher.kyber.rules.joins.order import _with_implied_edges

pytestmark = pytest.mark.unit


def test_a_class_spanning_three_leaves_gets_the_missing_edge():
    edges = [((0, "p_k"), (2, "l_k")), ((1, "ps_k"), (2, "l_k")), ((1, "ps_s"), (2, "l_s"))]
    out = _with_implied_edges(edges)
    assert out[: len(edges)] == edges
    added = out[len(edges) :]
    assert added == [((0, "p_k"), (1, "ps_k"))]
    # Idempotent: the closed edge set implies nothing further.
    assert _with_implied_edges(out) == out


def test_leaves_already_joined_get_no_extra_edge():
    edges = [((0, "a"), (1, "a")), ((1, "a"), (2, "a")), ((0, "a"), (2, "a"))]
    assert _with_implied_edges(edges) == edges


def _joins(node, out):
    if isinstance(node, dict):
        if node.get("op") == "hash_join":
            out.append(node)
        for value in node.values():
            _joins(value, out)
    elif isinstance(node, list):
        for value in node:
            _joins(value, out)
    return out


def test_the_filtered_dimension_joins_its_bridge_table_first():
    names = ["green" if i % 100 == 0 else "red" for i in range(1000)]
    s = bt.Session()
    s.register("part", bt.from_pydict({"p_k": list(range(1000)), "p_name": names}))
    ps = range(40_000)
    partsupp = {"ps_k": [i % 1000 for i in ps], "ps_s": [i % 40 for i in ps]}
    s.register("partsupp", bt.from_pydict(partsupp))
    li = range(400_000)
    lineitem = {"l_k": [i % 1000 for i in li], "l_s": [i % 40 for i in li]}
    s.register("lineitem", bt.from_pydict(lineitem))
    q = (
        "SELECT count(*) AS n FROM part, partsupp, lineitem WHERE p_name = 'green' "
        "AND ps_k = l_k AND p_k = l_k AND ps_s = l_s"
    )
    ds = s.sql(q)
    opt = kyber.optimize(ds._plan, sources=ds._sources)
    keyed = [sorted(j["left_keys"] + j["right_keys"]) for j in _joins(opt.ir, [])]
    # Positive: some join is keyed on exactly the implied pair. Before the implied edge the plan
    # joined `lineitem` to `part` first and `partsupp` last, and no join paired these two keys.
    assert ["p_k", "ps_k"] in keyed, keyed
    assert ds.collect().to_pydict() == {"n": [160_000]}

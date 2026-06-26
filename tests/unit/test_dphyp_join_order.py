"""DPhyp join enumeration must match the exhaustive DP oracle.

`_rebuild_dphyp` enumerates only connected subgraph/complement pairs to scale join
ordering past the exhaustive DP's leaf cap. Correctness is pinned by an oracle: for
any graph the exhaustive `_rebuild_dp` can solve, DPhyp must find the *same optimal
cost* (it explores the same join partitions, same min-on-left orientation). We check
that across many random connected join graphs, then that DPhyp still produces a
valid full-coverage plan for larger (star/chain) graphs the oracle can't reach.
"""

from __future__ import annotations

import random

import pyarrow as pa

from batcher.config import active_config
from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.rules.join_order import _rebuild_dp, _rebuild_dphyp
from batcher.kyber.stats.estimator import StatsEstimator
from batcher.plan.logical import Join, Project, Scan
from batcher.plan.schema import SchemaRef
from batcher.plan.source_stats import SourceStatistics


def _random_connected_graph(n: int, rng: random.Random) -> list[tuple[int, int]]:
    """A connected graph on `n` leaves: a random spanning tree plus a few extra edges."""
    edges = [(rng.randint(0, i - 1), i) for i in range(1, n)]  # spanning tree
    extra = rng.randint(0, n)  # a few cycles → real bushy choices
    for _ in range(extra):
        a, b = rng.sample(range(n), 2)
        if (a, b) not in edges and (b, a) not in edges:
            edges.append((a, b))
    return edges


def _build_inputs(n: int, rng: random.Random):
    """Leaves, ColRef edges, required output, and a costing context for an n-leaf join."""
    graph = _random_connected_graph(n, rng)
    incident: list[set[str]] = [set() for _ in range(n)]
    col_edges: list[tuple[tuple[int, str], tuple[int, str]]] = []
    ndv: dict[str, float] = {}
    for k, (i, j) in enumerate(graph):
        col = f"c{k}"
        incident[i].add(col)
        incident[j].add(col)
        col_edges.append(((i, col), (j, col)))
        ndv[col] = float(rng.choice([5, 20, 100, 500, 5000]))

    leaves: list[Scan] = []
    stats: list[SourceStatistics] = []
    for i in range(n):
        cols = sorted(incident[i]) or [f"x{i}"]
        leaves.append(Scan(i, SchemaRef(pa.schema([pa.field(c, pa.int64()) for c in cols]))))
        stats.append(SourceStatistics(row_count=rng.randint(10, 200_000)))

    first_col = sorted(incident[0])[0]
    required = [(first_col, (0, first_col))]
    est = StatsEstimator([None] * n, learned={"__column_ndv__": ndv}, source_stats=stats)
    ctx = OptimizerContext(config=active_config(), sources=[None] * n, hub=None, estimator=est)
    return leaves, col_edges, required, ctx


def _tree_cost(node, cost) -> float:
    """Sum each join node's cost over a rebuilt plan tree — the accumulated cost the DP
    minimizes (the final `Project` is free)."""
    if isinstance(node, Project):
        return _tree_cost(node.input, cost)
    if isinstance(node, Join):
        return cost.cost(node).total() + _tree_cost(node.left, cost) + _tree_cost(node.right, cost)
    return 0.0


def test_dphyp_matches_exhaustive_dp_on_random_graphs():
    rng = random.Random(20240624)
    for _ in range(120):
        n = rng.randint(3, 9)  # within the exhaustive oracle's reach (kept cheap)
        leaves, edges, required, ctx = _build_inputs(n, rng)
        oracle = _rebuild_dp(leaves, edges, required, ctx)
        dphyp = _rebuild_dphyp(leaves, edges, required, ctx)
        assert oracle is not None and dphyp is not None
        c_oracle = _tree_cost(oracle, ctx.costs())
        c_dphyp = _tree_cost(dphyp, ctx.costs())
        # Same global optimum (relative tolerance for float accumulation).
        assert abs(c_oracle - c_dphyp) <= 1e-6 * max(1.0, c_oracle), (n, c_oracle, c_dphyp)


def test_dphyp_covers_all_leaves_for_large_chain():
    # 14-leaf chain (leaf k ⋈ leaf k+1) — past the exhaustive DP's 12-leaf cap, but a
    # sparse graph DPhyp handles (connected subsets are contiguous ranges, O(n²)). It
    # must produce a valid full-coverage bushy plan covering every leaf exactly once.
    rng = random.Random(1)
    n = 14
    incident: list[set[str]] = [set() for _ in range(n)]
    col_edges = []
    ndv = {}
    for k in range(n - 1):  # chain edge between leaf k and leaf k+1
        col = f"c{k}"
        incident[k].add(col)
        incident[k + 1].add(col)
        col_edges.append(((k, col), (k + 1, col)))
        ndv[col] = float(rng.choice([10, 100, 1000]))
    leaves = [
        Scan(i, SchemaRef(pa.schema([pa.field(c, pa.int64()) for c in sorted(incident[i])])))
        for i in range(n)
    ]
    stats = [SourceStatistics(row_count=rng.randint(10, 100_000)) for _ in range(n)]
    required = [("c0", (0, "c0"))]
    est = StatsEstimator([None] * n, learned={"__column_ndv__": ndv}, source_stats=stats)
    ctx = OptimizerContext(config=active_config(), sources=[None] * n, hub=None, estimator=est)

    out = _rebuild_dphyp(leaves, col_edges, required, ctx)
    assert isinstance(out, Project)
    # Every leaf source must appear exactly once in the rebuilt join tree.
    seen: set[int] = set()
    _collect_scan_ids(out, seen)
    assert seen == set(range(n))


def test_dphyp_bails_to_greedy_on_dense_large_graph(monkeypatch):
    # A dense (near-complete) graph has ~3ⁿ connected subsets/pairs — DPhyp must bail
    # to greedy (return None) rather than blow the planning budget (small-query
    # mandate). A tiny budget makes the bail observable without a huge enumeration.
    from batcher.kyber.rules import join_order as jo

    monkeypatch.setattr(jo, "_MAX_DP_PAIRS", 50)
    n = 13
    incident: list[set[str]] = [set() for _ in range(n)]
    col_edges = []
    ndv = {}
    k = 0
    for i in range(n):
        for j in range(i + 1, n):  # complete graph
            col = f"c{k}"
            incident[i].add(col)
            incident[j].add(col)
            col_edges.append(((i, col), (j, col)))
            ndv[col] = 100.0
            k += 1
    leaves = [
        Scan(i, SchemaRef(pa.schema([pa.field(c, pa.int64()) for c in sorted(incident[i])])))
        for i in range(n)
    ]
    stats = [SourceStatistics(row_count=1000) for _ in range(n)]
    required = [("c0", (0, "c0"))]
    est = StatsEstimator([None] * n, learned={"__column_ndv__": ndv}, source_stats=stats)
    ctx = OptimizerContext(config=active_config(), sources=[None] * n, hub=None, estimator=est)

    assert _rebuild_dphyp(leaves, col_edges, required, ctx) is None


def _collect_scan_ids(node, out: set[int]) -> None:
    if isinstance(node, Scan):
        out.add(node.source_id)
        return
    for child in _node_children(node):
        _collect_scan_ids(child, out)


def _node_children(node):
    from batcher.plan.visitor import children

    return children(node)

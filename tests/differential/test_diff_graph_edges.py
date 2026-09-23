"""The graph algorithms at their edges, against NetworkX.

Four families of bug lived here, and each one returned a plausible answer rather than an
error, which is why each gets a test that pins a value an independent implementation computes:

* **An undirected graph walked one way.** `Graph.from_edges(directed=False)` stores each edge
  once, in whichever orientation the row was written. Every algorithm that follows edges read
  that table as if it were directed, so `bfs` from the end of an undirected chain reached
  nothing and `pagerank` ranked it as a one-way street. The components and triangle code
  symmetrized; the rest did not.
* **An iteration cap hit in silence.** An exact algorithm stopped at its cap returned a wrong
  answer with no signal: a 300-node chain came back as 201 components, a 301-node ring as 301
  strong components, and every shortest path was cut at 20 hops. The approximate ones dropped
  the `converged` flag their loop computed, and a divergent Katz series returned the direction
  of its blow-up, normalized to look like an answer.
* **A self-loop counted as a neighbour** by `transitivity`, so a triangle with a loop scored
  0.6 rather than 1.0.
* **A node with no neighbour pair dropped** from `clustering_coefficient` rather than scored
  0.0, so `average_clustering` averaged over the wrong denominator.

NetworkX is the oracle: it is the reference implementation of every algorithm here, and an
independent one. An undirected edge list below writes each edge once, in a deliberate mix of
orientations, because a row listing both directions of one undirected edge is two parallel
edges to Batcher and one edge to ``nx.Graph``.
"""

from __future__ import annotations

import warnings

import pytest

import batcher as bt
import batcher.graph as bg
from batcher import PlanError

pytestmark = pytest.mark.differential

nx = pytest.importorskip("networkx", reason="the independent graph oracle")

#: A connected, non-bipartite undirected graph (the triangle 2-3-5 keeps power iteration from
#: oscillating), each edge written once and several larger-endpoint first.
CONNECTED = [(2, 1), (2, 3), (4, 3), (2, 5), (5, 4), (6, 4), (3, 5)]
#: The same plus a second component, for the searches that must not cross into it.
SPLIT = [*CONNECTED, (8, 7)]
CONNECTED_NODES = sorted({n for edge in CONNECTED for n in edge})
SPLIT_NODES = sorted({n for edge in SPLIT for n in edge})
#: Weights for `CONNECTED`, chosen so the cheapest path is not the fewest hops.
WEIGHTS = [1.0, 4.0, 1.0, 1.0, 1.0, 2.0, 1.0]


def _graph(edges, *, directed=False, weights=None, nodes=None) -> bg.Graph:
    table = {"src": [a for a, _ in edges], "dst": [b for _, b in edges]}
    if weights is not None:
        table["w"] = weights
    g = bg.Graph.from_edges(
        bt.from_pydict(table), directed=directed, weight=None if weights is None else "w"
    )
    return g if nodes is None else g.with_nodes(bt.from_pydict({"node": nodes}))


def _nx(edges, *, directed=False, weights=None, nodes=()):
    g = nx.DiGraph() if directed else nx.Graph()
    g.add_nodes_from(nodes)
    for i, (a, b) in enumerate(edges):
        g.add_edge(a, b, weight=1.0 if weights is None else weights[i])
    return g


def _as_dict(ds, value: str, key: str = "node") -> dict:
    got = ds.to_pydict()
    return dict(zip(got[key], got[value], strict=True))


def _seeds(*nodes) -> bt.Dataset:
    return bt.from_pydict({"node": list(nodes)})


#: Batcher runs at its default `tolerance` (1e-6 per node per round), which keeps these tests
#: to tens of rounds; NetworkX runs to 1e-12. The gap is bounded by the default tolerance
#: over one minus the contraction rate, well inside this.
TOL = 1e-4


def _approx(got: dict, expected: dict, tol: float = TOL) -> None:
    assert set(got) == set(expected)
    assert got == pytest.approx(expected, abs=tol)


# --- undirected graphs are walked both ways -----------------------------------------------


def test_the_audit_repro_bfs_and_pagerank_on_an_undirected_chain():
    """The two-line repro: from node 3 of ``1 - 2 - 3``, both other nodes are reachable."""
    g = _graph([(1, 2), (2, 3)])
    assert _as_dict(bg.bfs(g, _seeds(3)), "depth") == {3: 0, 2: 1, 1: 2}
    got = _as_dict(bg.pagerank(g), "pagerank")
    _approx(got, nx.pagerank(_nx([(1, 2), (2, 3)]), tol=1e-12, max_iter=1000))
    assert [round(got[n], 3) for n in (1, 2, 3)] == [0.257, 0.486, 0.257]


def test_pagerank_matches_networkx_on_an_undirected_graph():
    got = _as_dict(bg.pagerank(_graph(SPLIT)), "pagerank")
    _approx(got, nx.pagerank(_nx(SPLIT), tol=1e-12, max_iter=1000))


def test_weighted_pagerank_matches_networkx_on_an_undirected_graph():
    g = _graph(CONNECTED, weights=WEIGHTS)
    got = _as_dict(bg.pagerank(g), "pagerank")
    _approx(got, nx.pagerank(_nx(CONNECTED, weights=WEIGHTS), tol=1e-12, max_iter=1000))


def test_personalized_pagerank_matches_networkx_on_an_undirected_graph():
    got = _as_dict(
        bg.personalized_pagerank(_graph(SPLIT), _seeds(1)),
        "pagerank",
    )
    expected = nx.pagerank(_nx(SPLIT), personalization={1: 1.0}, tol=1e-12, max_iter=1000)
    _approx(got, expected)


def test_katz_matches_networkx_on_an_undirected_graph():
    got = _as_dict(bg.katz_centrality(_graph(SPLIT)), "katz_centrality")
    _approx(got, nx.katz_centrality(_nx(SPLIT), alpha=0.1, beta=1.0, tol=1e-12, max_iter=1000))


def test_eigenvector_centrality_matches_networkx_on_an_undirected_graph():
    got = _as_dict(
        bg.eigenvector_centrality(_graph(CONNECTED)),
        "eigenvector_centrality",
    )
    _approx(got, nx.eigenvector_centrality(_nx(CONNECTED), max_iter=5000, tol=1e-12))


def test_hits_matches_networkx_on_an_undirected_graph():
    """On an undirected graph a hub and an authority are the same thing, as in NetworkX."""
    got = bg.hits(_graph(CONNECTED)).to_pydict()
    hubs, _ = nx.hits(_nx(CONNECTED), max_iter=5000, tol=1e-12)
    norm = sum(v * v for v in hubs.values()) ** 0.5
    expected = {k: v / norm for k, v in hubs.items()}
    _approx(dict(zip(got["node"], got["hub"], strict=True)), expected)
    _approx(dict(zip(got["node"], got["authority"], strict=True)), expected)


@pytest.mark.parametrize("source", [1, 4, 6])
def test_bfs_matches_networkx_on_an_undirected_graph(source):
    got = _as_dict(bg.bfs(_graph(SPLIT), _seeds(source)), "depth")
    assert got == nx.single_source_shortest_path_length(_nx(SPLIT), source)


def test_weighted_shortest_paths_match_networkx_on_an_undirected_graph():
    g = _graph(CONNECTED, weights=WEIGHTS)
    got = _as_dict(bg.shortest_path_lengths(g, _seeds(6)), "distance")
    expected = nx.single_source_dijkstra_path_length(_nx(CONNECTED, weights=WEIGHTS), 6)
    _approx(got, expected)


def test_k_hop_and_reachable_match_networkx_on_an_undirected_graph():
    g, ref = _graph(SPLIT), _nx(SPLIT)
    within_two = {n for n, d in nx.single_source_shortest_path_length(ref, 6).items() if 0 < d <= 2}
    assert set(bg.k_hop_neighbors(g, _seeds(6), 2).to_pydict()["node"]) == within_two
    assert set(bg.reachable_from(g, _seeds(7)).to_pydict()["node"]) == {7, 8}
    assert set(bg.reachable_from(g, _seeds(6)).to_pydict()["node"]) == set(CONNECTED_NODES)


def test_harmonic_centrality_over_every_source_matches_networkx_on_an_undirected_graph():
    """Every node as a source makes the estimate the exact centrality NetworkX computes."""
    got = _as_dict(
        bg.harmonic_centrality(_graph(SPLIT), _seeds(*SPLIT_NODES)), "harmonic_centrality"
    )
    _approx(got, nx.harmonic_centrality(_nx(SPLIT)))


def test_harmonic_centrality_matches_networkx_on_a_directed_graph():
    edges = [(1, 2), (2, 3), (3, 1), (3, 4), (5, 4)]
    nodes = sorted({n for e in edges for n in e})
    got = _as_dict(
        bg.harmonic_centrality(_graph(edges, directed=True), _seeds(*nodes)), "harmonic_centrality"
    )
    _approx(got, nx.harmonic_centrality(_nx(edges, directed=True)))


def test_betweenness_matches_networkx_on_both_kinds_of_graph():
    """Summed over every source; an undirected graph counts each pair from both ends."""
    edges = [(1, 2), (2, 3), (3, 4), (2, 5), (5, 4), (4, 6)]
    nodes = sorted({n for e in edges for n in e})
    directed = _as_dict(
        bg.betweenness_centrality(_graph(edges, directed=True), _seeds(*nodes)), "betweenness"
    )
    _approx(directed, nx.betweenness_centrality(_nx(edges, directed=True), normalized=False))
    undirected = _as_dict(bg.betweenness_centrality(_graph(edges), _seeds(*nodes)), "betweenness")
    halved = {k: v / 2 for k, v in undirected.items()}
    _approx(halved, nx.betweenness_centrality(_nx(edges), normalized=False))


def test_message_passing_sees_every_neighbour_on_an_undirected_graph():
    g = _graph([(1, 2), (3, 2)])
    feats = bt.from_pydict({"node": [1, 2, 3], "x": [10.0, 20.0, 30.0]})
    got = _as_dict(bg.aggregate_neighbors(g, feats, ["x"]), "x_mean")
    assert got == {1: 20.0, 2: 20.0, 3: 20.0}


def test_strong_components_of_an_undirected_graph_are_its_components():
    got = bg.strongly_connected_components(_graph(SPLIT)).to_pydict()
    groups: dict = {}
    for node, label in zip(got["node"], got["component"], strict=True):
        groups.setdefault(label, set()).add(node)
    assert sorted(map(sorted, groups.values())) == sorted(
        map(sorted, nx.connected_components(_nx(SPLIT)))
    )


# --- exact algorithms run to their fixpoint ------------------------------------------------


def _chain(n: int) -> list[tuple[int, int]]:
    return [(i, i + 1) for i in range(n)]


def test_connected_components_of_a_chain_past_the_old_cap_is_one_component():
    """The old default of 100 rounds split a 300-node chain into 201 components."""
    got = bg.connected_components(_graph(_chain(150), directed=True))
    assert got.count() == 151
    assert set(got.to_pydict()["component"]) == {0}


def test_a_ring_past_the_old_cap_is_one_strong_component():
    """The old default of 20 rounds reported a 301-node ring as 301 singletons."""
    ring = [*_chain(24), (24, 0)]
    got = bg.strongly_connected_components(_graph(ring, directed=True)).to_pydict()
    assert len(got["node"]) == 25
    assert set(got["component"]) == {24}


def test_strong_components_match_networkx_on_a_tangle():
    edges = [(1, 2), (2, 3), (3, 1), (3, 4), (4, 5), (5, 4), (5, 6), (7, 6), (6, 8), (8, 7), (9, 1)]
    got = bg.strongly_connected_components(_graph(edges, directed=True)).to_pydict()
    groups: dict = {}
    for node, label in zip(got["node"], got["component"], strict=True):
        groups.setdefault(label, set()).add(node)
        assert label == max(groups[label] | {label}), "labelled by the largest id"
    assert sorted(map(sorted, groups.values())) == sorted(
        map(sorted, nx.strongly_connected_components(_nx(edges, directed=True)))
    )


def test_shortest_paths_past_the_old_cap_reach_the_end_of_the_chain():
    """The old default of 20 relaxation rounds cut every path at 20 hops."""
    got = _as_dict(
        bg.shortest_path_lengths(_graph(_chain(25), directed=True), _seeds(0)), "distance"
    )
    assert got == {i: float(i) for i in range(26)}


def test_bfs_past_the_old_default_depth_reaches_the_end_of_the_chain():
    got = _as_dict(bg.bfs(_graph(_chain(14), directed=True), _seeds(0)), "depth")
    assert got == {i: i for i in range(15)}


#: A 12-node chain, the shape every exact algorithm needs one round per hop on, and a ring
#: of the same length for strong components, which settle a chain with ascending ids at once.
_CHAIN = [(i, i + 1) for i in range(11)]
_RING = [*_CHAIN, (11, 0)]


@pytest.mark.parametrize(
    ("edges", "call"),
    [
        (_CHAIN, lambda g: bg.connected_components(g, max_iterations=2)),
        (_RING, lambda g: bg.strongly_connected_components(g, max_iterations=3)),
        (_CHAIN, lambda g: bg.shortest_path_lengths(g, _seeds(0), max_iterations=3)),
        (_CHAIN, lambda g: bg.topological_order(g, max_iterations=3)),
        (_CHAIN, lambda g: bg.is_dag(g, max_iterations=3)),
    ],
    ids=["components", "strong_components", "shortest_paths", "topological_order", "is_dag"],
)
def test_an_exact_algorithm_refuses_to_return_a_truncated_answer(edges, call):
    """A cap the caller set that stops the fixpoint early is an error, not a wrong answer."""
    with pytest.raises(PlanError, match="fixpoint"):
        call(_graph(edges, directed=True))


def test_k_core_refuses_a_cap_shorter_than_its_cascade():
    """A chain peels two end nodes per round at k=2, so one round cannot finish it."""
    with pytest.raises(PlanError, match="fixpoint"):
        bg.k_core(_graph(_chain(12)), 2, max_iterations=1)
    assert bg.k_core(_graph(_chain(12)), 2).num_nodes() == 0


def test_a_cap_large_enough_is_not_an_error():
    g = _graph(_chain(5), directed=True)
    assert bg.connected_components(g, max_iterations=10).count_distinct("component") == 1
    assert bg.is_dag(g, max_iterations=10)


# --- approximate algorithms warn, and a divergent Katz raises -------------------------------


def test_pagerank_warns_when_the_cap_stops_it_before_it_converges():
    with pytest.warns(bg.ConvergenceWarning, match="pagerank"):
        bg.pagerank(_graph(CONNECTED), max_iterations=2)


def test_a_converged_run_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", bg.ConvergenceWarning)
        bg.pagerank(_graph(CONNECTED))


@pytest.mark.parametrize(
    "call",
    [
        lambda g: bg.personalized_pagerank(g, _seeds(1), max_iterations=2),
        lambda g: bg.eigenvector_centrality(g, max_iterations=2),
        lambda g: bg.hits(g, max_iterations=2),
        lambda g: bg.katz_centrality(g, max_iterations=2),
    ],
    ids=["personalized_pagerank", "eigenvector", "hits", "katz"],
)
def test_every_approximate_algorithm_warns_at_its_cap(call):
    with pytest.warns(bg.ConvergenceWarning):
        call(_graph(CONNECTED))


def test_label_propagation_warns_when_labels_keep_swapping():
    """Two nodes adopt each other's label every round and never settle."""
    with pytest.warns(bg.ConvergenceWarning, match="label_propagation"):
        bg.label_propagation(_graph([(1, 2)]), max_iterations=3)


def test_katz_raises_when_the_series_diverges():
    """A triangle's leading eigenvalue is 2, so attenuation 0.9 is far past 1/2."""
    with pytest.raises(PlanError, match="diverges"):
        bg.katz_centrality(_graph([(1, 2), (2, 3), (3, 1)]), attenuation=0.9, max_iterations=30)


def test_katz_merely_warns_when_it_is_converging_slowly():
    """A directed 3-cycle has eigenvalue 1; attenuation 0.9 converges, just not in 5 rounds."""
    g = _graph([(1, 2), (2, 3), (3, 1), (3, 4)], directed=True)
    with pytest.warns(bg.ConvergenceWarning, match="katz"):
        bg.katz_centrality(g, attenuation=0.9, max_iterations=5)


# --- self-loops and isolated nodes ---------------------------------------------------------

TRIANGLE_WITH_LOOP = [(1, 2), (2, 3), (3, 1), (1, 1)]


@pytest.mark.parametrize("directed", [True, False], ids=["directed", "undirected"])
def test_a_self_loop_is_not_a_neighbour_to_transitivity_or_clustering(directed):
    """networkx scores a triangle with a loop as fully transitive; it used to come out 0.6."""
    g = _graph(TRIANGLE_WITH_LOOP, directed=directed)
    ref = _nx(TRIANGLE_WITH_LOOP)
    assert bg.transitivity(g) == pytest.approx(nx.transitivity(ref)) == pytest.approx(1.0)
    _approx(_as_dict(bg.clustering_coefficient(g), "clustering"), nx.clustering(ref))
    assert _as_dict(bg.triangle_count(g), "triangles") == nx.triangles(ref)


def test_transitivity_matches_networkx_on_a_graph_with_open_triples():
    edges = [(1, 2), (2, 3), (3, 1), (3, 4), (4, 5), (5, 5), (4, 6)]
    assert bg.transitivity(_graph(edges)) == pytest.approx(nx.transitivity(_nx(edges)))


def test_isolated_and_loop_only_nodes_score_zero_clustering_as_in_networkx():
    """A triangle plus an isolated node averages 0.75, not 1.0."""
    edges = [(1, 2), (2, 3), (3, 1), (8, 8)]
    g = _graph(edges, nodes=[1, 2, 3, 8, 9])
    ref = _nx(edges, nodes=[9])
    _approx(_as_dict(bg.clustering_coefficient(g), "clustering"), nx.clustering(ref))
    assert bg.average_clustering(g) == pytest.approx(nx.average_clustering(ref))
    assert bg.average_clustering(g) == pytest.approx(0.6)
    assert _as_dict(bg.triangle_count(g), "triangles") == nx.triangles(ref)


def test_average_clustering_of_a_triangle_and_an_isolated_node_is_three_quarters():
    g = _graph([(1, 2), (2, 3), (3, 1)], nodes=[4])
    assert bg.average_clustering(g) == pytest.approx(0.75)

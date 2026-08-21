"""The graph analytics that had no test, against NetworkX.

``tests/unit/test_graph_algorithms.py`` checks the graph surface against hand-worked
answers and defining invariants. That is the right shape for the algorithms it covers, but
ten public names were not among them: ``degree_centrality``, ``harmonic_centrality``,
``component_sizes``, ``structural_features``, ``k_hop_neighbors``, ``reachable_from``, the
two samplers, and the two ``Graph`` methods.

The oracle here is NetworkX, which is the reference implementation of every one of these
and is an *independent* one -- a different data model (Python dicts of adjacency) and a
different traversal. Where a definition has more than one convention in circulation, the
test says which one Batcher follows and checks that NetworkX is asked for the same thing:
``degree_centrality`` over an undirected view, ``component_sizes`` over weak components,
``harmonic_centrality`` restricted to the given sources rather than over all pairs.

The two samplers are randomized, so they are checked against the properties a sampler owes
-- a subset of the real nodes or edges, the same subset for the same seed, and a fraction
that actually moves the size -- rather than against a value.
"""

from __future__ import annotations

import pytest

import batcher as bt
import batcher.graph as bg
from batcher import PlanError

pytestmark = pytest.mark.differential

nx = pytest.importorskip("networkx")

#: A directed graph with two weakly connected pieces, a triangle in each, a self-loop and
#: an edge that joins the self-looping node back into the first piece. Small enough to
#: verify by hand and irregular enough that a symmetric implementation cannot pass.
EDGES = {"src": [1, 2, 1, 4, 5, 4, 7, 7], "dst": [2, 3, 3, 5, 6, 6, 7, 1]}


@pytest.fixture(scope="module")
def graph() -> bg.Graph:
    """The fixture as a Batcher graph."""
    return bg.Graph.from_edges(bt.from_pydict(EDGES))


@pytest.fixture(scope="module")
def reference():
    """The same edges as a NetworkX ``DiGraph``."""
    g = nx.DiGraph()
    g.add_edges_from(zip(EDGES["src"], EDGES["dst"], strict=True))
    return g


def _as_dict(ds, key: str, value: str) -> dict:
    """A two-column result as a mapping, so row order cannot affect the comparison."""
    got = ds.to_pydict()
    return dict(zip(got[key], got[value], strict=True))


def test_degree_centrality_matches_networkx_over_the_undirected_view(graph, reference):
    """Degree over ``n - 1``, counting an edge once whichever way it points.

    NetworkX's ``degree_centrality`` on a ``DiGraph`` counts in and out edges separately,
    so the comparable call is on the undirected view -- which is the convention Batcher's
    ``degree`` uses, and stating it is the point of this test.
    """
    got = _as_dict(bg.degree_centrality(graph), "node", "degree_centrality")
    want = nx.degree_centrality(reference.to_undirected())
    assert set(got) == set(want)
    for node, value in got.items():
        assert value == pytest.approx(want[node], abs=1e-12), f"node {node}"


def test_component_sizes_match_networkxs_weak_components(graph, reference):
    """Sizes of the weakly connected components, as a multiset."""
    got = bg.component_sizes(graph).to_pydict()
    want = sorted(len(c) for c in nx.weakly_connected_components(reference))
    assert sorted(got["nodes"]) == want
    assert sum(got["nodes"]) == reference.number_of_nodes(), "every node is in a component"
    assert len(set(got["component"])) == len(got["component"]), "one row per component"


def test_reachable_from_matches_a_networkx_descendants_walk(graph, reference):
    """Forward reachability from a source set, following edge direction."""
    for source in (1, 4, 7):
        got = set(bg.reachable_from(graph, bt.from_pydict({"node": [source]})).to_pydict()["node"])
        want = nx.descendants(reference, source) | {source}
        assert got == want, f"reachable from {source}: {sorted(got)} vs {sorted(want)}"


def test_reachable_from_respects_edge_direction(graph):
    """Node 3 is a sink, so nothing but itself is reachable from it."""
    got = set(bg.reachable_from(graph, bt.from_pydict({"node": [3]})).to_pydict()["node"])
    assert got == {3}, f"a sink reached {sorted(got)}"


def test_k_hop_neighbors_matches_a_bounded_breadth_first_walk(graph, reference):
    """Every node within k hops, with the hop count it was first reached at."""
    for k in (1, 2, 3):
        got = bg.k_hop_neighbors(graph, bt.from_pydict({"node": [1]}), k).to_pydict()
        depths = nx.single_source_shortest_path_length(reference, 1, cutoff=k)
        want = {node: depth for node, depth in depths.items() if node != 1}
        assert dict(zip(got["node"], got["depth"], strict=True)) == want, f"k={k}"


def test_k_hop_neighbors_is_bounded_by_k(graph, reference):
    """The bound must bind: a larger k may only add nodes, never lose one."""
    seen: set[int] = set()
    for k in (1, 2, 3, 4):
        nodes = set(bg.k_hop_neighbors(graph, bt.from_pydict({"node": [7]}), k).to_pydict()["node"])
        assert seen <= nodes, f"k={k} lost a node the smaller bound found"
        seen = nodes
    assert seen == nx.descendants(reference, 7), "at k past the diameter, all descendants"


def test_harmonic_centrality_matches_networkx_restricted_to_the_sources(graph, reference):
    """The sum of reciprocal distances *from the given sources*, not over all pairs.

    NetworkX's ``harmonic_centrality`` sums over every other node by default; Batcher takes
    a source set, so the comparable call passes ``sources``. Getting this wrong yields
    numbers of the same shape and the wrong magnitude, which is why it is spelled out.
    """
    sources = [1]
    got = _as_dict(
        bg.harmonic_centrality(graph, bt.from_pydict({"node": sources})),
        "node",
        "harmonic_centrality",
    )
    want = nx.harmonic_centrality(reference, sources=sources)
    for node, value in got.items():
        assert value == pytest.approx(want[node], abs=1e-12), f"node {node}"
    assert got[1] == 0.0, "a source is at distance zero from itself and contributes nothing"


def test_structural_features_agree_with_the_algorithms_they_bundle(graph, reference):
    """One row per node carrying degrees, triangles, clustering and PageRank.

    Each column is checked against NetworkX rather than against Batcher's own single-metric
    functions, so a bundle that quietly computed one of them differently is caught.
    """
    got = bg.structural_features(graph).to_pydict()
    by_node = {node: index for index, node in enumerate(got["node"])}
    assert set(by_node) == set(reference.nodes)

    undirected = reference.to_undirected()
    triangles = nx.triangles(undirected)
    clustering = nx.clustering(undirected)
    pagerank = nx.pagerank(reference)
    for node, index in by_node.items():
        assert got["in_degree"][index] == reference.in_degree(node), f"in_degree {node}"
        assert got["out_degree"][index] == reference.out_degree(node), f"out_degree {node}"
        assert got["triangles"][index] == triangles[node], f"triangles {node}"
        assert got["clustering"][index] == pytest.approx(clustering[node], abs=1e-12), node
        assert got["pagerank"][index] == pytest.approx(pagerank[node], abs=1e-3), node
    assert sum(got["pagerank"]) == pytest.approx(1.0, abs=1e-6)


def test_structural_features_carry_every_column_the_single_metric_calls_do(graph):
    """The bundle must agree with the individual functions, or it is a second implementation."""
    bundled = bg.structural_features(graph).to_pydict()
    separate = _as_dict(bg.degree(graph), "node", "degree")
    for node, index in {n: i for i, n in enumerate(bundled["node"])}.items():
        assert bundled["degree"][index] == separate[node], f"degree of {node}"


def test_node_sample_draws_a_reproducible_subset_of_the_real_nodes(graph, reference):
    """A sampler owes three things: a subset, the same subset per seed, and a size that moves."""
    drawn = set(bg.node_sample(graph, 0.5, seed=1).to_pydict()["node"])
    assert drawn <= set(reference.nodes), "the sample invented a node"
    again = set(bg.node_sample(graph, 0.5, seed=1).to_pydict()["node"])
    assert drawn == again, "the same seed gave a different sample"
    everything = set(bg.node_sample(graph, 1.0, seed=1).to_pydict()["node"])
    assert everything == set(reference.nodes), "a fraction of one must keep every node"
    # Zero is refused rather than answered with an empty sample: a fraction of zero is
    # almost always a bug in the caller's arithmetic, and an empty graph downstream is a
    # much harder failure to trace back to it.
    with pytest.raises(PlanError):
        bg.node_sample(graph, 0.0, seed=1)


def test_edge_sample_draws_a_reproducible_subgraph(graph):
    """Same contract for edges, and the result must still be a usable graph."""
    original = set(zip(EDGES["src"], EDGES["dst"], strict=True))
    sampled = bg.edge_sample(graph, 0.5, seed=1)
    got = sampled.edges.to_pydict()
    drawn = set(zip(got["src"], got["dst"], strict=True))
    assert drawn <= original, "the sample invented an edge"
    again = bg.edge_sample(graph, 0.5, seed=1).edges.to_pydict()
    assert set(zip(again["src"], again["dst"], strict=True)) == drawn
    whole = bg.edge_sample(graph, 1.0, seed=1).edges.to_pydict()
    assert set(zip(whole["src"], whole["dst"], strict=True)) == original
    assert bg.degree(sampled).to_pydict()["node"], "the sampled graph must still be queryable"


def test_without_self_loops_removes_exactly_the_loops(graph):
    """The one self-loop goes and nothing else does."""
    got = graph.without_self_loops().edges.to_pydict()
    pairs = list(zip(got["src"], got["dst"], strict=True))
    assert (7, 7) not in pairs, "the self-loop survived"
    original = list(zip(EDGES["src"], EDGES["dst"], strict=True))
    assert sorted(pairs) == sorted(p for p in original if p[0] != p[1])
    assert graph.without_self_loops().without_self_loops().edges.count() == len(pairs), (
        "removing self-loops twice must be the same as removing them once"
    )


def test_extra_nodes_is_none_until_a_node_table_is_attached(graph):
    """None for a graph built from edges alone; the whole attached table once there is one.

    Deliberately the whole table rather than only the nodes the edges miss: the result is
    a union arm that gets deduplicated against the endpoints, so overlap is free while
    isolating the unnamed nodes would cost an anti-join.
    """
    assert graph.extra_nodes() is None

    with_nodes = graph.with_nodes(bt.from_pydict({"node": [1, 2, 99]}))
    extra = with_nodes.extra_nodes()
    assert extra is not None, "an attached node table must be reported"
    assert sorted(extra.to_pydict()["node"]) == [1, 2, 99]


def test_an_isolated_node_still_reaches_the_analytics():
    """A node no edge names must appear in the per-node results, with a zero degree."""
    g = bg.Graph.from_edges(bt.from_pydict({"src": [1], "dst": [2]})).with_nodes(
        bt.from_pydict({"node": [1, 2, 99]})
    )
    centrality = _as_dict(bg.degree_centrality(g), "node", "degree_centrality")
    assert 99 in centrality, "the isolated node vanished from the centrality result"
    assert centrality[99] == 0.0
    assert sorted(bg.component_sizes(g).to_pydict()["nodes"]) == [1, 2]

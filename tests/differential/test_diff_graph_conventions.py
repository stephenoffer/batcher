"""Graph algorithms against networkx where a convention or a convergence bound is at stake.

networkx is the oracle here, standing in for DuckDB, which has no graph algorithms to compare
against. networkx computes the reference values on the same edge list. The rest are either
hand-computed on graphs small enough to check by eye, such as a two-node path and a 1-2-3-4
chain, or invariants such as a single component and the result not depending on batch size.

Each case here pins a behaviour that used to be a documented limitation: betweenness had
to be halved by hand and could not be normalized, modularity counted a self-loop
differently from networkx with no way to ask for networkx's rule, shortest paths refused
every negative weight, connected components took a round count that tracked how the node
ids happened to be laid out, and `diameter_estimate` searched all its sources as one
frontier, so adding a source could *lower* it.
"""

from __future__ import annotations

import random

import pytest

import batcher as bt
import batcher.graph as bg
import batcher.graph.components as components_module
from batcher import PlanError

nx = pytest.importorskip("networkx")
community = pytest.importorskip("networkx.algorithms.community")

pytestmark = pytest.mark.differential


def _random_edges(nodes: int, edges: int, seed: int) -> list[tuple[int, int]]:
    graph = nx.gnm_random_graph(nodes, edges, seed=seed)
    return list(graph.edges())


def _graph(edges: list[tuple[int, int]], *, directed: bool, nodes: int) -> bg.Graph:
    table = bt.from_pydict({"src": [a for a, _ in edges], "dst": [b for _, b in edges]})
    g = bg.Graph.from_edges(table, directed=directed)
    return g.with_nodes(bt.from_pydict({"node": list(range(nodes))}))


def _as_dict(ds: bt.Dataset, value: str) -> dict[object, float]:
    got = ds.to_pydict()
    return dict(zip(got["node"], got[value], strict=True))


# --- betweenness -----------------------------------------------------------------------


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize(("scale", "normalized"), [("networkx", False), ("normalized", True)])
def test_betweenness_scales_match_networkx_exactly_over_every_source(directed, scale, normalized):
    edges = _random_edges(30, 70, seed=3)
    reference = nx.DiGraph(edges) if directed else nx.Graph(edges)
    reference.add_nodes_from(range(30))
    g = _graph(edges, directed=directed, nodes=30)
    got = _as_dict(
        bg.betweenness_centrality(g, g.nodes().select("node"), scale=scale, source_batch_size=7),
        "betweenness",
    )
    want = nx.betweenness_centrality(reference, normalized=normalized)
    assert set(got) == set(want)
    for n, value in want.items():
        assert got[n] == pytest.approx(value, abs=1e-12)


@pytest.mark.parametrize("directed", [False, True])
def test_betweenness_extrapolates_a_sample_the_way_networkx_does(directed):
    edges = _random_edges(30, 70, seed=4)
    reference = nx.DiGraph(edges) if directed else nx.Graph(edges)
    reference.add_nodes_from(range(30))
    # networkx draws its k sources with `random.Random(seed).sample(list(G), k)`.
    sample = random.Random(9).sample(list(reference.nodes()), 8)
    g = _graph(edges, directed=directed, nodes=30)
    got = _as_dict(
        bg.betweenness_centrality(g, bt.from_pydict({"node": sample}), scale="normalized"),
        "betweenness",
    )
    want = nx.betweenness_centrality(reference, k=8, seed=9, normalized=True)
    for n, value in want.items():
        assert got[n] == pytest.approx(value, abs=1e-12)


def test_betweenness_batch_size_does_not_change_the_answer():
    edges = _random_edges(25, 60, seed=5)
    g = _graph(edges, directed=False, nodes=25)
    every = g.nodes().select("node")
    one = _as_dict(bg.betweenness_centrality(g, every, source_batch_size=1), "betweenness")
    many = _as_dict(bg.betweenness_centrality(g, every, source_batch_size=100), "betweenness")
    assert one.keys() == many.keys()
    for n, value in one.items():
        assert many[n] == pytest.approx(value, abs=1e-9)
    # And the default raw sum is still the both-ends count networkx halves.
    halved = nx.betweenness_centrality(nx.Graph(edges), normalized=False)
    assert any(v > 0 for v in halved.values())
    for n, value in halved.items():
        assert one[n] == pytest.approx(2 * value, abs=1e-9)


def test_betweenness_refuses_an_unknown_scale():
    g = _graph([(0, 1)], directed=True, nodes=2)
    with pytest.raises(PlanError, match="scale"):
        bg.betweenness_centrality(g, g.nodes().select("node"), scale="textbook")


# --- modularity ------------------------------------------------------------------------


@pytest.mark.parametrize("symmetrize", [False, True])
def test_modularity_networkx_self_loop_rule_matches_networkx(symmetrize):
    rng = random.Random(2)
    raw = [(rng.randrange(20), rng.randrange(20)) for _ in range(60)] + [(3, 3), (7, 7)]
    reference = nx.Graph()
    for a, b in raw:
        weight = rng.choice([0.5, 1.0, 2.0])
        if reference.has_edge(a, b):
            reference[a][b]["weight"] += weight
        else:
            reference.add_edge(a, b, weight=weight)
    labels = {n: n % 3 for n in reference}
    parts = [{n for n in reference if labels[n] == c} for c in range(3)]
    rows = list(reference.edges(data="weight"))
    g = bg.Graph.from_edges(
        bt.from_pydict(
            {
                "src": [a for a, _, _ in rows],
                "dst": [b for _, b, _ in rows],
                "w": [w for *_, w in rows],
            }
        ),
        weight="w",
        directed=False,
    )
    if symmetrize:
        g = g.to_undirected()
    table = bt.from_pydict({"node": list(labels), "community": list(labels.values())})
    want = community.modularity(reference, parts, weight="weight")
    assert bg.modularity(g, table, self_loops="networkx") == pytest.approx(want, abs=1e-12)
    # The default convention is a different number on a graph with loops.
    assert bg.modularity(g, table) != pytest.approx(want, abs=1e-6)


# --- shortest paths with negative weights -----------------------------------------------


def test_negative_weights_match_networkx_bellman_ford():
    rng = random.Random(7)
    reference = nx.DiGraph()
    # A DAG with mixed-sign weights has no negative cycle by construction.
    for _ in range(80):
        a, b = sorted(rng.sample(range(25), 2))
        reference.add_edge(a, b, weight=rng.choice([-3.0, -1.0, 0.5, 2.0, 4.0]))
    rows = list(reference.edges(data="weight"))
    g = bg.Graph.from_edges(
        bt.from_pydict(
            {
                "src": [a for a, _, _ in rows],
                "dst": [b for _, b, _ in rows],
                "w": [w for *_, w in rows],
            }
        ),
        weight="w",
    )
    source = min(reference.nodes())
    got = _as_dict(bg.shortest_path_lengths(g, bt.from_pydict({"node": [source]})), "distance")
    want = nx.single_source_bellman_ford_path_length(reference, source)
    assert any(v < 0 for v in want.values())
    assert got == pytest.approx(want)


def test_a_negative_cycle_unreachable_from_the_source_does_not_block_it():
    e = bt.from_pydict({"src": [1, 5, 6], "dst": [2, 6, 5], "w": [-1.0, -1.0, -1.0]})
    g = bg.Graph.from_edges(e, weight="w")
    got = bg.shortest_path_lengths(g, bt.from_pydict({"node": [1]})).sort("node")
    assert got.to_pydict() == {"node": [1, 2], "distance": [0.0, -1.0]}


# --- connected components ---------------------------------------------------------------


def _rounds(ids: list[int], monkeypatch: pytest.MonkeyPatch) -> tuple[int, int]:
    calls = [0]
    real = components_module.iterate

    def counting(initial, step, **kwargs):
        def counted(state):
            calls[0] += 1
            return step(state)

        return real(initial, counted, **kwargs)

    monkeypatch.setattr(components_module, "iterate", counting)
    g = bg.Graph.from_edges(bt.from_pydict({"src": ids[:-1], "dst": ids[1:]}))
    components = bg.connected_components(g).count_distinct("component")
    return calls[0], components


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_components_settle_a_shuffled_path_in_logarithmic_rounds(seed, monkeypatch):
    ids = list(range(300))
    random.Random(seed).shuffle(ids)
    rounds, components = _rounds(ids, monkeypatch)
    assert components == 1
    # Before hooking this took 79 to 134 rounds; log2(300) is about 8.
    assert rounds <= 25, rounds


def test_components_match_networkx_on_a_random_forest_of_pieces():
    edges = _random_edges(200, 150, seed=11)
    g = _graph(edges, directed=False, nodes=200)
    got = _as_dict(bg.connected_components(g), "component")
    reference = nx.Graph(edges)
    reference.add_nodes_from(range(200))
    want = {n: min(part) for part in nx.connected_components(reference) for n in part}
    assert got == want


# --- source-sampled estimates -----------------------------------------------------------


def test_diameter_estimate_never_falls_when_a_source_is_added():
    g = bg.Graph.from_edges(bt.from_pydict({"src": [1, 2, 3], "dst": [2, 3, 4]}), directed=False)
    one = bg.diameter_estimate(g, bt.from_pydict({"node": [1]}))
    two = bg.diameter_estimate(g, bt.from_pydict({"node": [1, 4]}))
    assert (one, two) == (3, 3)


def test_diameter_estimate_is_the_largest_source_eccentricity():
    edges = _random_edges(40, 60, seed=12)
    reference = nx.Graph(edges)
    sources = sorted(reference.nodes())[:6]
    want = max(max(nx.single_source_shortest_path_length(reference, s).values()) for s in sources)
    g = _graph(edges, directed=False, nodes=40)
    got = bg.diameter_estimate(g, bt.from_pydict({"node": sources}), source_batch_size=4)
    assert got == want


def test_harmonic_centrality_batch_size_does_not_change_the_answer():
    edges = _random_edges(30, 50, seed=13)
    g = _graph(edges, directed=False, nodes=30)
    sources = bt.from_pydict({"node": list(range(0, 30, 3))})
    one = _as_dict(bg.harmonic_centrality(g, sources, source_batch_size=1), "harmonic_centrality")
    all_ = _as_dict(bg.harmonic_centrality(g, sources), "harmonic_centrality")
    assert one == pytest.approx(all_)

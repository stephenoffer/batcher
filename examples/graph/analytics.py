"""Graph analytics end to end: diagnose, rank, cluster, measure.

Builds one small social graph and runs every analytic over it, in the order you would
actually run them: cheap diagnostics first to learn what the graph is, then the expensive
algorithms the diagnostics say are worth running.

    python examples/graph/analytics.py
"""

from __future__ import annotations

import batcher as bt
import batcher.graph as bg


def social_graph() -> bg.Graph:
    """Two tight clusters joined by one bridge, plus a pendant and an isolated node."""
    edges = bt.from_pydict(
        {
            "src": ["a", "b", "a", "d", "e", "d", "c", "a", "x"],
            "dst": ["b", "c", "c", "e", "f", "f", "d", "z", "x"],
            "weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0],
        }
    )
    return bg.Graph.from_edges(edges, weight="weight").with_nodes(
        bt.from_pydict({"node": ["a", "b", "c", "d", "e", "f", "x", "z", "lonely"]})
    )


def diagnose_before_spending(g: bg.Graph) -> None:
    """The cheap numbers that say which expensive algorithms are worth running."""
    print("--- one row of diagnostics ---")
    summary = bg.summarize(g).to_pydict()
    print(summary)
    # The fixture is stated in `social_graph`'s docstring: 9 nodes, 9 edges, one isolated.
    # A diagnostic that silently disagrees with its own input is worse than no diagnostic.
    assert summary["nodes"] == [9] and summary["edges"] == [9]
    assert summary["isolated"] == [1] and summary["directed"] == [True]
    assert summary["average_degree"] == [2.0], "2 * 9 edges / 9 nodes"

    print("--- degree, both directions and weighted ---")
    table = (
        bg.in_degree(g)
        .join(bg.out_degree(g), on="node", how="left")
        .join(bg.degree(g), on="node", how="left")
        .join(bg.weighted_degree(g), on="node", how="left")
        .sort("node")
    )
    degrees = table.to_pydict()
    print(degrees)
    # Degree must be the two directions summed, for every node, or one of the three is wrong.
    for node, deg, into, out in zip(
        degrees["node"], degrees["degree"], degrees["in_degree"], degrees["out_degree"], strict=True
    ):
        assert deg == into + out, f"{node}: degree {deg} != {into} in + {out} out"
    assert sum(degrees["in_degree"]) == sum(degrees["out_degree"]) == 9, "one per edge"
    # The c->d edge carries weight 3, so weighted degree exceeds plain degree exactly there.
    weighted = dict(zip(degrees["node"], degrees["weighted_degree"], strict=True))
    plain = dict(zip(degrees["node"], degrees["degree"], strict=True))
    assert {n for n in plain if weighted[n] > plain[n]} == {"c", "d"}

    print("--- the degree distribution is the shape of the graph ---")
    distribution = bg.degree_distribution(g).to_pydict()
    print(distribution)
    assert sum(distribution["nodes"]) == 9, "the distribution must account for every node"
    average = bg.average_degree(g)
    print("average degree:", round(average, 4))
    # The distribution and the average are two views of one quantity; derive one from the
    # other rather than printing both and hoping.
    total = sum(d * n for d, n in zip(distribution["degree"], distribution["nodes"], strict=True))
    assert abs(total / 9 - average) < 1e-12
    isolated = bg.isolated_nodes(g).to_pydict()["node"]
    print("isolated:", isolated)
    assert isolated == ["lonely"], "the fixture has exactly one isolated node"

    print("--- density, reciprocity, assortativity ---")
    print(
        {
            "density": round(bg.density(g), 4),
            "reciprocity": round(bg.reciprocity(g), 4),
            "assortativity": round(bg.assortativity(g), 4),
        }
    )


def find_the_pieces(g: bg.Graph) -> None:
    """Components first: nothing path-based means anything across them."""
    print("--- components ---")
    components = bg.connected_components(g).sort("node").to_pydict()
    print(components)
    sizes = bg.component_sizes(g).to_pydict()
    print(sizes)
    # Three pieces — the main cluster, the self-loop at x, and `lonely` — so the graph is
    # not connected, and the sizes must partition the node set exactly.
    assert sum(sizes["nodes"]) == 9 and sorted(sizes["nodes"], reverse=True) == [7, 1, 1]
    assert len(set(components["component"])) == len(sizes["component"]) == 3
    connected = bg.is_connected(g)
    print("connected as a whole:", connected)
    assert connected is False, "three components cannot be connected"

    biggest = bg.largest_component(g)
    print("largest component:", biggest.num_nodes(), "nodes,", biggest.num_edges(), "edges")

    print("--- k-cores peel the sparse fringe away ---")
    for k in (1, 2, 3):
        print(f"  {k}-core:", sorted(bg.k_core(g, k).nodes().to_pydict()["node"]))


def rank_the_nodes(g: bg.Graph) -> None:
    """Six definitions of important, over the same graph."""
    print("--- centrality ---")
    ranked = (
        bg.pagerank(g)
        .join(bg.degree_centrality(g), on="node", how="left")
        .join(bg.eigenvector_centrality(g), on="node", how="left")
        .join(bg.katz_centrality(g), on="node", how="left")
        .join(bg.hits(g), on="node", how="left")
        .sort("pagerank", descending=True)
    )
    out = ranked.to_pydict()
    for i, node in enumerate(out["node"]):
        print(
            f"  {node:>7}  pagerank={out['pagerank'][i]:.4f}"
            f"  degree={out['degree_centrality'][i]:.3f}"
            f"  katz={out['katz_centrality'][i]:.3f}"
            f"  hub={out['hub'][i]:.3f}  auth={out['authority'][i]:.3f}"
        )

    print("--- personalized: what is close to 'a' specifically ---")
    near = bg.personalized_pagerank(g, bt.from_pydict({"node": ["a"]}))
    top = near.sort("pagerank", descending=True).limit(4).to_pydict()
    print([(n, round(v, 4)) for n, v in zip(top["node"], top["pagerank"], strict=True)])


def measure_cohesion(g: bg.Graph) -> None:
    """Triangles, clustering, and whether the communities found are real."""
    print("--- triangles ---")
    found_triangles = bg.triangles(g).sort("a", "b", "c").to_pydict()
    print(found_triangles)
    per_node = bg.triangle_count(g).sort("node").to_pydict()
    print(per_node)
    # Every triangle contributes to exactly three nodes' counts, so the two views must
    # reconcile — the classic place an off-by-one in a triangle enumerator hides.
    assert sum(per_node["triangles"]) == 3 * len(found_triangles["a"])
    for a, b, c in zip(
        found_triangles["a"], found_triangles["b"], found_triangles["c"], strict=True
    ):
        assert len({a, b, c}) == 3, "a triangle needs three distinct nodes"

    print("--- clustering, locally and globally ---")
    cc = bg.clustering_coefficient(g).sort("node").to_pydict()
    print({n: round(v, 4) for n, v in zip(cc["node"], cc["clustering"], strict=True)})
    print("average clustering:", round(bg.average_clustering(g), 4))
    print("transitivity:", round(bg.transitivity(g), 4))

    print("--- communities, and whether they mean anything ---")
    found = bg.label_propagation(g)
    print(found.sort("node").to_pydict())
    partition_q = bg.modularity(g, found)
    print("modularity of the partition:", round(partition_q, 4))
    everyone = found.select("node", community=bt.lit(0))
    trivial_q = bg.modularity(g, everyone)
    print("modularity of one big community:", round(trivial_q, 4))
    # Putting everything in one community is the degenerate partition and its modularity is
    # 0 by definition — which is the number that makes the other one interpretable, and the
    # reason this section says "whether they mean anything".
    assert abs(trivial_q) < 1e-9, f"the one-community partition must score 0, got {trivial_q}"
    assert -1.0 <= partition_q <= 1.0


def measure_distance(g: bg.Graph) -> None:
    """Distances are from a source set, never all-pairs."""
    undirected = g.to_undirected()
    seeds = bt.from_pydict({"node": ["a"]})

    print("--- hop distance from 'a' ---")
    hops = bg.bfs(undirected, seeds).sort("depth", "node").to_pydict()
    print(hops)
    assert hops["node"][0] == "a" and hops["depth"][0] == 0, "the seed is at depth 0"
    assert hops["depth"] == sorted(hops["depth"]), "BFS visits in non-decreasing depth"
    two_hop = sorted(bg.k_hop_neighbors(undirected, seeds, 2).to_pydict()["node"])
    print("2-hop neighbourhood:", two_hop)
    reachable = sorted(bg.reachable_from(undirected, seeds).to_pydict()["node"])
    print("reachable at all:", reachable)
    # Each of these is a strictly wider view of the same traversal, so they have to nest.
    depth_of = dict(zip(hops["node"], hops["depth"], strict=True))
    assert set(two_hop) == {n for n, d in depth_of.items() if 0 < d <= 2}
    assert set(reachable) == set(depth_of)
    assert "lonely" not in reachable, "an isolated node is reachable from nothing"
    diameter = bg.diameter_estimate(undirected, seeds)
    print("diameter (lower bound from 'a'):", diameter)
    assert diameter == max(hops["depth"]), "the estimate from one source is its deepest hop"

    print("--- weighted distance respects the weights ---")
    weighted_paths = bg.shortest_path_lengths(g, seeds).sort("distance", "node").to_pydict()
    print(weighted_paths)
    costs = dict(zip(weighted_paths["node"], weighted_paths["distance"], strict=True))
    assert costs["a"] == 0.0
    # The section's claim, checked: `d` is two hops from `a` but the c->d edge costs 3, so
    # its weighted distance is 4 and not 2. An implementation ignoring `weight=` would
    # print 2 here and look perfectly reasonable.
    assert costs["d"] == 4.0, f"the weight-3 edge must be paid for, got {costs['d']}"

    print("--- harmonic centrality, estimated from two sources ---")
    est = bg.harmonic_centrality(undirected, bt.from_pydict({"node": ["a", "f"]}))
    top = est.sort("harmonic_centrality", descending=True).limit(3).to_pydict()
    print([(n, round(v, 4)) for n, v in zip(top["node"], top["harmonic_centrality"], strict=True)])


def predict_links(g: bg.Graph) -> None:
    """Which unconnected pairs look like they should be connected."""
    print("--- candidates, then five ways to score them ---")
    pairs = bg.candidate_pairs(g, max_degree=8).select("a", "b")
    scored = pairs
    for score in (
        bg.common_neighbors,
        bg.jaccard_similarity,
        bg.adamic_adar,
        bg.resource_allocation,
        bg.preferential_attachment,
    ):
        scored = score(g, scored)
    out = scored.sort("adamic_adar", descending=True).limit(5).to_pydict()
    for i in range(len(out["a"])):
        print(
            f"  {out['a'][i]}-{out['b'][i]}  common={out['common_neighbors'][i]}"
            f"  jaccard={out['jaccard'][i]:.3f}  aa={out['adamic_adar'][i]:.4f}"
            f"  ra={out['resource_allocation'][i]:.4f}"
            f"  pa={out['preferential_attachment'][i]}"
        )


def main() -> None:
    g = social_graph().cache()
    diagnose_before_spending(g)
    find_the_pieces(g)
    rank_the_nodes(g)
    measure_cohesion(g)
    measure_distance(g)
    predict_links(g)


if __name__ == "__main__":
    main()

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
    print(bg.summarize(g).to_pydict())

    print("--- degree, both directions and weighted ---")
    table = (
        bg.in_degree(g)
        .join(bg.out_degree(g), on="node", how="left")
        .join(bg.degree(g), on="node", how="left")
        .join(bg.weighted_degree(g), on="node", how="left")
        .sort("node")
    )
    print(table.to_pydict())

    print("--- the degree distribution is the shape of the graph ---")
    print(bg.degree_distribution(g).to_pydict())
    print("average degree:", round(bg.average_degree(g), 4))
    print("isolated:", bg.isolated_nodes(g).to_pydict()["node"])

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
    print(bg.connected_components(g).sort("node").to_pydict())
    print(bg.component_sizes(g).to_pydict())
    print("connected as a whole:", bg.is_connected(g))

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
    print(bg.triangles(g).sort("a", "b", "c").to_pydict())
    print(bg.triangle_count(g).sort("node").to_pydict())

    print("--- clustering, locally and globally ---")
    cc = bg.clustering_coefficient(g).sort("node").to_pydict()
    print({n: round(v, 4) for n, v in zip(cc["node"], cc["clustering"], strict=True)})
    print("average clustering:", round(bg.average_clustering(g), 4))
    print("transitivity:", round(bg.transitivity(g), 4))

    print("--- communities, and whether they mean anything ---")
    found = bg.label_propagation(g)
    print(found.sort("node").to_pydict())
    print("modularity of the partition:", round(bg.modularity(g, found), 4))
    everyone = found.select("node", community=bt.lit(0))
    print("modularity of one big community:", round(bg.modularity(g, everyone), 4))


def measure_distance(g: bg.Graph) -> None:
    """Distances are from a source set, never all-pairs."""
    undirected = g.to_undirected()
    seeds = bt.from_pydict({"node": ["a"]})

    print("--- hop distance from 'a' ---")
    print(bg.bfs(undirected, seeds).sort("depth", "node").to_pydict())
    print(
        "2-hop neighbourhood:", sorted(bg.k_hop_neighbors(undirected, seeds, 2).to_pydict()["node"])
    )
    print("reachable at all:", sorted(bg.reachable_from(undirected, seeds).to_pydict()["node"]))
    print("diameter (lower bound from 'a'):", bg.diameter_estimate(undirected, seeds))

    print("--- weighted distance respects the weights ---")
    print(bg.shortest_path_lengths(g, seeds).sort("distance", "node").to_pydict())

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

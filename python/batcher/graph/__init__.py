"""Graph analytics and graph-ML features over an edge table.

A graph here is a `Dataset` of edges, and every algorithm is a sequence of joins and
aggregations over it. That is the whole design: a join is a join, so PageRank over a
billion edges distributes across a Ray cluster and spills under memory pressure using
exactly the machinery any other query uses, with no second execution path to keep in
agreement and no graph-shaped index to build first.

Start with `Graph.from_edges` and `summarize`. The cheap diagnostics tell you which of
the expensive algorithms are worth running: a graph that is one giant component behaves
nothing like a thousand islands, and a degree distribution with a long tail will make
triangle counting cost far more than its average degree suggests.
"""

from __future__ import annotations

from batcher.graph._graph import Graph
from batcher.graph.build import (
    co_occurrence_graph,
    knn_graph,
    spatial_graph,
    threshold_graph,
)
from batcher.graph.centrality import (
    betweenness_centrality,
    degree_centrality,
    eigenvector_centrality,
    hits,
    katz_centrality,
    pagerank,
    personalized_pagerank,
)
from batcher.graph.community import (
    average_clustering,
    clustering_coefficient,
    label_propagation,
    modularity,
    transitivity,
    triangle_count,
    triangles,
)
from batcher.graph.components import (
    component_sizes,
    connected_components,
    is_connected,
    k_core,
    largest_component,
    strongly_connected_components,
)
from batcher.graph.degree import (
    average_degree,
    degree,
    degree_distribution,
    in_degree,
    isolated_nodes,
    out_degree,
    weighted_degree,
)
from batcher.graph.features import (
    aggregate_neighbors,
    propagate_features,
    structural_features,
)
from batcher.graph.sampling import (
    edge_sample,
    ego_network,
    neighbor_sample,
    node_sample,
    random_walks,
)
from batcher.graph.similarity import (
    adamic_adar,
    candidate_pairs,
    common_neighbors,
    jaccard_similarity,
    preferential_attachment,
    resource_allocation,
)
from batcher.graph.summary import assortativity, density, reciprocity, summarize
from batcher.graph.traversal import (
    bfs,
    diameter_estimate,
    harmonic_centrality,
    is_dag,
    k_hop_neighbors,
    nodes_in_cycles,
    reachable_from,
    shortest_path_lengths,
    topological_order,
)

__all__ = [
    "Graph",
    "adamic_adar",
    "aggregate_neighbors",
    "assortativity",
    "average_clustering",
    "average_degree",
    "betweenness_centrality",
    "bfs",
    "candidate_pairs",
    "clustering_coefficient",
    "co_occurrence_graph",
    "common_neighbors",
    "component_sizes",
    "connected_components",
    "degree",
    "degree_centrality",
    "degree_distribution",
    "density",
    "diameter_estimate",
    "edge_sample",
    "ego_network",
    "eigenvector_centrality",
    "harmonic_centrality",
    "hits",
    "in_degree",
    "is_connected",
    "is_dag",
    "isolated_nodes",
    "jaccard_similarity",
    "k_core",
    "k_hop_neighbors",
    "katz_centrality",
    "knn_graph",
    "label_propagation",
    "largest_component",
    "modularity",
    "neighbor_sample",
    "node_sample",
    "nodes_in_cycles",
    "out_degree",
    "pagerank",
    "personalized_pagerank",
    "preferential_attachment",
    "propagate_features",
    "random_walks",
    "reachable_from",
    "reciprocity",
    "resource_allocation",
    "shortest_path_lengths",
    "spatial_graph",
    "strongly_connected_components",
    "structural_features",
    "summarize",
    "threshold_graph",
    "topological_order",
    "transitivity",
    "triangle_count",
    "triangles",
    "weighted_degree",
]

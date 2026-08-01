# Graph reference

Graph analytics and graph-ML features over an edge table. Import from `batcher.graph`.

A graph here is a `Dataset` of edges, and every algorithm is a sequence of joins and
aggregations over it. A join is a join, so PageRank over a billion edges distributes
across a Ray cluster and spills under memory pressure using the machinery any other query
uses. There is no second execution path and no graph-shaped index to build first.

To learn these rather than look them up, start with {doc}`/user-guide/analyze/graphs`.

```{eval-rst}
.. currentmodule:: batcher.graph
```

## The graph handle

A graph is an edge table plus the conventions the algorithms read: which columns hold the endpoints, whether direction means anything, and where the weights are. `Graph.from_edges` normalizes an arbitrary table into that shape once.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Graph
```

## Degree

The cheapest thing you can ask a graph, and usually the first. One `group_by` over the edge list. Note that on a symmetrized graph `degree` counts each edge twice, so `out_degree` is the neighbour count there.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   degree
   in_degree
   out_degree
   weighted_degree
   degree_distribution
   average_degree
   isolated_nodes
```

## Centrality

Which nodes matter, by six different definitions of matter. `pagerank` is the default; `personalized_pagerank` is the recommendation and local-neighbourhood primitive; `hits` is the one that separates hubs from authorities.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   pagerank
   personalized_pagerank
   degree_centrality
   eigenvector_centrality
   katz_centrality
   hits
```

## Components and cores

What the separate pieces are, and what survives peeling the sparse parts away. Run `component_sizes` on any graph you did not build yourself: one giant component and a thousand islands need different treatment everywhere downstream.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   connected_components
   component_sizes
   largest_component
   is_connected
   k_core
```

## Communities and triangles

Triangles are the smallest structure that tells a real social graph from a random one with the same degrees. Triangle counting is a three-way join and is the most expensive operation here, driven by the highest-degree node rather than the average.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   triangles
   triangle_count
   clustering_coefficient
   average_clustering
   transitivity
   label_propagation
   modularity
```

## Distances

Single-source and multi-source, never all-pairs: an all-pairs matrix is quadratic in node count and does not fit. The summary statistics that normally come from all-pairs are computed from a sample of sources and named as estimates.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   bfs
   shortest_path_lengths
   k_hop_neighbors
   reachable_from
   harmonic_centrality
   diameter_estimate
```

## Link prediction

How alike two nodes are, judged by who they connect to. Scored for candidate pairs you supply, because scoring all pairs is quadratic. `candidate_pairs` is the standard generator, and its `max_degree` argument is the standard defence against a hub producing a trillion candidates on its own.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   candidate_pairs
   common_neighbors
   jaccard_similarity
   adamic_adar
   resource_allocation
   preferential_attachment
```

## Sampling

What graph ML actually runs on. `neighbor_sample` bounds a GNN layer's cost regardless of degree; `random_walks` turns a graph into sequences an embedding model can read. Every function is deterministic given a seed, because an embedding trained on walks you cannot regenerate is one you cannot debug.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   neighbor_sample
   random_walks
   node_sample
   edge_sample
   ego_network
```

## Graph-ML features

Turning a graph into a feature table. `aggregate_neighbors` is one round of message passing, the arithmetic core of every GNN; stacking it and handing the result to a gradient-boosted model is a strong baseline that trains in seconds.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   aggregate_neighbors
   propagate_features
   structural_features
```

## Graph-level summary

The cheap diagnostics to run before spending anything. `summarize` deliberately excludes triangles, components and centrality, which each cost a shuffle.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   summarize
   density
   reciprocity
   assortativity
```

## See also

- {doc}`/user-guide/analyze/graphs`: the guide that teaches these.
- {doc}`/user-guide/analyze/joins`: the join mechanics every algorithm here composes.
- {doc}`/api/relational/functions`: the scalar and aggregate functions the results feed.

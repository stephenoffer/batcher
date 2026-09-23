# Graphs

This page covers graph analytics in Batcher: building a graph from an edge table, running centrality, component, community, path, and link-prediction algorithms, and turning a graph into features a model can train on. Every algorithm is relational underneath, so it runs on the same optimizer, spill, and distributed machinery as any other query, over a graph stored wherever a table can be.

## A graph is an edge table

There is no graph data structure to build and no index to load. A graph is a {py:class}`Dataset <batcher.Dataset>` with two columns naming the endpoints, and every algorithm is a sequence of joins and aggregations over it:

```python
import batcher as bt
import batcher.graph as bg

follows = bt.from_pydict(
    {
        "user": ["ann", "bob", "cy", "dee", "ann"],
        "follows": ["bob", "cy", "ann", "ann", "cy"],
    }
)
g = bg.Graph.from_edges(follows, src="user", dst="follows")
print(bg.summarize(g).to_pydict())
# {'nodes': [4], 'edges': [5], 'density': [0.4166666666666667], 'reciprocity': [0.4],
#  'average_degree': [2.5], 'max_degree': [4], 'isolated': [0], 'directed': [True]}
```

That choice is the whole design. A join is a join, so PageRank over an edge table too large for one machine distributes across a Ray cluster and spills under memory pressure using exactly the machinery any other query uses. The graph can live wherever a table can: Parquet on object storage, a lakehouse table, a database extract. And node identity is whatever the column holds, so strings and UUIDs work with no vertex-id mapping on the side.

:::{tip}
`summarize` is the right first call on any graph you did not build yourself. The cheap numbers tell you which expensive algorithms are worth running: a graph that is one giant component behaves nothing like a thousand islands, and a degree distribution with a long tail makes triangle counting cost far more than its average degree suggests.
:::

## Directed and undirected graphs

`Graph.from_edges` treats direction as meaningful unless you pass `directed=False`. An undirected graph still stores each edge once, in whichever orientation the row was written, and every algorithm that follows edges walks it in both directions. On the path `1 - 2 - 3`, written with its second edge backwards, a search from either end reaches the other, and PageRank gives both ends the same score:

```python
path3 = bg.Graph.from_edges(bt.from_pydict({"src": [1, 3], "dst": [2, 2]}), directed=False)
print(bg.bfs(path3, bt.from_pydict({"node": [1]})).sort("node").to_pydict())
# {'node': [1, 2, 3], 'depth': [0, 1, 2]}
print([round(v, 4) for v in bg.pagerank(path3).sort("node").to_pydict()["pagerank"]])
# [0.2568, 0.4865, 0.2568]
```

Each row is one edge. A table that lists both `(1, 2)` and `(2, 1)` for one undirected edge gives the graph two parallel edges between those nodes, which doubles their weight in PageRank and their `degree`. Deduplicate such a table first, or build it with the default `directed=True`. The algorithms that are defined on an undirected graph, which are components, triangles, clustering, `k_core` and the link-prediction scores, ignore direction and collapse parallel edges whichever flag you pass.

## Isolated nodes are invisible unless you say otherwise

An edge table cannot express a node with no edges. Such a node appears in no row. That silently changes every per-node average:

```python
g_edges_only = bg.Graph.from_edges(bt.from_pydict({"src": [1], "dst": [2]}))
print(g_edges_only.num_nodes(), round(bg.average_degree(g_edges_only), 3))
# 2 1.0

with_all = g_edges_only.with_nodes(bt.from_pydict({"node": [1, 2, 3, 4]}))
print(with_all.num_nodes(), round(bg.average_degree(with_all), 3))
# 4 0.5
```

The edge between 1 and 2 is the same in both graphs. What changes is which nodes exist:

![Before and after, over the same single edge row with src 1 and dst 2. Built with Graph.from_edges alone, the graph holds nodes 1 and 2 joined by one edge, because a node with no edges appears in no row and so does not exist: num_nodes() is 2 and average_degree is 1.0. After with_nodes() attaches a node table holding 1, 2, 3 and 4, the graph holds the same edge from 1 to 2 plus nodes 3 and 4, which are isolated with no edges: num_nodes() is 4 and average_degree is 0.5. Both answers are correct for their question.](/_static/diagrams/graph_isolated_nodes.svg)

Both answers are correct for their question. Attach the node table when the denominator should include everyone.

## Centrality: who matters

`pagerank` is the default. A node scores highly when important nodes point at it and do not point at much else, which is what makes it far harder to game than degree:

```python
star = bt.from_pydict({"src": [1, 2, 3, 4], "dst": [0, 0, 0, 0]})
sg = bg.Graph.from_edges(star)
ranked = bg.pagerank(sg).sort("pagerank", descending=True)
print([(n, round(v, 4)) for n, v in zip(*ranked.to_pydict().values())])
# [(0, 0.5238), (1, 0.119), (2, 0.119), (3, 0.119), (4, 0.119)]
```

Ranks sum to 1, including the mass that would otherwise leak. A node with no outgoing edges, such as node 0 here, has nowhere to send its rank, and a hand-rolled PageRank that does not redistribute that mass quietly stops summing to 1 while still looking plausible.

`personalized_pagerank` teleports back to a chosen set instead of anywhere, which is the recommendation primitive. Seed it with the items one user touched, and the ranking that comes back orders everything else by closeness to those, measured through the whole graph.

```python
chain = bg.Graph.from_edges(bt.from_pydict({"src": [1, 2, 3], "dst": [2, 3, 4]}))
near_1 = bg.personalized_pagerank(chain, bt.from_pydict({"node": [1]}))
print(
    [
        (n, round(v, 3))
        for n, v in zip(*near_1.sort("pagerank", descending=True).to_pydict().values())
    ]
)
# [(1, 0.314), (2, 0.267), (3, 0.227), (4, 0.193)]
```

`hits` makes a distinction PageRank cannot. On a citation graph a survey paper and a seminal paper are both important, in opposite directions. It reports a hub score and an authority score per node.

`betweenness_centrality` finds *bridges* rather than hubs. A node joining two dense clusters can have a small degree and a tiny PageRank while every path between the clusters runs through it. That is the single-point-of-failure measure.

```python
bridge = bt.from_pydict({"src": ["a", "b", "c", "d"], "dst": ["c", "c", "d", "e"]})
bg2 = bg.Graph.from_edges(bridge)
scored = bg.betweenness_centrality(bg2, bt.from_pydict({"node": ["a", "b"]}))
print(scored.sort("betweenness", descending=True).to_pydict())
# {'node': ['c', 'd', 'a', 'b', 'e'], 'betweenness': [4.0, 2.0, 0.0, 0.0, 0.0]}
```

It is an estimate over the sources you give it, because the exact measure needs shortest paths from every node. The values scale with the source count, so compare ranks between runs rather than magnitudes.

## Components: what the pieces are

```python
islands = bt.from_pydict({"src": [1, 2, 8], "dst": [2, 3, 9]})
ig = bg.Graph.from_edges(islands)
print(bg.component_sizes(ig).to_pydict())
# {'component': [1, 8], 'nodes': [3, 2]}
print(bg.largest_component(ig).num_edges())
# 2
```

Component labels are the smallest node id in the component rather than an arbitrary number, so they are stable across runs and comparable between them.

`k_core` is the graduated version: repeatedly removing every node with fewer than `k` neighbors leaves the densely interconnected part. The removal cascades, since dropping a node lowers its neighbors' degrees too.

```python
triangle_plus_tail = bt.from_pydict({"src": [1, 2, 3, 1], "dst": [2, 3, 1, 9]})
tg = bg.Graph.from_edges(triangle_plus_tail)
print(sorted(bg.k_core(tg, 2).nodes().to_pydict()["node"]))
# [1, 2, 3]
```

:::{warning}
`Graph.to_undirected` materializes both directions of every edge, so on the result `degree` counts each neighbor twice. Use `out_degree` for a neighbor count on a symmetrized graph. Getting this wrong is why a k-core would keep the pendant nodes it exists to peel.
:::

On a *directed* graph, "connected" has a second, stronger meaning: two nodes are in the same *strongly* connected component only when each can reach the other following edge direction. A chain is one weak component and N strong ones, because nothing gets back:

```python
chain_and_cycle = bt.from_pydict({"src": [1, 2, 3, 3], "dst": [2, 3, 1, 4]})
dg = bg.Graph.from_edges(chain_and_cycle)
print(bg.strongly_connected_components(dg).sort("node").to_pydict()["component"])
# [3, 3, 3, 4]
print(bg.connected_components(dg).sort("node").to_pydict()["component"])
# [1, 1, 1, 1]
```

## Dependency graphs

A build order, a task schedule and a package graph are one question. Can these be ordered so every edge points forward? If not, what is in the way?

`topological_order` returns *levels* rather than a flat sequence, because nodes at the same level are mutually independent and a scheduler can run a whole level at once:

```python
deps = bt.from_pydict({"src": ["a", "a", "b", "c"], "dst": ["b", "c", "d", "d"]})
print(bg.topological_order(bg.Graph.from_edges(deps)).sort("node").to_pydict())
# {'node': ['a', 'b', 'c', 'd'], 'level': [0, 1, 1, 2]}
```

A node inside or downstream of a cycle can never lose its last incoming edge, so it is absent from the order. That makes the count the acyclicity test, and it means the diagnostic comes free:

```python
broken = bt.from_pydict({"src": ["a", "b", "c", "c"], "dst": ["b", "c", "b", "d"]})
bg_broken = bg.Graph.from_edges(broken)
print(bg.is_dag(bg_broken), sorted(bg.nodes_in_cycles(bg_broken).to_pydict()["node"]))
# False ['b', 'c', 'd']
```

## Triangles, clustering and communities

Triangles are the smallest structure that distinguishes a real social graph from a random one with the same degrees. If your friends know each other, the graph has triangles.

```python
two_triangles = bt.from_pydict({"src": [1, 2, 1, 4, 5, 4], "dst": [2, 3, 3, 5, 6, 6]})
cg = bg.Graph.from_edges(two_triangles)
print(bg.triangles(cg).count(), round(bg.average_clustering(cg), 3))
# 2 1.0
```

`label_propagation` finds communities in near-linear time with no parameter beyond a round cap, and `modularity` scores the result. Above roughly 0.3 means real structure, and near zero means the partition explains nothing the degree sequence does not already:

```python
communities = bg.label_propagation(cg)
print(communities.sort("node").to_pydict()["community"])
# [1, 1, 1, 4, 4, 4]
print(round(bg.modularity(cg, communities), 4))
# 0.5
```

:::{important}
Triangle counting is a three-way join, so its cost is driven by the highest-degree node rather than the average. On a scale-free graph one celebrity account can dominate the whole run. Run `k_core` first, or cap the degree, before counting triangles on a large graph.
:::

## Distances

`bfs` expands a frontier one hop at a time, so its cost is proportional to the part of the graph it reaches rather than to the whole of it:

```python
path = bg.Graph.from_edges(bt.from_pydict({"src": [1, 2, 3], "dst": [2, 3, 4]}))
print(bg.bfs(path, bt.from_pydict({"node": [1]})).sort("node").to_pydict())
# {'node': [1, 2, 3, 4], 'depth': [0, 1, 2, 3]}
```

`shortest_path_lengths` is the weighted version, and it finds the cheap detour rather than the short one:

```python
detour = bt.from_pydict({"src": [1, 1, 2], "dst": [2, 3, 3], "w": [1.0, 9.0, 1.0]})
dg = bg.Graph.from_edges(detour, weight="w")
print(bg.shortest_path_lengths(dg, bt.from_pydict({"node": [1]})).sort("node").to_pydict())
# {'node': [1, 2, 3], 'distance': [0.0, 1.0, 2.0]}
```

There is no all-pairs function. That is deliberate, because an all-pairs distance matrix is quadratic in node count and does not fit anywhere. `harmonic_centrality` and `diameter_estimate` take a set of sources and are named for being estimates.

## Iteration caps and convergence

Most algorithms here repeat a join until nothing changes. What happens when a round cap arrives first depends on whether the algorithm computes an exact answer or an approximation.

The exact ones are `connected_components` and the functions built on it, `strongly_connected_components`, `k_core`, `bfs`, `shortest_path_lengths`, `reachable_from`, `topological_order`, `is_dag` and `nodes_in_cycles`. They have no nearly-right answer: a component split in two is wrong. Their `max_iterations` or `max_depth` defaults to `None`, which runs until the answer is final and never needs more rounds than the graph has nodes. A `max_iterations` you set that stops one early raises {py:exc}`PlanError <batcher.PlanError>` instead of returning the truncated result. The `max_depth` of the searches is different: it is a filter you choose, so nodes beyond it are left out without an error.

```python
long_chain = bg.Graph.from_edges(
    bt.from_pydict({"src": list(range(149)), "dst": list(range(1, 150))})
)
print(bg.connected_components(long_chain).count_distinct("component"))
# 1
try:
    bg.connected_components(long_chain, max_iterations=2)
except bt.PlanError as refused:
    print(str(refused).split(".")[0])
# connected_components() did not reach its fixpoint within max_iterations=2 rounds
```

The approximate ones are `pagerank`, `personalized_pagerank`, `katz_centrality`, `eigenvector_centrality`, `hits` and `label_propagation`. At the cap they return their last iterate, which is usually a usable ranking, and warn with `ConvergenceWarning` so a truncated run is not mistaken for a converged one. Raise `max_iterations`, loosen `tolerance`, or turn the warning into an error when a truncated answer is not acceptable:

```python
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    bg.pagerank(long_chain, max_iterations=3)
print([w.category.__name__ for w in caught])
# ['ConvergenceWarning']
```

```{eval-rst}
.. autoclass:: batcher.graph.ConvergenceWarning
```

`katz_centrality` can also diverge. Its series converges only when `attenuation` is below the reciprocal of the adjacency matrix's largest eigenvalue, and past that the scores grow without bound. A run whose change per round is still growing when it reaches the cap raises `PlanError` rather than returning the direction of the blow-up. `1 / (1 + the largest weighted_degree)` is an attenuation that converges on every graph.

## Link prediction

Which unconnected pairs look like they should be? The scores differ in how much a shared neighbor is worth:

```python
social = bt.from_pydict({"src": [1, 3, 1, 3], "dst": [2, 2, 4, 4]})
sg2 = bg.Graph.from_edges(social)
pairs = bg.candidate_pairs(sg2)
print(pairs.sort("a").to_pydict())
# {'a': [1, 2], 'b': [3, 4], 'common': [2, 2]}
scored = bg.adamic_adar(sg2, pairs.select("a", "b"))
print([round(v, 4) for v in scored.to_pydict()["adamic_adar"]])
# [2.8854, 2.8854]
```

`adamic_adar` weights a shared neighbor by `1 / log(its degree)`, on the insight that sharing an obscure neighbor is strong evidence and sharing a celebrity is almost none. `preferential_attachment` ignores neighbors entirely and scores by degree, which makes it the baseline the others have to beat. A neighborhood score that does not beat it is not using the neighborhood.

:::{warning}
`candidate_pairs` is a self-join of the adjacency table, so its cost is the sum of squared degrees. One node with a million neighbors produces a trillion pairs on its own. Pass `max_degree` to drop hubs from the generator, which loses few real candidates for exactly the reason `adamic_adar` formalizes.
:::

## Graph ML: sampling and features

A GNN cannot see a large graph at once. The two standard ways around that are both here.

`neighbor_sample` bounds how many neighbors each node contributes, so a layer's cost is set by the bound rather than by the worst node. This is GraphSAGE, applied once per layer:

```python
skew = bt.from_pydict({"src": [1, 1, 1, 1, 2], "dst": [2, 3, 4, 5, 3]})
kg = bg.Graph.from_edges(skew)
print(bg.neighbor_sample(kg, 2).group_by("src").agg(n=bt.count()).sort("src").to_pydict())
# {'src': [1, 2], 'n': [2, 1]}
```

`random_walks` turns the graph into sequences, which is how DeepWalk and node2vec produce node embeddings: feed the walks to a word-embedding model and the geometry that comes back reflects the graph's structure.

```python
cycle = bg.Graph.from_edges(bt.from_pydict({"src": [1, 2, 3], "dst": [2, 3, 1]}))
walk = bg.random_walks(cycle, bt.from_pydict({"node": [1]}), 3)
print(walk.sort("step").to_pydict()["node"])
# [1, 2, 3, 1]
```

Both are deterministic given a seed. That is not politeness. A neighbor sample that changes between the training and inference passes is an accuracy loss that looks like drift, and an embedding trained on walks you cannot regenerate is one you cannot debug.

For features, `aggregate_neighbors` is one round of message passing, and `propagate_features` stacks several while keeping each round's output:

```python
flow = bg.Graph.from_edges(bt.from_pydict({"src": [1, 2], "dst": [2, 3]}))
feats = bt.from_pydict({"node": [1, 2, 3], "x": [1.0, 0.0, 0.0]})
print(bg.propagate_features(flow, feats, ["x"], 2).sort("node").to_pydict())
# {'node': [1, 2, 3], 'x': [1.0, 0.0, 0.0], 'x_hop1': [None, 1.0, 0.0],
#  'x_hop2': [None, 0.0, 1.0]}
```

That stack of hop columns is what a GNN learns to weight. Handing it to a gradient-boosted model instead is a strong baseline that trains in seconds and is far easier to explain.

`structural_features` goes the other way. It describes each node by its position alone, with no node attributes at all. On fraud and abuse problems those columns are frequently the strongest signal available, because the behavior is a shape in the graph rather than a property of any single account.

## When the data is not already an edge list

Embeddings, coordinates and interaction logs all want graph analysis, and none of them arrive as edges. Four constructors make that step explicit.

`knn_graph` connects each vector to its nearest neighbors, which is the bridge from an embedding space to every algorithm above:

```python
vecs = bt.from_pydict({"node": ["a", "b", "c"], "vector": [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]})
print(bg.knn_graph(vecs, 1).edges.sort("src").to_pydict()["dst"])
# ['b', 'a', 'b']
```

`threshold_graph` connects everything closer than a cut-off, which is the deduplication shape: take `connected_components` of the result and each component is a cluster of records that are the same thing. The transitive closure is the point, since A matching B and B matching C groups all three even when A and C do not match directly. A record that matched nothing comes back as its own cluster rather than vanishing:

```python
records = bt.from_pydict(
    {"node": ["a", "b", "c"], "vector": [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]}
)
dedup = bg.threshold_graph(records, 0.9)
print(bg.connected_components(dedup).sort("node").to_pydict()["component"])
# ['a', 'a', 'c']
```

`spatial_graph` connects positions within a geodesic radius in meters, so the radius means the same thing at every latitude:

```python
sites = bt.from_pydict(
    {
        "node": ["ferry", "pier", "opera"],
        "geometry": [
            "POINT(-122.3937 37.7955)",
            "POINT(-122.3930 37.7960)",
            "POINT(151.2153 -33.8568)",
        ],
    }
)
near = bg.spatial_graph(sites, 200.0)
print(near.edges.sort("src").to_pydict()["src"], near.num_nodes())
# ['ferry', 'pier'] 3
```

`co_occurrence_graph` projects a user-item log into an item-item graph, which is the classic collaborative-filtering signal:

```python
baskets = bt.from_pydict(
    {
        "user": ["u1", "u1", "u2", "u2", "u3"],
        "item": ["bread", "jam", "bread", "jam", "shovel"],
    }
)
print(bg.co_occurrence_graph(baskets, min_count=2).edges.sort("src").to_pydict())
# {'src': ['bread', 'jam'], 'dst': ['jam', 'bread'], 'weight': [2.0, 2.0]}
```

:::{warning}
All four compare every pair unless you block them. A million rows is a trillion comparisons. Every one takes a `block` argument that restricts comparison to rows sharing a key, and it is exact within each block. Partition by a coarse cluster id, a date, a category, or a geohash prefix. Choosing that key is the engineering in each of these, not an optimization to add later.
:::

## Reshaping a graph before an algorithm reads it

Most graph work starts by narrowing. A centrality measure over a whole interaction log
answers a question nobody asked, and a self-loop quietly inflates the node it belongs to.
Two reshaping operations do that narrowing, and both return a new `Graph`, so they compose
before any algorithm touches the edges.

`without_self_loops` drops every `a -> a` edge. A self-loop is an edge like any other until
you say otherwise, and it is the one that flatters a node's own score.

```python
import batcher as bt
from batcher.graph import Graph

edges = bt.from_pydict({"src": [1, 2, 3, 3], "dst": [2, 3, 1, 3]})
g = Graph.from_edges(edges, src="src", dst="dst")
print(g.num_edges(), g.without_self_loops().num_edges())
```

`extra_nodes` answers where a graph's node set came from. A graph built only from edges
derives its nodes from them, so a node that appears in no edge is invisible, and
`extra_nodes` returns `None` to say so. After `with_nodes` attaches a roster, `extra_nodes`
hands that roster back. The point is that a caller can tell the two apart without comparing
them, which matters before reporting a node count.

```python
print(g.extra_nodes())

roster = bt.from_pydict({"node": [1, 2, 3, 99]})
with_isolated = g.with_nodes(roster)
print(sorted(with_isolated.extra_nodes().to_pydict()["node"]))
print(sorted(with_isolated.nodes().to_pydict()["node"]))
```

Node 99 exists in the roster and in no edge, so it reaches `nodes()` only through
`with_nodes`. An algorithm that scores every node scores it too, at whatever value the
absence of edges implies.

## On a cluster

Read the edge table from Parquet or a lakehouse table, and the same calls run on a Ray cluster. The graph functions call `collect` internally and take no `distributed=` argument, so the session option `distributed.mode` is how you pin them. `"auto"`, the default, decides per query from the input size and the cluster. `"always"` sends every query to the Ray path and starts a local Ray when none is running. `"never"` keeps everything on one node.

```python
# docs: skip
import batcher as bt
import batcher.graph as bg

g = bg.Graph.from_edges(bt.read.parquet("s3://<your-bucket>/edges/*.parquet"))
with bt.config.option_context("distributed.mode", "always"):
    print(bg.summarize(g).to_pydict())
    ranks = bg.pagerank(g).collect()
    triangles = bg.triangle_count(g).collect()
```

Replace `<your-bucket>` with the location of your edge files. The algorithms return the same answer on one node and across workers, up to the last bits of a floating-point sum. Each of the following was run under `"never"` and under `"always"` on 80,000 edges in four Parquet files, with the same result both ways: the degree functions and `degree_distribution`, `summarize`, `density`, `reciprocity`, `assortativity`, `triangle_count`, `clustering_coefficient`, `average_clustering`, `transitivity`, `k_core`, `connected_components`, `strongly_connected_components`, `label_propagation`, `modularity`, `pagerank`, `personalized_pagerank`, `katz_centrality`, `eigenvector_centrality`, `hits`, `betweenness_centrality`, `harmonic_centrality`, `bfs`, `shortest_path_lengths`, `diameter_estimate`, `topological_order`, `is_dag`, the link-prediction scores and `candidate_pairs`, `structural_features`, `aggregate_neighbors`, `neighbor_sample`, `random_walks` and `ego_network`. The functions built directly on these, such as `component_sizes` and `k_hop_neighbors`, and the four graph builders were not run separately. [`tests/integration/test_graph_distributed.py`](https://github.com/stephenoffer/batcher/blob/main/tests/integration/test_graph_distributed.py) holds the main ones to that result.

The iterative algorithms carry one cost the others don't. Each round's per-node state is collected into the driver and handed back as the next round's input, because a lazy plan built fifty rounds deep would re-run every earlier round when it finally executed. So each round pays a round trip through the driver on top of its joins, and the driver has to hold one row per node. The edge-side joins still run on the workers. `structural_features` and `average_clustering` pass their per-node tables through the driver once in the same way, because an aggregate over a `union` cannot feed a join or a second aggregate on the cluster, and so does `degree_distribution` on a graph with a node table attached.

## Requirements and limitations

- Iterative algorithms pass per-node state through the driver once per round, as the cluster section above describes. A graph with more nodes than the driver can hold will not finish, however many workers you add. The degree functions, `triangle_count`, `clustering_coefficient` and the link-prediction scores are single aggregates over the edges and have no such limit.
- A round costs a few queries whatever the graph's size, so an exact algorithm on a long path pays that fixed cost once per hop. `connected_components` jumps labels along the path, which settled a 300-node path in 11 rounds with its ids in order and in 79 with them shuffled, but `strongly_connected_components`, `bfs`, `shortest_path_lengths` and `topological_order` advance one hop per round.
- `betweenness_centrality` runs one breadth-first search forward and one sweep back per source, each hop a separate query, so its cost is sources times depth queries. It is meant for tens of sampled sources, not for exact betweenness over every node of a large graph. `harmonic_centrality` searches from all its sources at once, so its query count is the search depth, and its per-round state is one row per source and reached node.
- An in-memory edge table never distributes under `"auto"`, by design. `distributed="auto"` routes a plan whose sources are all resident in the driver to single-node at any size, because shipping resident data out and gathering it back costs more than the compute it parallelizes. Read the edges from Parquet or a lakehouse table to get the distributed path. {py:func}`bt.from_pydict <batcher.from_pydict>` stays local no matter how large it is.
- `Graph.cache` helps single-node only. {py:meth}`Dataset.cache <batcher.Dataset.cache>` memoizes a result in a process-local LRU, which the distributed executor does not consult, so on a cluster each algorithm re-reads the edge table.
- `connected_components` is weakly connected. The graph is symmetrized first, so `a -> b -> c` is one component even though nothing reaches `a`. `strongly_connected_components` is the direction-respecting version and costs more, because it is a coloring algorithm rather than Tarjan's, whose depth-first search has no relational form.
- Distances are single-source or multi-source, never all-pairs. `harmonic_centrality`, `diameter_estimate` and `betweenness_centrality` all take a source set and are estimates over it. There is no exact betweenness or closeness centrality, because both need all-pairs shortest paths. For betweenness the ranking stabilizes long before the values do, so sample a few dozen high-degree sources and compare ranks rather than magnitudes. On an undirected graph, betweenness summed over every node as a source counts each pair from both ends, so halve it to get the textbook value.
- `label_propagation` is not stable under small changes and does not always settle. Ties break deterministically, so a run is reproducible, but a slightly different graph can produce a very different partition, and two neighbors can trade labels forever, which ends in a `ConvergenceWarning`. Score the result with `modularity` rather than trusting the label count.
- `shortest_path_lengths` requires non-negative weights, and refuses a negative one rather than diverging.
- `density` and `modularity` do not count a self-loop the way networkx does. `density` leaves self-loops out, and `modularity` counts a self-loop's weight once in its node's strength where networkx counts it twice.

## Runnable examples

The scripts in [`examples/graph`](https://github.com/stephenoffer/batcher/tree/main/examples/graph) run end to end and check their answers against values worked out by hand:

- [`undirected_graphs.py`](https://github.com/stephenoffer/batcher/blob/main/examples/graph/undirected_graphs.py): an undirected graph walked both ways, clustering with self-loops and isolated nodes, and what a cap does to an exact and an approximate algorithm.
- [`pagerank_and_centrality.py`](https://github.com/stephenoffer/batcher/blob/main/examples/graph/pagerank_and_centrality.py): PageRank on a customer-to-nation graph, checked against its closed form.
- [`triangles_and_clustering.py`](https://github.com/stephenoffer/batcher/blob/main/examples/graph/triangles_and_clustering.py): triangle counts, local clustering and transitivity on two triangles joined by a bridge.
- [`degree_and_components.py`](https://github.com/stephenoffer/batcher/blob/main/examples/graph/degree_and_components.py), [`shortest_paths.py`](https://github.com/stephenoffer/batcher/blob/main/examples/graph/shortest_paths.py), [`analytics.py`](https://github.com/stephenoffer/batcher/blob/main/examples/graph/analytics.py) and [`graph_ml.py`](https://github.com/stephenoffer/batcher/blob/main/examples/graph/graph_ml.py) cover the rest of this page.

## See also

- {doc}`/api/relational/domains/graph`: every graph function, grouped and enumerated.
- {doc}`/user-guide/analyze/joins`: the join mechanics every algorithm here composes.
- {doc}`/user-guide/analyze/aggregations`: the `group_by` behind every degree count.
- {doc}`/user-guide/analyze/domains/geospatial`: the geometry behind `spatial_graph`.

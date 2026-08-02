# Graphs

This page covers graph analytics in Batcher: building a graph from an edge table, running
the algorithms, and turning a graph into features a model can train on.

## A graph is an edge table

There is no graph data structure to build and no index to load. A graph is a `Dataset`
with two columns naming the endpoints, and every algorithm is a sequence of joins and
aggregations over it:

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

That choice is the whole design. A join is a join, so PageRank over a billion edges
distributes across a Ray cluster and spills under memory pressure using exactly the
machinery any other query uses. The graph can live wherever a table can: Parquet on
object storage, a lakehouse table, a database extract. And node identity is whatever the
column holds, so strings and UUIDs work with no vertex-id mapping on the side.

:::{tip}
`summarize` is the right first call on any graph you did not build yourself. The cheap
numbers tell you which expensive algorithms are worth running: a graph that is one giant
component behaves nothing like a thousand islands, and a degree distribution with a long
tail makes triangle counting cost far more than its average degree suggests.
:::

## Isolated nodes are invisible unless you say otherwise

An edge table cannot express a node with no edges, because such a node appears in no row.
That silently changes every per-node average:

```python
g_edges_only = bg.Graph.from_edges(bt.from_pydict({"src": [1], "dst": [2]}))
print(g_edges_only.num_nodes(), round(bg.average_degree(g_edges_only), 3))
# 2 1.0

with_all = g_edges_only.with_nodes(bt.from_pydict({"node": [1, 2, 3, 4]}))
print(with_all.num_nodes(), round(bg.average_degree(with_all), 3))
# 4 0.5
```

Both answers are correct for their question. Attach the node table when the denominator
should include everyone.

## Centrality: who matters

`pagerank` is the default. A node scores highly when important nodes point at it and do
not point at much else, which is what makes it far harder to game than degree:

```python
star = bt.from_pydict({"src": [1, 2, 3, 4], "dst": [0, 0, 0, 0]})
sg = bg.Graph.from_edges(star)
ranked = bg.pagerank(sg).sort("pagerank", descending=True)
print([(n, round(v, 4)) for n, v in zip(*ranked.to_pydict().values())])
# [(0, 0.5238), (1, 0.119), (2, 0.119), (3, 0.119), (4, 0.119)]
```

Ranks sum to 1, including the mass that would otherwise leak. A node with no outgoing
edges (node 0 here) has nowhere to send its rank, and a hand-rolled PageRank that does
not redistribute that mass quietly stops summing to 1 while still looking plausible.

`personalized_pagerank` teleports back to a chosen set instead of anywhere, which is the
recommendation primitive: seed it with what one user touched and the ranking that comes
back is what else is close to those, measured through the whole graph.

```python
chain = bg.Graph.from_edges(bt.from_pydict({"src": [1, 2, 3], "dst": [2, 3, 4]}))
near_1 = bg.personalized_pagerank(chain, bt.from_pydict({"node": [1]}))
print([(n, round(v, 3)) for n, v in zip(*near_1.sort("pagerank", descending=True).to_pydict().values())])
# [(1, 0.314), (2, 0.267), (3, 0.227), (4, 0.193)]
```

`hits` makes a distinction PageRank cannot: on a citation graph a survey paper and a
seminal paper are both important, in opposite directions. It reports a hub score and an
authority score per node.

`betweenness_centrality` finds *bridges* rather than hubs: a node joining two dense
clusters can have a small degree and a tiny PageRank while every path between the clusters
runs through it. That is the single-point-of-failure measure.

```python
bridge = bt.from_pydict({"src": ["a", "b", "c", "d"], "dst": ["c", "c", "d", "e"]})
bg2 = bg.Graph.from_edges(bridge)
scored = bg.betweenness_centrality(bg2, bt.from_pydict({"node": ["a", "b"]}))
print(scored.sort("betweenness", descending=True).to_pydict())
# {'node': ['c', 'd', 'a', 'b', 'e'], 'betweenness': [4.0, 2.0, 0.0, 0.0, 0.0]}
```

It is an estimate over the sources you give it, because the exact measure needs shortest
paths from every node. The values scale with the source count, so compare ranks between
runs rather than magnitudes.

## Components: what the pieces are

```python
islands = bt.from_pydict({"src": [1, 2, 8], "dst": [2, 3, 9]})
ig = bg.Graph.from_edges(islands)
print(bg.component_sizes(ig).to_pydict())
# {'component': [1, 8], 'nodes': [3, 2]}
print(bg.largest_component(ig).num_edges())
# 2
```

Component labels are the smallest node id in the component rather than an arbitrary
number, so they are stable across runs and comparable between them.

`k_core` is the graduated version: repeatedly removing every node with fewer than `k`
neighbours leaves the densely interconnected part. The removal cascades, since dropping a
node lowers its neighbours' degrees too.

```python
triangle_plus_tail = bt.from_pydict({"src": [1, 2, 3, 1], "dst": [2, 3, 1, 9]})
tg = bg.Graph.from_edges(triangle_plus_tail)
print(sorted(bg.k_core(tg, 2).nodes().to_pydict()["node"]))
# [1, 2, 3]
```

:::{warning}
`Graph.to_undirected` materializes both directions of every edge, so on the result
`degree` counts each neighbour twice. Use `out_degree` for a neighbour count on a
symmetrized graph. Getting this wrong is why a k-core would keep the pendant nodes it
exists to peel.
:::

On a *directed* graph, "connected" has a second, stronger meaning: two nodes are in the
same **strongly** connected component only when each can reach the other following edge
direction. A chain is one weak component and N strong ones, because nothing gets back:

```python
chain_and_cycle = bt.from_pydict({"src": [1, 2, 3, 3], "dst": [2, 3, 1, 4]})
dg = bg.Graph.from_edges(chain_and_cycle)
print(bg.strongly_connected_components(dg).sort("node").to_pydict()["component"])
# [3, 3, 3, 4]
print(bg.connected_components(dg).sort("node").to_pydict()["component"])
# [1, 1, 1, 1]
```

## Dependency graphs

A build order, a task schedule and a package graph are all the same question: can these
be ordered so every edge points forward, and if not, what is in the way.

`topological_order` returns *levels* rather than a flat sequence, because nodes at the
same level are mutually independent and a scheduler can run a whole level at once:

```python
deps = bt.from_pydict({"src": ["a", "a", "b", "c"], "dst": ["b", "c", "d", "d"]})
print(bg.topological_order(bg.Graph.from_edges(deps)).sort("node").to_pydict())
# {'node': ['a', 'b', 'c', 'd'], 'level': [0, 1, 1, 2]}
```

A node inside or downstream of a cycle can never lose its last incoming edge, so it is
simply absent from the order. That makes the count the acyclicity test, and it means the
diagnostic comes free:

```python
broken = bt.from_pydict({"src": ["a", "b", "c", "c"], "dst": ["b", "c", "b", "d"]})
bg_broken = bg.Graph.from_edges(broken)
print(bg.is_dag(bg_broken), sorted(bg.nodes_in_cycles(bg_broken).to_pydict()["node"]))
# False ['b', 'c', 'd']
```

## Triangles, clustering and communities

Triangles are the smallest structure that distinguishes a real social graph from a random
one with the same degrees. If your friends know each other, the graph has triangles.

```python
two_triangles = bt.from_pydict(
    {"src": [1, 2, 1, 4, 5, 4], "dst": [2, 3, 3, 5, 6, 6]}
)
cg = bg.Graph.from_edges(two_triangles)
print(bg.triangles(cg).count(), round(bg.average_clustering(cg), 3))
# 2 1.0
```

`label_propagation` finds communities in near-linear time with no parameter beyond a
round cap, and `modularity` scores the result. Above roughly 0.3 means real structure;
near zero means the partition explains nothing the degree sequence does not already:

```python
communities = bg.label_propagation(cg)
print(communities.sort("node").to_pydict()["community"])
# [1, 1, 1, 4, 4, 4]
print(round(bg.modularity(cg, communities), 4))
# 0.5
```

:::{important}
Triangle counting is a three-way join, so its cost is driven by the highest-degree node
rather than the average. On a scale-free graph one celebrity account can dominate the
whole run. Run `k_core` first, or cap the degree, before counting triangles on a large
graph.
:::

## Distances

`bfs` expands a frontier one hop at a time, so its cost is proportional to the part of
the graph it reaches rather than to the whole of it:

```python
path = bg.Graph.from_edges(bt.from_pydict({"src": [1, 2, 3], "dst": [2, 3, 4]}))
print(bg.bfs(path, bt.from_pydict({"node": [1]})).sort("node").to_pydict())
# {'node': [1, 2, 3, 4], 'depth': [0, 1, 2, 3]}
```

`shortest_path_lengths` is the weighted version, and it finds the cheap detour rather than
the short one:

```python
detour = bt.from_pydict({"src": [1, 1, 2], "dst": [2, 3, 3], "w": [1.0, 9.0, 1.0]})
dg = bg.Graph.from_edges(detour, weight="w")
print(bg.shortest_path_lengths(dg, bt.from_pydict({"node": [1]})).sort("node").to_pydict())
# {'node': [1, 2, 3], 'distance': [0.0, 1.0, 2.0]}
```

There is no all-pairs function, and that is deliberate: an all-pairs distance matrix is
quadratic in node count and does not fit anywhere. `harmonic_centrality` and
`diameter_estimate` take a set of sources and are named for being estimates.

## Link prediction

Which pairs that are not connected look like they should be. The scores differ in how
much a shared neighbour is worth:

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

`adamic_adar` weights a shared neighbour by `1 / log(its degree)`, on the insight that
sharing an obscure neighbour is strong evidence and sharing a celebrity is almost none.
`preferential_attachment` ignores neighbours entirely and scores by degree, which makes it
the baseline the others have to beat: a neighbourhood score that does not is not using the
neighbourhood.

:::{warning}
`candidate_pairs` is a self-join of the adjacency table, so its cost is the sum of squared
degrees. One node with a million neighbours produces a trillion pairs on its own. Pass
`max_degree` to drop hubs from the generator, which loses few real candidates for exactly
the reason `adamic_adar` formalizes.
:::

## Graph ML: sampling and features

A GNN cannot see a large graph at once. The two standard ways around that are both here.

`neighbor_sample` bounds how many neighbours each node contributes, so a layer's cost is
set by the bound rather than by the worst node. This is GraphSAGE, applied once per layer:

```python
skew = bt.from_pydict({"src": [1, 1, 1, 1, 2], "dst": [2, 3, 4, 5, 3]})
kg = bg.Graph.from_edges(skew)
print(bg.neighbor_sample(kg, 2).group_by("src").agg(n=bt.count()).sort("src").to_pydict())
# {'src': [1, 2], 'n': [2, 1]}
```

`random_walks` turns the graph into sequences, which is how DeepWalk and node2vec produce
node embeddings: feed the walks to a word-embedding model and the geometry that comes back
reflects the graph's structure.

```python
cycle = bg.Graph.from_edges(bt.from_pydict({"src": [1, 2, 3], "dst": [2, 3, 1]}))
walk = bg.random_walks(cycle, bt.from_pydict({"node": [1]}), 3)
print(walk.sort("step").to_pydict()["node"])
# [1, 2, 3, 1]
```

Both are deterministic given a seed. That is not politeness: a neighbour sample that
changes between the training and inference passes is an accuracy loss that looks like
drift, and an embedding trained on walks you cannot regenerate is one you cannot debug.

For features, `aggregate_neighbors` is one round of message passing, and
`propagate_features` stacks several while keeping each round's output:

```python
flow = bg.Graph.from_edges(bt.from_pydict({"src": [1, 2], "dst": [2, 3]}))
feats = bt.from_pydict({"node": [1, 2, 3], "x": [1.0, 0.0, 0.0]})
print(bg.propagate_features(flow, feats, ["x"], 2).sort("node").to_pydict())
# {'node': [1, 2, 3], 'x': [1.0, 0.0, 0.0], 'x_hop1': [None, 1.0, 0.0],
#  'x_hop2': [None, 0.0, 1.0]}
```

That stack of hop columns is what a GNN learns to weight. Handing it to a gradient-boosted
model instead is a strong baseline that trains in seconds and is far easier to explain.

`structural_features` goes the other way and describes each node by its position alone,
needing no node attributes at all. On fraud and abuse problems those columns are
frequently the strongest signal available, because the behaviour is a shape in the graph
rather than a property of any single account.

## When the data is not already an edge list

Embeddings, coordinates and interaction logs all want graph analysis, and none of them
arrive as edges. Four constructors make that step explicit.

`knn_graph` connects each vector to its nearest neighbours, which is the bridge from an
embedding space to every algorithm above:

```python
vecs = bt.from_pydict(
    {"node": ["a", "b", "c"], "vector": [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]}
)
print(bg.knn_graph(vecs, 1).edges.sort("src").to_pydict()["dst"])
# ['b', 'a', 'b']
```

`threshold_graph` connects everything closer than a cut-off, which is the deduplication
shape: take `connected_components` of the result and each component is a cluster of
records that are the same thing. The transitive closure is the point, since A matching B
and B matching C groups all three even when A and C do not match directly. A record that
matched nothing comes back as its own cluster rather than vanishing:

```python
records = bt.from_pydict(
    {"node": ["a", "b", "c"], "vector": [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]}
)
dedup = bg.threshold_graph(records, 0.9)
print(bg.connected_components(dedup).sort("node").to_pydict()["component"])
# ['a', 'a', 'c']
```

`spatial_graph` connects positions within a geodesic radius in metres, so the radius
means the same thing at every latitude:

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

`co_occurrence_graph` projects a user-item log into an item-item graph, which is the
classic collaborative-filtering signal:

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
All four compare every pair unless you block them. A million rows is a trillion
comparisons. Every one takes a `block` argument that restricts comparison to rows sharing
a key, and it is exact within each block: partition by a coarse cluster id, a date, a
category, or a geohash prefix. Choosing that key is the engineering in each of these, not
an optimization to add later.
:::


## Requirements and limitations

- **Iterative algorithms pass per-node state through the driver once per round.** A lazy
  plan built fifty iterations deep would re-run every earlier iteration on execution, so
  each round's state is collected and re-wrapped. The edge-side joins still distribute --
  the state is one row per *node*, not per edge -- but the driver round-trip is the real
  ceiling on an iterative algorithm here: a graph with more nodes than the driver can hold
  will not finish, however many workers you add. The degree functions are single-pass and
  have no such limit.
- **Eleven algorithms cannot run under an explicit `distributed=True`.** They build a plan
  shape the distributed executor has no path for, so `collect(distributed=True)` on a
  file-backed graph raises `PlanError` rather than running. They still compute the right
  answer single-node, and the default `distributed="auto"` still returns it, so this is a
  scaling ceiling rather than a wrong result. The affected functions are `triangles`,
  `triangle_count`, `clustering_coefficient`, `structural_features`, `candidate_pairs`,
  `degree_distribution`, and the five link-prediction scores `adamic_adar`,
  `common_neighbors`, `jaccard_similarity`, `preferential_attachment` and
  `resource_allocation`.

  One engine limit accounts for all eleven, and it is reproducible in plain Batcher with no
  graph code (`tests/integration/test_distributed.py` pins it with its controls). An
  aggregate whose input contains a `union` cannot feed a join or a second aggregate. On its
  own it distributes, and it can feed a filter, a sort or a `distinct`, but a `group_by` or
  a join above it has no distributed path. Every one of these algorithms counts something
  per node, which means aggregating over the edge table read from both directions, and then
  joins or re-aggregates that count.

  Three natural readings of that limit are wrong, so do not plan around them: it is not
  about outer joins, since a `distinct` over the same union feeds a left join fine; it is
  not about a pipeline breaker over a `union`, since a `distinct` there is fine; and it is
  not about two-level aggregation, since `group_by` over `group_by` distributes over a plain
  scan.

  There are two ways around it. Where the join only restored zero rows for nodes that
  contributed none, emit those rows as another `union` arm feeding the same `group_by`
  instead; that is what the degree functions do and why they distribute. Otherwise
  materialize the aggregate with `bt.from_arrow(ds.collect())` before the next step, which
  clears the limit at the cost of passing that intermediate through the driver.
- **`betweenness_centrality` and `co_occurrence_graph` hang under `distributed=True`.**
  Unlike the eleven above they raise nothing: the query reserves the cluster and then waits
  on a task that can never be scheduled, and Carbonite reports `distributed barrier has
  waited 240s with 0/1 tasks finished, cluster CPU 32/32 in use` until you kill it. Both
  return in under a second single-node on the same input, and passing `num_workers` does
  not avoid it. Use the default `distributed="auto"`, which runs them single-node and
  returns the right answer. A hang is worse than a refusal, so this is the one limitation
  here worth checking before you script an unattended job against a file-backed graph.
- **An in-memory edge table never distributes, by design.** `distributed="auto"` routes a
  plan whose sources are all resident in the driver to single-node at any size, because
  shipping resident data out and gathering it back costs more than the compute it
  parallelizes. Read the edges from Parquet or a lakehouse table to get the distributed
  path; `bt.from_pydict` will stay local no matter how large it is.
- **`Graph.cache` helps single-node only.** `Dataset.cache` memoizes a result in a
  process-local LRU, which the distributed executor does not consult. Caching a graph
  before running several algorithms is worth it locally and is inert on a cluster, where
  each algorithm re-reads the edge table.
- **`connected_components` is weakly connected.** The graph is symmetrized first, so
  `a -> b -> c` is one component even though nothing reaches `a`.
  `strongly_connected_components` is the direction-respecting version, and is more
  expensive: it is a colouring algorithm rather than Tarjan's, whose depth-first search
  has no relational form.
- **Distances are single-source or multi-source, never all-pairs.**
  `harmonic_centrality`, `diameter_estimate` and `betweenness_centrality` all take a
  source set and are estimates over it. There is no *exact* betweenness or closeness
  centrality, because both need all-pairs shortest paths, which is quadratic in node
  count. For betweenness the ranking stabilizes long before the values do, so sample a
  few dozen high-degree sources and compare ranks rather than magnitudes.
- **`label_propagation` is not stable under small changes.** Ties break deterministically,
  so a run is reproducible, but a slightly different graph can produce a very different
  partition. Score with `modularity` rather than trusting the label count.
- **Weights must be non-negative for `shortest_path_lengths`**, which refuses a negative
  one rather than diverging.

## See also

- {doc}`/api/relational/graph`: every graph function, grouped and enumerated.
- {doc}`/user-guide/analyze/joins`: the join mechanics every algorithm here composes.
- {doc}`/user-guide/analyze/aggregations`: the `group_by` behind every degree count.

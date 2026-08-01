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

## Requirements and limitations

- **Iterative algorithms materialize per-node state once per round.** A lazy plan built
  fifty iterations deep would re-run every earlier iteration on execution, so the state is
  collected and re-wrapped each round. That is bounded by the node count rather than the
  edge count, but it does mean the state passes through the client process.
- **Components are weakly connected.** The graph is symmetrized first, so `a -> b -> c` is
  one component even though nothing reaches `a`. There is no strongly-connected-components
  function.
- **Distances are single-source or multi-source, never all-pairs.** `harmonic_centrality`
  and `diameter_estimate` take a source set and are estimates over it; there is no exact
  betweenness or closeness centrality, both of which need all-pairs.
- **`label_propagation` is not stable under small changes.** Ties break deterministically,
  so a run is reproducible, but a slightly different graph can produce a very different
  partition. Score with `modularity` rather than trusting the label count.
- **Weights must be non-negative for `shortest_path_lengths`**, which refuses a negative
  one rather than diverging.

## See also

- {doc}`/api/relational/graph`: every graph function, grouped and enumerated.
- {doc}`/user-guide/analyze/joins`: the join mechanics every algorithm here composes.
- {doc}`/user-guide/analyze/aggregations`: the `group_by` behind every degree count.

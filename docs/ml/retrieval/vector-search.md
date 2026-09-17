# Vector search

This page covers nearest-neighbor search in Batcher, from an exact scan in the engine to an approximate index over Lance. Two very different jobs get called vector search. "Score 50,000 candidate vectors against a query and take the top 10" is a sort, and it belongs in the engine. "Find the nearest 10 of 500 million" needs an approximate nearest-neighbor (ANN) index, because a full scan per query is too slow. Pick the wrong one and you either build an index you never needed or wait on a linear scan you shouldn't have run.

## Brute force, in the engine

When the vectors already sit in a column, score them with the {py:class}`.list <batcher.plan.expr_ir.namespaces.collections._ListNamespace>` distance expressions. There's no index, no extra service and no data movement, only a projection and a top-n.

```python
import batcher as bt
from batcher import array, col

docs = bt.from_pydict(
    {
        "id": [1, 2, 3, 4],
        "title": ["cats", "dogs", "kittens", "trains"],
        "vec": [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [-1.0, 0.0]],
    }
)
query = array(1.0, 0.0)

hits = docs.with_columns(dist=col("vec").list.cosine_distance(query)).sort("dist").limit(2)
print(hits.select("id", "title").to_pydict())
# {'id': [1, 3], 'title': ['cats', 'kittens']}
```

{py:meth}`ds.ml.nearest_neighbors(query, column, k, metric) <batcher.api.dataset.ml.DatasetML.nearest_neighbors>` is the one-call shorthand for that projection, sort and limit. `metric` is `"cosine"` by default, or `"l2"`, `"dot"`, `"l1"` or `"hamming"`:

```python
hits = docs.ml.nearest_neighbors([1.0, 0.0], column="vec", k=2)  # nearest first, + `distance`
```

Use the explicit form when you want the score column under your own name or combined with other predicates. Use the verb when you want the top `k` and nothing else.

Two companions complete the pattern. {py:meth}`ds.ml.similarity_to(query, column=, metric=) <batcher.api.dataset.ml.DatasetML.similarity_to>` scores every row against the query *without* the top-`k` cut, which is what thresholding and reranking need. {py:meth}`ds.ml.normalize_embeddings(column) <batcher.api.dataset.ml.DatasetML.normalize_embeddings>` unit-normalizes an embedding column so a later {py:meth}`.list.dot <batcher.plan.expr_ir.namespaces.collections._ListNamespace.dot>` ranks exactly as cosine does, at lower cost:

```python
scored = docs.ml.normalize_embeddings("vec").ml.similarity_to([1.0, 0.0], column="vec")
```

{py:meth}`cosine_distance <batcher.plan.expr_ir.namespaces.collections._ListNamespace.cosine_distance>` computes `1 - cosine_similarity`: 0 for identical direction, 1 for orthogonal, 2 for opposite. Sort it ascending and the nearest comes first. {py:meth}`.list.l2_distance <batcher.plan.expr_ir.namespaces.collections._ListNamespace.l2_distance>` is Euclidean, and `.list.dot` the raw inner product.

Three more read a single vector's *magnitude* instead of a pairwise distance. {py:meth}`.list.l2_norm() <batcher.plan.expr_ir.namespaces.collections._ListNamespace.l2_norm>` gives the Euclidean length. {py:meth}`.list.l1_norm() <batcher.plan.expr_ir.namespaces.collections._ListNamespace.l1_norm>` gives the Manhattan length, the sum of absolute values that L1 normalization divides by. {py:meth}`.list.max_abs() <batcher.plan.expr_ir.namespaces.collections._ListNamespace.max_abs>` returns the largest magnitude in the row, the divisor for MaxAbs scaling, which maps a feature vector into `[-1, 1]` without shifting its zero.

Other embedding geometries have matching metrics. {py:meth}`.list.l1_distance <batcher.plan.expr_ir.namespaces.collections._ListNamespace.l1_distance>` is
Manhattan distance, the sum of absolute differences. {py:meth}`.list.hamming_distance <batcher.plan.expr_ir.namespaces.collections._ListNamespace.hamming_distance>` handles binary or quantized embeddings, where each element is `0`, `1` or a small integer. It counts differing positions, which is far cheaper than a float metric and is what a binary vector index ranks by:

```python
# docs: skip
from batcher import col

# `bits` columns are quantized 0/1 embeddings; rank by how many bits differ.
nearest = docs.with_columns(dist=col("bits").list.hamming_distance(query_bits)).sort("dist")
```

:::{tip}
Normalize at ingest, then rank with `.list.dot`. On unit vectors the dot product ranks exactly as cosine does and skips two square roots per row.
:::

```python
unit = docs.with_columns(vec=col("vec").list.normalize())
ranked = unit.select("id", score=col("vec").list.dot(query)).sort("score", descending=True)
print(ranked.to_pydict())
# {'id': [1, 3, 2, 4], 'score': [1.0, 0.9938837346736189, 0.0, -1.0]}
```

Use `top_k` instead of `sort().limit(k)` when you only want the winners. It keeps a bounded heap instead of ordering the whole relation.

```python
print(
    docs.with_columns(dist=col("vec").list.cosine_distance(query))
    .top_k(2, by="dist", descending=False)
    .select("id")
    .to_pydict()
)
# {'id': [1, 3]}
```

## Filter first, then score

A vector distance is one more expression, so it composes with everything else. A metadata filter runs *before* the distance is computed, and the optimizer pushes it toward the scan, so a query scoped to one tenant scores only that tenant's rows.

```python
scoped = bt.from_pydict(
    {
        "id": [1, 2, 3],
        "tenant": ["a", "b", "a"],
        "vec": [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]],
    }
)
out = (
    scoped.filter(col("tenant") == "a")
    .with_columns(dist=col("vec").list.cosine_distance(query))
    .sort("dist")
)
print(out.select("id", "tenant").to_pydict())
# {'id': [1, 3], 'tenant': ['a', 'a']}
```

A standalone vector service tends to make this hard. A post-filter asks for 10, gets 10, and finds 9 belong to another tenant. A pre-filter can defeat the index. In the engine it's a predicate.

## Build an ANN index over Lance

Brute force costs O(rows x dims) per query. Past a few million vectors, or under real query concurrency, build an index: write the vectors to Lance, index the column, and search it.

```python
# docs: skip
from batcher.ml import build_vector_index, vector_search

build_vector_index("s3://bucket/vectors.lance", "embedding")
hits = vector_search(
    "s3://bucket/vectors.lance",
    query_vector,
    column="embedding",
    k=10,
    columns=["id", "title"],
    filter="tenant = 'a'",
)
top = hits.collect()  # k rows, nearest first, with a _distance column
```

`vector_search` returns a {py:class}`Dataset <batcher.Dataset>`, so the hits join, filter and aggregate like any other relation. `filter` is a SQL predicate applied with the search. `nprobes` trades latency for recall, because more probes search more of the index. `refine_factor` re-ranks `k * refine_factor` candidates with exact distances, buying back recall the approximate index lost. Both default to `None`, which leaves the choice to Lance. The index is IVF_PQ by default, and the column must be a `fixed_size_list` of floats, so embed with `output_type="fixed_size_list"`. Vector search needs the `batcher-engine[lance]` extra.

:::{warning}
An ANN index is approximate by construction. It can miss a true nearest neighbor and won't tell you it did. If your application can't tolerate that, as with a compliance lookup or a dedup key, use brute force over a filtered candidate set, not a higher `nprobes`.
:::

## Join on meaning

{py:meth}`ds.ml.similarity_join <batcher.api.dataset.ml.DatasetML.similarity_join>` matches every row of one dataset against the similar rows of another. That's entity resolution, such as a product catalog against a supplier feed or a CRM against a billing export, where the join key is "means the same thing" rather than "is the same string".

```python
catalog = bt.from_pydict({"sku": [1, 2], "vec": [[1.0, 0.0], [0.0, 1.0]]})
feed = bt.from_pydict({"item": [10, 11], "vec": [[0.99, 0.01], [0.0, 1.0]]})

matched = catalog.ml.similarity_join(
    feed, left_on="vec", threshold=0.9, left_key="sku", right_key="item"
)
print(matched.to_pydict())
# {'key_a': [1, 2], 'key_b': [10, 11],
#  'similarity': [0.999948988700964, 1.0]}
```

Comparing every pair is O(n x m) and infeasible at scale, so the join bands SimHash signatures to generate candidates and then scores the candidates *exactly*. Precision is guaranteed: no pair below `threshold` is ever returned. Recall is the dial, and more `bands` buys it with more candidates and more work. A pair with similarity `s`
survives banding with probability `1 - (1 - s^(num_bits/bands))^bands`.

:::{note}
Rows with a null or empty vector are dropped rather than banded. They have no direction, so they couldn't clear any threshold, and left in they would all collide into one enormous candidate bucket.
:::

## Choose an approach

The following table matches a corpus size and query pattern to the tool:

| Situation | Reach for |
| --- | --- |
| Candidates already narrowed by a filter, or a reranking pass | {py:meth}`.list.cosine_distance <batcher.plan.expr_ir.namespaces.collections._ListNamespace.cosine_distance>` + `top_k` |
| Millions of vectors, repeated queries, latency matters | Lance index + `vector_search` |
| Every row of A against the nearest rows of B | `ds.ml.similarity_join` |
| Exact duplicates or near-duplicate text, not vectors | `distinct` / `drop_near_duplicates` |

The exact path is the one that distributes, which is the reverse of what most readers expect. A global top-k is the top-k of the shards' top-ks, so exact search shards. `vector_search` is a single driver call into Lance against one dataset:

![The two ways to answer a nearest-neighbour query, and the inversion that reads backwards until you see it: the exact path is the one that shards. Both start from one ds.ml.embed call that computes a fixed_size_list column of float32 vectors, one per row, inside the engine. Exact search needs no build step: ds.ml.nearest_neighbors(q, k) covers cosine, l2, l1, hamming and dot, lowers to a distance expression plus a sort and a limit k over Rust kernels in one scan, and shards, because a global top-k is the top-k of the shards' top-ks. It is mergeable, so one core or a hundred machines run it unchanged, it costs a full scan per query, and it is the recommended path to a few million rows. Approximate search needs the vectors written to Lance first: build_vector_index builds an IVF_PQ index, and vector_search(uri, q, k) is a single driver call into Lance against one unsharded dataset, returning k rows and a _distance column. nprobes and refine_factor are passed through verbatim and default to None, so Lance decides; raising either probes more of the index, buying recall back with time. Nothing here measures the recall given up, because recall_at_k scores a set you hand it and no code wires it to the index, and ds.ml.embed's default output_type='tensor' is not indexable, so build_vector_index raises rather than mis-index it.](/_static/diagrams/vector_index_search.svg)

## See also

- {doc}`Embeddings </ml/retrieval/embeddings>`: producing and normalizing the vectors.
- {doc}`RAG </ml/retrieval/rag>`: retrieval feeding a generation step.
- {doc}`Expressions API </api/relational/expressions>`: the full `.list` vector method set.
- {doc}`Expression evaluation </architecture/deep-dives/query/expression-evaluation>`: why a vector distance is one more vectorized expression.
- {doc}`Sort internals </architecture/deep-dives/operators/sort-internals>`: the bounded heap behind `top_k`.
- {doc}`RAG index recipe </cookbook/ml/pipelines/text/rag-index>`: building and querying the index.
- {doc}`Distinct and dedup </user-guide/transform/rows/distinct-and-dedup>`: the exact and near-duplicate tools in the last row of the table.
- {doc}`ML API </api/models/ml>`: the `build_vector_index`, `vector_search` and {py:meth}`similarity_join <batcher.api.dataset.ml.DatasetML.similarity_join>` reference.

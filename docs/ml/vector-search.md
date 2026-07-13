# Vector search

Two very different things get called vector search. One is "score 50,000 candidate
vectors against a query and take the top 10", which is a `sort` and belongs in the
engine. The other is "find the nearest 10 of 500 million", which needs an ANN index,
because a brute-force scan will take minutes. Pick the wrong one and you either build an
index you never needed or wait for a linear scan you should not have run.

## Brute force, in the engine

When the vectors already ride in a column, score them with the `.list` distance
expressions. No index, no extra service, no data movement: it is a projection and a
top-n.

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

hits = (
    docs.with_columns(dist=col("vec").list.cosine_distance(query))
    .sort("dist")
    .head(2)
)
print(hits.select("id", "title").to_pydict())
# {'id': [1, 3], 'title': ['cats', 'kittens']}
```

`cosine_distance` is `1 - cosine_similarity`: 0 for identical direction, 1 for
orthogonal, 2 for opposite. It sorts ascending, so nearest comes first.
`.list.l2_distance` is Euclidean, and `.list.dot` is the raw inner product.

:::{tip}
Normalize at ingest, then rank with `.list.dot`. On unit vectors the dot product ranks
exactly as cosine does, and it skips two square roots per row.
:::

```python
unit = docs.with_columns(vec=col("vec").list.normalize())
ranked = unit.select("id", score=col("vec").list.dot(query)).sort("score", descending=True)
print(ranked.to_pydict())
# {'id': [1, 3, 2, 4], 'score': [1.0, 0.9938837346736189, 0.0, -1.0]}
```

Use `top_k` rather than `sort().head(k)` when you only want the winners: it keeps a
bounded heap instead of ordering the whole relation.

```python
print(docs.with_columns(dist=col("vec").list.cosine_distance(query))
      .top_k(2, by="dist", descending=False)
      .select("id").to_pydict())
# {'id': [1, 3]}
```

## Filter first, then score

The reason in-engine search is worth having is that a vector distance is just another
expression, so it composes with everything else. A metadata filter runs *before* the
distance is computed, and the optimizer pushes it into the scan, so a query scoped to one
tenant scores only that tenant's rows.

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

This is the thing a standalone vector database makes hard. There, a metadata filter is
either a post-filter (you asked for 10, got 10, and 9 belong to another tenant) or a
pre-filter that defeats the index. Here it is a predicate.

## ANN: an index over Lance

Brute force is O(rows × dims) per query. Past a few million vectors, or under any kind
of query concurrency, build an index. Write the vectors to Lance, index the column, and
search it.

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

`vector_search` returns a `Dataset`, so the hits join, filter, and aggregate like any
other relation. `nprobes` trades recall for latency (more probes, more of the index
searched); `refine_factor` re-scores an over-fetched candidate set with exact distances,
which buys back most of the recall an approximate index loses. Vector search needs the
`batcher-engine[lance]` extra.

:::{warning}
An ANN index is approximate by construction: it can miss a true nearest neighbour, and
it will not tell you that it did. If your application cannot tolerate that (a compliance
lookup, a dedup key), brute force over a filtered candidate set is the honest answer, not
a higher `nprobes`.
:::

## Joining on meaning

`ds.ml.similarity_join` is the other shape: not one query against a corpus, but every row
of one dataset matched against the nearest rows of another. This is entity resolution, a
product catalogue against a supplier feed or a CRM against a billing export, where the
join key is "means the same thing" rather than "is the same string".

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

Comparing every pair is O(n × m) and impossible at scale, so this bands SimHash
signatures to generate candidates and then scores the candidates *exactly*. Precision is
therefore guaranteed: no pair below `threshold` is ever returned. Recall is the dial, and
more `bands` buys it with more candidates and more work. A pair with similarity `s`
survives banding with probability `1 - (1 - s^(num_bits/bands))^bands`.

:::{note}
Rows with a null or empty vector are dropped rather than banded. They have no direction,
so they could not clear any threshold anyway, and left in, they would all collide into
one enormous candidate bucket.
:::

## Which one

| Situation | Reach for |
| --- | --- |
| Candidates already narrowed by a filter, or a reranking pass | `.list.cosine_distance` + `top_k` |
| Millions of vectors, repeated queries, latency matters | Lance index + `vector_search` |
| Every row of A against the nearest rows of B | `ds.ml.similarity_join` |
| Exact duplicates or near-duplicate text, not vectors | `distinct` / `drop_near_duplicates` |

## See also

- [Embeddings](embeddings.md): producing and normalizing the vectors.
- [RAG](rag.md): retrieval feeding a generation step.
- [Expressions API](../api/expressions.md): the full `.list` vector method set.
- [Expression evaluation](../deep-dives/expression-evaluation.md): why a vector distance
  is just another vectorized expression, and what that buys.
- [Sort internals](../deep-dives/sort-internals.md): the bounded heap behind `top_k`.
- [RAG index recipe](../examples/ml/rag-index.md): building and querying the index.
- [Distinct and dedup](../user-guide/distinct-and-dedup.md): the exact and near-duplicate
  tools the last row of that table points at.
- [ML API](../api/ml.md): the `build_vector_index`, `vector_search`, and
  `similarity_join` reference.

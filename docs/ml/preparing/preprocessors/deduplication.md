# Deduplication and matching

This page covers removing near-duplicate rows and joining rows that mean the same thing
without sharing a key. Both run before a model sees the data. Both are relational plans the
engine executes, not Python loops.

## Fuzzy deduplication

Exact deduplication is {py:meth}`distinct() <batcher.Dataset.distinct>`. On a web-scale training corpus it barely helps.
The duplicates there are the same article behind a different header, or the same page with
a changed timestamp, and no two of them are byte-identical.

{py:meth}`ds.ml.near_duplicates <batcher.api.dataset.ml.DatasetML.near_duplicates>` finds those pairs as `(key_a, key_b, jaccard)` rows.
{py:meth}`ds.ml.drop_near_duplicates <batcher.api.dataset.ml.DatasetML.drop_near_duplicates>` removes them, keeping one representative per cluster.

```python
import batcher as bt

docs = bt.from_pydict(
    {
        "text": [
            "the quick brown fox jumps over the lazy dog",
            "the quick brown fox jumps over the lazy dog!",  # near-duplicate
            "a treatise on the migratory habits of geese",
        ]
    }
)
print(docs.ml.drop_near_duplicates("text", threshold=0.7).count())
# 2
print(docs.distinct().count())  # exact dedup keeps all three
# 3
```

Here is the mechanism. `str.minhash` reduces each document to a signature of `num_perm`
integers over its character `ngram`-shingles, 128 and 5 by default. The fraction of
positions two signatures agree on, computed by `list.jaccard`, estimates the documents'
Jaccard similarity. LSH banding then turns the similarity search into an equi-join on a band
hash, and every candidate pair is verified against `threshold` before it is returned.

So banding costs recall and never precision. `bands` is the dial, 16 by default: more bands
means more candidates, more recall, and more work. The whole thing is a projection, an
`explode`, and some joins, so it runs wherever a join runs.

## Matching on meaning with similarity_join

MinHash answers "are these two documents made of the same words". It says nothing about
two rows that *mean* the same thing in different words. That is a question for embeddings.
{py:meth}`ds.ml.similarity_join <batcher.api.dataset.ml.DatasetML.similarity_join>` is the same two-stage recipe with the signature swapped:
{py:meth}`.list.simhash <batcher.plan.expr_ir.namespaces.collections._ListNamespace.simhash>` replaces `str.minhash`, and verification is the exact
`list.cosine_similarity` over the original vectors.

```python
import batcher as bt

catalog = bt.from_pydict({"sku": [1, 2], "v": [[1.0, 0.0], [0.0, 1.0]]})
feed = bt.from_pydict({"ref": [10], "v": [[1.0, 0.02]]})
pairs = catalog.ml.similarity_join(
    feed, left_on="v", threshold=0.9, left_key="sku", right_key="ref"
)
print(pairs.select("key_a", "key_b").to_pydict())
# {'key_a': [1], 'key_b': [10]}
```

Use it for entity resolution, such as matching a product catalog against a supplier feed
or a CRM against a billing system. It fits any join whose key is "means the same thing"
rather than "is the same string".

`simhash` is Charikar's random-hyperplane LSH. `num_bits` hyperplanes are drawn through
the origin and each bit records which side of one the vector falls on. Two vectors an
angle `theta` apart agree on each bit with probability `1 - theta/pi`, so the fraction of
agreeing bits estimates the angle. That is the vector-space counterpart of MinHash's
Jaccard estimate. The hyperplanes are derived by hashing `(seed, bit, dimension)` rather
than stored, so every partition and every machine draws the same ones and a signature
computed on one node is comparable with one computed on another.

As in fuzzy dedup, banding governs recall and never precision. No pair below `threshold`
is returned, but a pair above it can miss every band, and `bands` (8 by default here) is
the dial. A row whose vector is null or empty has no direction and can't clear any
threshold, so it is dropped rather than banded. Left in, every such row would collide with
every other and blow the candidate set up quadratically.

## See also

- {doc}`/user-guide/transform/rows/distinct-and-dedup`: exact and keyed deduplication.
- {doc}`/ml/retrieval/embeddings`: producing the vectors {py:meth}`similarity_join <batcher.api.dataset.ml.DatasetML.similarity_join>` matches on.
- {doc}`/ml/retrieval/vector-search`: nearest-neighbour search when you want the top matches rather than every pair above a threshold.
- {doc}`/ml/preparing/preprocessors/index`: the rest of the preprocessor family.

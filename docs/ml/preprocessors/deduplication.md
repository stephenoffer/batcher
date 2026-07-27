# Deduplication and matching

This page covers removing near-duplicate rows and joining rows that mean the same thing
without sharing a key. Both are preprocessing steps in the sense that matters: they run
before a model sees the data, and both are relational operations on the engine rather
than Python loops.

## Fuzzy deduplication

Exact deduplication is `distinct()`. On a web-scale training corpus it barely helps,
because the duplicates are the same article behind a different header, or the same page
with a changed timestamp. Removing *those* is the single biggest win in preprocessing an
LLM pretraining set.

`ds.ml.near_duplicates` finds the pairs, and `ds.ml.drop_near_duplicates` removes them,
keeping one representative per cluster.

```python
import batcher as bt

docs = bt.from_pydict({"text": [
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox jumps over the lazy dog!",   # near-duplicate
    "a treatise on the migratory habits of geese",
]})
print(docs.ml.drop_near_duplicates("text", threshold=0.7).count())
# 2
print(docs.distinct().count())   # exact dedup keeps all three
# 3
```

Under the hood, `str.minhash` reduces each document to a fixed-length signature whose
positional agreement rate, computed by `list.jaccard`, estimates the documents' Jaccard
similarity. LSH banding then turns the similarity join into an equi-join on a band hash.
Every returned pair is **verified** against the threshold, so banding only costs recall,
never precision. `bands` is the dial. More bands means more candidates, more recall, and
more work.

Both are ordinary relational plans built from a projection, an `explode`, and some joins,
so they run wherever a join runs.

## Matching on meaning with similarity_join

MinHash answers "are these two documents made of the same words". It says nothing about
two rows that *mean* the same thing in different words. That is a question for embeddings,
and `ds.ml.similarity_join` is the same two-stage recipe with the signature swapped:
`.list.simhash` replaces `str.minhash`, and the verification is the **exact**
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

This is entity resolution, matching a product catalog against a supplier feed or a CRM
against a billing system, and it is also retrieval over a corpus. It covers any join
whose key is "means the same thing" rather than "is the same string".

`simhash` is Charikar's random-hyperplane LSH. `num_bits` hyperplanes are drawn through
the origin and each bit records which side of one the vector falls on. Two vectors an
angle `theta` apart agree on each bit with probability `1 - theta/pi`, so the fraction of
agreeing bits estimates the angle. That is the vector-space counterpart of MinHash's
Jaccard estimate. The hyperplanes are derived by hashing `(seed, bit, dimension)` rather
than stored, so every partition and every machine draws the same ones and a signature
computed on one node is comparable with one computed on another.

Exactly as in fuzzy dedup, banding governs **recall, never precision**. No pair below
`threshold` is ever returned, but a pair above it can miss every band. `bands` is the
dial. Rows whose vector is null or empty have no direction, cannot clear any threshold,
and are dropped rather than banded. Left in, they would all collide and blow the
candidate set up quadratically.

## See also

:::{seealso}
- {doc}`../../user-guide/distinct-and-dedup`: exact and keyed deduplication.
- {doc}`../embeddings`: producing the vectors `similarity_join` matches on.
- {doc}`index`: the rest of the preprocessor family.
:::

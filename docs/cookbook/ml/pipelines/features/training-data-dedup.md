# Training-data dedup

Exact deduplication on a web crawl barely moves the number. The duplicates are not
byte-identical: they are the same article behind a different header, the same product page
with a changed timestamp, the same README vendored into forty repositories. `distinct()`
keeps all of them, the model sees the same text a dozen times, and the memorization it
buys shows up as a suspiciously good held-out score.

Fuzzy dedup is the highest-leverage pass in preparing a pretraining corpus, and it is two
method calls.

## Exact dedup does not find them

```python
import batcher as bt

docs = bt.from_pydict(
    {
        "doc_id": [1, 2, 3, 4],
        "text": [
            "the quick brown fox jumps over the lazy dog",
            "the quick brown fox jumps over the lazy dog!",  # a header changed
            "The Quick Brown Fox Jumps Over The Lazy Dog",  # a re-render
            "a treatise on the migratory habits of geese",
        ],
    }
)
```

Four documents, two of which are the same document. Run each dedup over them:

::::{tab-set}
:::{tab-item} distinct()

```python
print(docs.distinct().count())
# 4
```

Nothing was removed. Byte equality is the wrong equality for a crawl.
:::

:::{tab-item} drop_near_duplicates()

```python
print(docs.ml.drop_near_duplicates("text", threshold=0.7, key="doc_id").count())
# 3
```

The punctuation-only twin is gone.
:::
::::

`drop_near_duplicates` keeps one representative per cluster (the row minimal among its
near-duplicates) and drops the rest. The case-changed row survives here because character
shingles are case-sensitive; lowercase the column first if you want it collapsed too. That
is a choice you make, not a default that quietly makes it for you.

## See the pairs before you delete anything

:::{tip}
`near_duplicates` returns the matched pairs with their estimated Jaccard similarity, so you
can look at what a threshold is about to remove before you remove it. Run this once on a
sample. A threshold that looks reasonable in a paper often eats an entire legitimate
category of your corpus.
:::

```python
pairs = docs.ml.near_duplicates("text", threshold=0.7, key="doc_id")
out = pairs.to_pydict()
print(out["key_a"], out["key_b"], [round(j, 2) for j in out["jaccard"]])
# [1] [2] [1.0]
```

:::{dropdown} How it finds them: MinHash signatures, LSH bands, and an exact verify

MinHash reduces each document to a fixed-length signature whose positional
agreement estimates Jaccard similarity over character n-gram shingles, and LSH banding turns
the similarity join into an equi-join on a band hash. Every candidate pair is then
**verified** against the threshold, so banding costs recall, never precision. No pair below
`threshold` is ever returned, but a similar pair can miss every band and be missed.
:::

The dials, in the order you should reach for them:

| Knob | Effect |
| --- | --- |
| `threshold` | The similarity a pair must clear. 0.8 is the usual pretraining setting. |
| `ngram` | Shingle width in characters. Larger is stricter; short documents need a smaller value. |
| `bands` | Recall/cost. More bands, more candidates, more recall, more work. |
| `num_perm` | Signature length. Standard error of the estimate is `1 / sqrt(num_perm)`. |

`bands` must divide `num_perm`, and the S-curve's knee sits near
`(1 / bands) ** (bands / num_perm)`. If you are missing duplicates you know are there,
raise `bands` before you lower `threshold`. Lowering the threshold changes what a duplicate
*is*, which is a different decision.

## Short documents need a smaller shingle

The default `ngram=5` shingles five characters at a time. On a corpus of one-line titles or
search queries, a document may be barely longer than a shingle, and everything looks
different from everything. Drop `ngram` when the documents are short.

```python
titles = bt.from_pydict(
    {
        "doc_id": [1, 2, 3],
        "text": ["red running shoes", "red running shoe", "blue winter coat"],
    }
)
print(titles.ml.near_duplicates("text", threshold=0.7, ngram=3, key="doc_id").count())
# 1
```

## Matching a short field against a reference value

MinHash/LSH clusters a *column* against itself. When you instead need to score each row
against one **known** string, reach for the edit metrics on `.str`. That covers deduping a
name column against a canonical spelling, or resolving records to a reference list.
`.str.jaro_similarity` and `.str.jaro_winkler_similarity` return a `[0, 1]` score, and
Jaro-Winkler weights a shared prefix, which is what you want for names. `.str.levenshtein`
gives the raw edit distance, and `.str.damerau_levenshtein` counts a swapped-letter typo as
a single edit:

```python
from batcher import col

people = bt.from_pydict({"name": ["Jonathan", "Johnathan", "Jon", "Michael"]})
scored = people.with_columns(
    sim=col("name").str.jaro_winkler_similarity("Jonathon"),
).filter(col("sim") > 0.8)  # keep the likely matches to the canonical spelling
```

## Check the test set for contamination

:::{important}
The dedup pass most people skip: your benchmark rows are on the internet, so they are in
your crawl. Any near-duplicate of a test document sitting in the training corpus makes the
evaluation meaningless, and it will not show up as a byte-identical match.
:::

Hunt near-duplicates across the union of the two corpora, then drop the training rows that
matched a test row.

```python
from batcher import col

train = bt.from_pydict(
    {
        "doc_id": [10, 11, 12],
        "text": [
            "the capital of France is Paris and it sits on the Seine",
            "photosynthesis converts light energy into chemical energy",
            "an unrelated document about hydraulic pumps and valves",
        ],
    }
)
test = bt.from_pydict(
    {"doc_id": [90], "text": ["The capital of France is Paris, and it sits on the Seine."]}
)

pairs = train.union(test).ml.near_duplicates("text", threshold=0.6, key="doc_id")

# A pair is contamination when one side of it is a test row.
test_ids = test.select(col("doc_id").alias("key_b"))
leaked = pairs.join(test_ids, on="key_b").select(col("key_a").alias("doc_id"))
clean = train.join(leaked, on="doc_id", how="anti")

print(leaked.to_pydict()["doc_id"])
# [10]
print(sorted(clean.to_pydict()["doc_id"]))
# [11, 12]
```

`near_duplicates` emits pairs with `key_a < key_b`, which is why the test IDs join on
`key_b` here: test rows were numbered above the training rows on purpose. Number them the
other way and you join on `key_a`. Do this once, before training, and the eval number you
report is one you can defend.

## Where it runs

Both calls lower to ordinary relational plans (a projection, an `explode`, a join, a filter)
so they run wherever a join runs: multi-core, distributed, spilled. There is no
special dedup engine and no driver-side set of hashes. That matters at the scale where
this pass is worth doing, which is the only scale where it is worth doing.

For dedup on *meaning* rather than on words (two descriptions of the same product, written
independently), `ds.ml.similarity_join` is the same two-stage recipe over embeddings:
SimHash bands the candidates, exact cosine verifies them. See
{doc}`preprocessors </ml/preparing/preprocessors/index>`.

## See also

- {doc}`Train/test split </cookbook/ml/pipelines/features/train-test-split>`: dedup first, then split, in that order.
- {doc}`Text embeddings </cookbook/ml/pipelines/text/text-embeddings>`: the embedding half of `similarity_join`.
- {doc}`Preprocessors </ml/preparing/preprocessors/index>`: MinHash, SimHash, and the LSH banding math.
- {doc}`Embeddings </ml/retrieval/embeddings>`: the encoder that makes semantic dedup possible.
- {doc}`Distinct and dedup </user-guide/transform/rows/distinct-and-dedup>`: the exact-match surface, and
  when it is enough.
- {doc}`Deduplication </cookbook/data-engineering/maintenance/deduplication>`: the same problem on an event stream,
  where the key is known.
- {doc}`ML API reference </api/models/ml>`: `near_duplicates`, `drop_near_duplicates`,
  `similarity_join`.
- {doc}`Join algorithms </architecture/deep-dives/operators/join-algorithms>`: the equi-join the LSH banding
  lowers to, and why it distributes.

# Text vectorization

This page describes how to turn a text column into numeric features with Batcher's
bag-of-words vectorizers, and how to choose between them.

A bag of words represents a document as counts over a fixed feature space, ignoring word
order. It's a weak model of language and a strong set of features: on a labelled corpus of
short documents, a linear model over TF-IDF features is still competitive with an embedding
model, trains in seconds, and produces coefficients you can read.

## What the vectorizers produce

Batcher has no sparse-vector column type, so a vectorized document is written as two
aligned list columns. For `output_column="features"`, the default, `transform` appends:

`features_indices`
: The feature positions this document uses, ascending.

`features_values`
: The value at each of those positions, element-for-element aligned with the indices.

Together they're one row of a compressed sparse row matrix. Pass `dense=True` instead and
you get a single fixed-width `List<Float64>` column named `features`, which is what a
trainer reading through {py:func}`iter_torch_batches <batcher.ml.iter_torch_batches>` wants.
Dense is only sensible for a small vocabulary, because every row then carries one value per
term whether or not the document uses it.

```python
import batcher as bt
from batcher.ml.preprocessors import CountVectorizer

docs = bt.from_pydict({"text": ["red car red", "blue bike", "red bike blue car"]})
counts = CountVectorizer("text").fit(docs)
print(counts.vocabulary_)
# ['bike', 'blue', 'car', 'red']

out = counts.transform(docs).to_pydict()
print(out["features_indices"])
# [[2, 3], [0, 1], [0, 1, 2, 3]]
print(out["features_values"])
# [[1.0, 2.0], [1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
```

## Choosing a vectorizer

Pick by whether you need a readable vocabulary and whether you can afford a pass over the
corpus to learn one:

| Vectorizer | Learns a vocabulary | Use it when |
|---|---|---|
| {py:class}`CountVectorizer <batcher.ml.preprocessors.CountVectorizer>` | Yes | You want raw counts, or a feature space you can name. |
| {py:class}`TfidfVectorizer <batcher.ml.preprocessors.TfidfVectorizer>` | Yes | Almost always, for a linear text classifier. |
| {py:class}`HashingVectorizer <batcher.ml.preprocessors.HashingVectorizer>` | No | The corpus is a stream, or you want no fitted state to ship. |

## TF-IDF weighting

A raw count says a document uses *the* forty times, which is true and carries no signal.
TF-IDF divides that out: a term's weight rises with its frequency inside the document and
falls with the number of documents that contain it, so what survives is what distinguishes
this document from the rest.

```python
from batcher.ml.preprocessors import TfidfVectorizer

tfidf = TfidfVectorizer("text").fit(docs)
print([round(v, 3) for v in tfidf.idf_])
# [1.288, 1.288, 1.288, 1.288]

weighted = tfidf.transform(docs).to_pydict()
print([round(v, 3) for v in weighted["features_values"][0]])
# [0.447, 0.894]
```

Rows are scaled to unit L2 norm by default, so document length doesn't dominate the
distance between two documents. Pass `norm="l1"` or `norm=None` to change that, and
`sublinear_tf=True` to replace a count `n` with `1 + log(n)` when a term appearing fifty
times shouldn't count fifty times as much as one appearing once.

The IDF formula is `ln((1 + n) / (1 + df)) + 1`, matching scikit-learn's smoothed default.
The smoothing is what makes a term the transform set has and the fit set doesn't safe to
weight rather than a division by zero.

## Controlling the vocabulary

Four settings decide which terms become features. Use them in this order:

1. `stop_words` drops words by name. Pass `"english"` for the built-in list, or your own
   sequence for a domain that has its own filler words.
1. `min_df` drops a term that too few documents use, which is where typos and one-off
   identifiers live. An int is an absolute document count; a float is a fraction of the
   corpus.
1. `max_df` drops a term that too many documents use, which catches corpus-specific
   boilerplate that a general stop-word list can't know about.
1. `max_features` keeps only the most frequent terms that survive the other three.

```python
tuned = TfidfVectorizer("text", stop_words="english", min_df=2, max_features=1000)
print(len(tuned.fit(docs).vocabulary_))
# 4
```

Prefer `min_df` and `max_df` over a stop-word list where you can. They're measured against
the corpus in front of you, and a list that suits news text will discard signal in clinical
notes.

## N-grams

`ngram_range` adds contiguous runs of tokens as their own terms, which is how a bag of
words recovers a little of the word order it throws away. `(1, 2)` keeps single words and
adjacent pairs:

```python
bigrams = CountVectorizer("text", ngram_range=(1, 2)).fit(docs)
print(bigrams.vocabulary_)
# ['bike', 'bike blue', 'blue', 'blue bike', 'blue car', 'car', 'car red', 'red', 'red bike', 'red car']
```

The vocabulary grows quickly, so pair a wide `ngram_range` with `min_df` or `max_features`.

## Hashing, for streams and for serving

{py:class}`HashingVectorizer <batcher.ml.preprocessors.HashingVectorizer>` decides a term's
feature index as `abs(hash(term)) % n_features`. Nothing is learned, so `fit` does nothing
and exists only so it composes into a
{py:class}`Chain <batcher.ml.preprocessors.Chain>` like every other preprocessor.

```python
from batcher.ml.preprocessors import HashingVectorizer

hashed = HashingVectorizer("text", n_features=2**18).fit_transform(docs)
print(len(hashed.to_pydict()["features_indices"][0]))
# 2
```

That buys three things a learned vocabulary can't. There's no fit pass, so it works on an
unbounded stream. There's no state to ship to a serving process, so training and serving
cannot disagree. And the feature space is fixed in advance, so memory is known before you
start.

What it costs is collisions and interpretability. Two terms can land on one feature, and no
feature can be named. Make `n_features` generous — a few hundred thousand is ordinary — and
a linear model tolerates the rest.

## Fitting on the training split only

A vectorizer is fitted state, so it belongs to the training split. Fit on the whole frame
and the held-out documents contribute to the vocabulary and to every IDF weight, which
inflates your validation score by an amount you can't measure.

```python
train, test = docs.ml.train_test_split(0.3, seed=0)
vectorizer = TfidfVectorizer("text").fit(train)
train_features = vectorizer.transform(train)
test_features = vectorizer.transform(test)
print(train_features.count() + test_features.count())
# 3
```

A term the test split uses and the training split didn't is ignored rather than bucketed,
matching scikit-learn: the fitted vocabulary *is* the feature space, and a document using
new words simply sets fewer features.

## How the fit scales

`fit` is a relational aggregate, not a dictionary built on the driver. The term column is
an ordinary expression, so the vocabulary comes from an `explode` into a `group_by`, which
is mergeable — the same fit runs on one core, on every core, or across a cluster, and
spills rather than failing on a corpus larger than memory.

The result is materialized on the driver and broadcast to every worker, though, so the
vocabulary itself is bounded. Past `max_vocabulary` the fit fails and names the ways out
rather than quietly building a plan nobody can run. Set `max_features`, raise `min_df`, or
switch to `HashingVectorizer`.

## Requirements and limitations

- A null document is read as an empty one. It keeps its row and produces no features,
  rather than being dropped from the output.
- Tokenization is regex-based and language-agnostic. There's no stemming, lemmatization, or
  subword tokenization; for those, use
  {py:class}`Tokenizer <batcher.ml.preprocessors.Tokenizer>` with a real tokenizer, or an
  embedding model.
- `max_features` breaks a frequency tie alphabetically so the fitted vocabulary is
  reproducible. scikit-learn leaves that tie to an unstable sort, so the two libraries can
  legitimately keep different terms from a tied group.
- `HashingVectorizer` uses a different hash function from scikit-learn's, so the two agree
  on a document's *values* but not on which index carries them.

## See also

- {doc}`encoding` for categorical columns, including the hashing trick applied to a single
  categorical value rather than a document.
- {doc}`pipelines` for composing a vectorizer with the rest of a feature pipeline.
- {doc}`/api/models/preprocessors` for the full reference.

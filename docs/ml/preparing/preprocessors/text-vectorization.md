# Text vectorization

This page covers the three vectorizers that turn a text column into bag-of-words features,
and how to pick between them.

A bag of words represents a document as counts over a fixed feature space, ignoring word
order. It's a weak model of language and a strong set of features. On a labelled corpus of
short documents, a linear model over TF-IDF features is still competitive with an embedding
model, and its coefficients can be read.

## What the vectorizers produce

Batcher has no sparse-vector column type, so a vectorized document is written as two
aligned list columns. For `output_column="features"`, the default, `transform` appends:

`features_indices`
: The feature positions this document uses, ascending.

`features_values`
: The value at each of those positions, element-for-element aligned with the indices.

Together they're one row of a compressed sparse row matrix. Pass `dense=True` for a single
fixed-width `List<Float64>` column named `features` instead, which is what a trainer reading
through {py:func}`iter_torch_batches <batcher.ml.iter_torch_batches>` wants. Keep dense for a
small vocabulary, because every row then carries one value per term whether the document
uses it or not.

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

A raw count says a document uses *the* forty times. True, and no signal at all. TF-IDF
divides that out. A term's weight rises with its frequency inside the document and falls
with the number of documents containing it, so the heavy terms are the ones that set this
document apart.

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
The `+ 1` terms keep a zero document frequency from dividing by zero.

## Controlling the vocabulary

Four settings decide which terms become features. Consider them in the following order:

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
the corpus in front of you, while a list that suits news text can discard signal in clinical
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

{py:class}`HashingVectorizer <batcher.ml.preprocessors.HashingVectorizer>` computes a term's
feature index as `str.hash64().abs() % n_features`, evaluated in the engine. The hash is the
engine's own, not Python's `hash()`, which varies between processes and would put training
and serving on different feature spaces. Nothing is learned. `fit` exists only so the step
composes into a {py:class}`Chain <batcher.ml.preprocessors.Chain>`.

```python
from batcher.ml.preprocessors import HashingVectorizer

hashed = HashingVectorizer("text", n_features=2**18).fit_transform(docs)
print(len(hashed.to_pydict()["features_indices"][0]))
# 2
```

With no fit pass, the vectorizer works on an unbounded stream. With no state to ship,
training and serving can't disagree. The feature space is fixed before you start, and so is
the memory.

You pay in collisions and interpretability. Two terms can land on one feature, and no
feature has a name. Make `n_features` generous. A few hundred thousand is ordinary.

## Fitting on the training split only

A vectorizer is fitted state, so fit it on the training split. Fit on the whole frame and
the held-out documents shape the vocabulary and every IDF weight, which inflates your
validation score by an amount you can't measure.

```python
train, test = docs.ml.train_test_split(0.3, seed=0)
vectorizer = TfidfVectorizer("text").fit(train)
train_features = vectorizer.transform(train)
test_features = vectorizer.transform(test)
print(train_features.count() + test_features.count())
# 3
```

A term the test split uses and the training split didn't is ignored, not bucketed, as in
scikit-learn. The fitted vocabulary is the feature space, so a document full of new words
sets fewer features.

## How the fit scales

`fit` is a relational aggregate, not a dictionary built on the driver. The term column is
an ordinary expression, so the vocabulary comes from an `explode` into a `group_by`. That's
mergeable, so the same fit runs on one core, on every core, or across a cluster, and spills
rather than failing on a corpus larger than memory.

The resulting vocabulary is held on the driver and broadcast to every worker, so it is
bounded. Past `max_vocabulary`, one million terms by default, the fit raises and names the
ways out: set `max_features`, raise `min_df`, or switch to `HashingVectorizer`.

## Requirements and limitations

- A null document is read as an empty one. It keeps its row and produces no features,
  rather than being dropped from the output.
- Tokenization is regex-based and language-agnostic. There's no stemming, lemmatization, or
  subword tokenization. For those, use
  {py:class}`Tokenizer <batcher.ml.preprocessors.Tokenizer>` with a real tokenizer, or an
  embedding model.
- `max_features` breaks a frequency tie alphabetically so the fitted vocabulary is
  reproducible. scikit-learn leaves that tie to an unstable sort, so the two libraries can
  legitimately keep different terms from a tied group.
- `HashingVectorizer` uses a different hash function from scikit-learn's, so the two agree
  on a document's *values* but not on which index carries them.


## Where a vectorizer puts its output

A text vectorizer produces a sparse result, which does not fit one column. Both
{py:class}`CountVectorizer <batcher.ml.CountVectorizer>` and
{py:class}`HashingVectorizer <batcher.ml.HashingVectorizer>` write two list columns
instead: one of positions and one of the values at those positions.

`indices_column` and `values_column` name them, derived from the `output_column` you chose,
so downstream code reads the names rather than reconstructing the convention.

```python
from batcher.ml.preprocessors import CountVectorizer

vectorizer = CountVectorizer("review")
print(vectorizer.indices_column, vectorizer.values_column)

renamed = CountVectorizer("review", output_column="bag")
print(renamed.indices_column, renamed.values_column)
```

Reading the names off the instance is what keeps a feature pipeline working when someone
changes `output_column`, and it is the only way a generic step can find the pair.

## See also

- {doc}`/ml/preparing/preprocessors/encoding`: categorical columns, including the hashing trick applied to one categorical value rather than a document.
- {doc}`/ml/preparing/preprocessors/feature-generation`: `TextStatFeaturizer`, for cheap surface statistics of the same text.
- {doc}`/ml/preparing/tokenization`: subword tokenization for a language model.
- {doc}`/ml/retrieval/embeddings`: dense vectors when a bag of words plateaus.
- {doc}`/ml/preparing/preprocessors/pipelines`: composing a vectorizer with the rest of a feature pipeline.
- {doc}`/api/models/preprocessors`: the full reference.

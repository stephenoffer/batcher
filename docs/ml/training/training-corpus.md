# Preparing a training corpus

This page covers the three steps between having text and being able to train on it: mixing
several sources at the ratio you meant, dropping the documents that are not prose, and removing
the evaluation data that leaked in. Each one is a data bug that presents as a model bug when
it's skipped, which is why they're worth doing before the expensive part.

Everything here is in `batcher.ml` and built on the public `Dataset` API, so each step is a plan
the optimizer sees whole rather than a pass over materialized rows.

## Mixing sources at declared weights

A training run is almost never one dataset. It's code at 15%, web text at 50%, books at 20%,
and a small high-quality set at the end. Those ratios are a hyperparameter.

Concatenating the sources gives you the ratio of their *sizes* instead, and that failure is
silent: the run trains, the loss falls, and the model is dominated by whichever corpus happened
to be biggest. `bt.ml.mix_corpora` samples each source to hit the weights you declared.

```python
import batcher as bt
from batcher.ml import mix_corpora

web = bt.from_pydict({"text": ["a web document"] * 800})
code = bt.from_pydict({"text": ["def f(): pass"] * 800})

mixed, report = mix_corpora(
    {"web": web, "code": code},
    {"web": 3, "code": 1},
    total_rows=400,
)
print(report.realized_weights)
# {'web': 0.75, 'code': 0.25}
```

Weights are normalized, so `{"web": 3, "code": 1}` and `{"web": 0.75, "code": 0.25}` mean the
same thing. Every row is tagged with a `source` column, which is what lets you read a loss or a
quality metric per corpus afterwards.

Read the `MixtureReport` rather than assuming the mixture matched its configuration. A source
with fewer rows than its weight calls for cannot fill its share, and `mix_corpora` will not
repeat rows to make it: repeating changes the effective number of epochs for that source, which
is a decision to make on purpose.

```python
big = bt.from_pydict({"text": ["b"] * 100})
tiny = bt.from_pydict({"text": ["t"] * 2})
_, report = mix_corpora({"big": big, "tiny": tiny}, {"big": 0.5, "tiny": 0.5}, total_rows=100)
print(report.shortfalls)
# {'tiny': 48}
```

Leaving `total_rows` unset picks the largest mixture no source has to repeat itself to fill.

## Filtering out what isn't prose

Navigation bars, link farms, tables of figures, and truncated boilerplate make up a large share
of scraped text, and none of it teaches a model much about language. `bt.ml.quality_filter`
applies the well-known web-corpus heuristics: a length floor, a word-shape check, and character
ratio caps on punctuation, digits, and non-ASCII.

```python
from batcher.ml import QualityThresholds, quality_filter, quality_report

docs = bt.from_pydict(
    {
        "text": [
            "A real sentence about something, written out properly.",
            "buy now!!!",
            "1234 5678 9012 3456",
        ]
    }
)
print(quality_filter(docs, "text", QualityThresholds(min_words=3)).to_pydict()["text"])
```

Run `quality_report` on a sample first. It gives the keep rate of each rule independently, and
the rule that removes the most is usually the one whose threshold is wrong for your corpus
rather than the one finding the most junk. A code dataset is mostly symbols and a financial one
mostly digits; prose thresholds delete either.

```python
print(quality_report(docs, "text", QualityThresholds(min_words=3)))
```

The per-rule numbers don't sum to `all`, because a bad document usually fails several rules at
once. A null document fails everything, since a row the filter can't read isn't a row to train
on.

:::{warning}
`max_non_ascii_ratio` defaults to `1.0`, which disables that rule. Lowering it is a crude
language gate for a corpus meant to be English, and applying it to a multilingual corpus deletes
most of the corpus.
:::

## Removing evaluation data

A benchmark score is only evidence if the model hasn't seen the answers. Test sets leak into web
crawls constantly: questions get quoted in blog posts, datasets get mirrored, and papers
reproduce their own examples. A corpus assembled from the web contains some of every public
benchmark, and the resulting score measures memorization while being reported as ability.

`bt.ml.decontaminate` drops the training documents that share a verbatim span with an evaluation
set.

```python
from batcher.ml import contamination_rate, decontaminate

train = bt.from_pydict(
    {"text": ["what is the capital of france", "an entirely unrelated document here"]}
)
evals = bt.from_pydict({"text": ["what is the capital of france"]})

print(contamination_rate(train, "text", evals, n=4))
print(decontaminate(train, "text", evals, n=4).to_pydict()["text"])
```

`n` is the whole judgement. The default of 13 tokens is where the published pipelines settled:
long enough that an accidental match between unrelated documents is vanishingly unlikely, short
enough to catch a quoted question. Shorter spans start matching ordinary English, and then you
are deleting real training data to remove contamination that was never there.

Measure with `contamination_rate` before removing. A small rate is contamination to drop. A
large one usually means `n` is too short for your corpus, not that your corpus is ruined.

Matching uses the same normalization the grounding metrics use, so a requoted question with
different casing or punctuation still matches. It's verbatim overlap, so a paraphrase of a test
question does not: this removes copies, not leakage in general.

The check runs as a join rather than a scan per document, so it scales the way every other join
here does instead of being quadratic in the corpus size.

## Ordering so a batch is not mostly padding

A training batch is a rectangle. Every sequence in it is padded to the longest one, and every
padded position costs the same attention arithmetic as a real token while teaching the model
nothing. On a corpus whose lengths vary — which is every natural-language corpus — a randomly
ordered batch pairs a 40-token example with a 2,000-token one and spends most of its compute on
padding.

Sorting the whole corpus by length fixes that and breaks the training: the model then sees all
the short examples first and all the long ones last, which is a curriculum nobody chose.

`bt.ml.length_grouped_order` is the standard compromise. It shuffles, cuts the shuffled stream
into megabatches, and sorts by length only *within* a megabatch, so batches are
length-homogeneous while the epoch order stays random.

```python
from batcher.ml import length_grouped_order, padding_waste

lengths = [1, 9, 2, 8, 3, 7, 4, 6]
corpus = bt.from_pydict({"tokens": [[1] * n for n in lengths]})

print(round(padding_waste(corpus, "tokens", batch_size=2), 3))
grouped = length_grouped_order(corpus, "tokens", batch_size=2)
print(round(padding_waste(grouped, "tokens", batch_size=2), 3))
# 0.333
# 0.091
```

Measure with `bt.ml.padding_waste` before deciding it is worth it. The benefit depends entirely
on the corpus's length distribution: on a uniform-length corpus it is zero either way, and the
ordering is complexity for nothing.

`megabatch_factor` is the dial. Larger means more length-homogeneous batches and less padding,
and also a longer run of similar-length examples; 20 to 100 is the usual range. The length is
read from the column's type, so the same call works on a text column before tokenization and on
a token-id column after.

Consume the result in order. The ordering is the product, and re-shuffling downstream throws it
away.

## The order to run them in

Decontaminate last. Mixing and filtering both change which documents are present, and a
contamination check is only meaningful over the corpus you are actually going to train on.

Filter before mixing when the sources differ in quality, so a noisy source does not consume its
weight with documents you were going to drop anyway.

Order last of all, after tokenization, since it is the shape of the batches you will actually
train on.

## See also

- {doc}`/ml/preparing/tokenization`: sequence packing and the token-id column, the next step after this page.
- {doc}`/ml/training/data-loaders`: feeding the prepared corpus to a training loop.
- {doc}`/ml/retrieval/llm-evaluation`: scoring what the trained model produces.
- {doc}`/user-guide/transform/distinct-and-dedup`: near-duplicate removal, which belongs in the same
  pipeline.

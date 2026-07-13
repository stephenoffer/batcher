# Tokenization

Tokenizing in the training loop is the classic way to leave a GPU idle. The tokenizer is
CPU work, it is embarrassingly parallel, and it produces a column, so it belongs in the
data pipeline: run once, written out, never run again. Do it as an engine stage and the
loop reads token ids straight off disk.

Which tool depends on what the text actually is:

| The column holds | Reach for | What you get |
| --- | --- | --- |
| Documents to feed a model | `Tokenizer` wrapping a fast tokenizer class | a `List<Int64>` column of token ids |
| Documents for causal-LM pretraining | `pack_sequences` over that token column | dense fixed-length blocks, nothing padded |
| A category, not a document | `LabelEncoder` | one integer per distinct value, learned on the train split |

## The Tokenizer preprocessor

`Tokenizer(column, tokenizer, output_column=None)` applies any callable from a string to
a list. It is a `Preprocessor`, so it has the standard `fit` / `transform` /
`fit_transform` contract, and it is stateless. There is nothing to learn, so `fit` is a
no-op and `transform` runs anywhere.

::::{tab-set}
:::{tab-item} A whitespace split

The toy version, to see the contract.

```python
import batcher as bt
from batcher.ml import Tokenizer

docs = bt.from_pydict({"id": [1, 2], "text": ["hello world", "one two three"]})

tok = Tokenizer("text", lambda s: s.split(), output_column="tokens")
tokenized = tok.fit_transform(docs)
print(tokenized.to_pydict())
# {'id': [1, 2], 'text': ['hello world', 'one two three'],
#  'tokens': [['hello', 'world'], ['one', 'two', 'three']]}
```

:::

:::{tab-item} A HuggingFace tokenizer

The real thing, as a class, with the fast (Rust) implementation.

```python
# docs: skip
import pyarrow as pa

import batcher as bt


class HFTokenizer:
    def __init__(self, model="bert-base-uncased", max_length=512):
        self.model = model
        self.max_length = max_length
        self._tok = None

    def __call__(self, batch):
        if self._tok is None:  # loaded once per worker, not per batch
            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.model, use_fast=True)
        encoded = self._tok(
            batch.column("text").to_pylist(),
            truncation=True,
            max_length=self.max_length,
        )
        return batch.append_column("input_ids", pa.array(encoded["input_ids"]))


ds = bt.read.parquet("s3://bucket/corpus.parquet")
tokens = ds.map_batches(
    HFTokenizer(),
    input_columns=["id", "text"],
    output_columns=["id", "text", "input_ids"],
    batch_size=1024,
)
tokens.write.parquet("s3://bucket/tokens.parquet")
```

:::
::::

:::{warning}
A real tokenizer has to be constructed **once per worker**, which means a class, not a
lambda. A lambda closing over a tokenizer object gets pickled to every worker, and a slow
tokenizer re-created per batch will be the bottleneck of the whole job — the GPU you were
trying to feed ends up waiting on the CPU stage that was supposed to feed it.
:::

`num_workers` defaults to `"auto"`, which fans the calls across every local core. A fast
tokenizer releases the GIL, so threads are the right pool; a pure-Python tokenizer needs
`multiprocessing=True` to get real parallelism.

## Token ids are a list column

The output is `List<Int64>`, an ordinary Arrow column. So the whole expression surface
applies, and length statistics are one aggregate rather than a pass over the corpus in
Python.

```python
from batcher import col

lengths = tokenized.select(n=col("tokens").list.len())
print(lengths.to_pydict())
# {'n': [2, 3]}

print(lengths.describe().to_pydict()["n"][:4])
# [2.0, 0.0, 2.5, 0.7071067811865476]
```

:::{important}
Look at that distribution before you set `max_length`. Truncation is silent: nothing
raises, and a corpus where 30% of the documents lost their tail trains perfectly happily
on the first 512 tokens of each. That is a decision you want to have made deliberately.
:::

Filtering by length is a predicate:

```python
print(tokenized.filter(col("tokens").list.len() >= 3).to_pydict()["id"])
# [2]
```

## Sequence packing for pretraining

A causal LM trains on fixed-length sequences. Padding each document up to `seq_len`
wastes exactly as much GPU time as the padding fraction, and on a corpus of short
documents that is most of it. `pack_sequences` concatenates documents end to end, inserts
an EOS token between them, and cuts the stream into `seq_len` blocks.

```python
from batcher.ml import pack_sequences

corpus = bt.from_pydict({"tokens": [[1, 2, 3], [4, 5], [6, 7, 8, 9]]})
packed = list(
    pack_sequences(
        corpus.iter_batches(),
        token_column="tokens",
        seq_len=4,
        eos_token=0,
        drop_remainder=True,
    )
)
print(packed[0].to_pydict())
# {'tokens': [[1, 2, 3, 0], [4, 5, 0, 6], [7, 8, 9, 0]]}
```

Three documents (3 + 2 + 4 tokens, plus one EOS each) became three dense sequences of
exactly 4 tokens, with nothing padded. Note the second one: it holds the end of document
2 and the start of document 3. That is the point, and it is why the EOS token matters,
since it is the only signal that a document boundary was crossed.

`drop_remainder=True` discards the tail that does not fill a block. Set it `False` and
pass `pad_token` to keep the tail, padded. `rows_per_batch` controls how many packed
sequences come back per output batch.

It operates on a batch iterator rather than a `Dataset`, so it composes with anything
that yields batches, and it holds one buffer of tokens at a time, so a trillion-token
corpus packs in bounded memory.

## Encoding labels and categories

Text that is a *label* rather than a document does not want a tokenizer. `LabelEncoder`
maps each distinct value to an integer, learned by a `fit` over the training split.

```python
from batcher.ml import LabelEncoder

labelled = bt.from_pydict({"sentiment": ["pos", "neg", "pos", "neu"]})
enc = LabelEncoder("sentiment")
enc.fit(labelled)
print(enc.fit_transform(labelled).to_pydict())
# {'sentiment': [2, 0, 2, 1]}
print(enc.classes_)
# ['neg', 'neu', 'pos']
```

A value unseen at fit time maps to `unknown_value` (`-1` by default) rather than raising,
so a new category appearing in production does not take the job down. Fit on train only.
Fitting on train+test leaks the test distribution into the encoding.

## Where the work runs

Tokenization is CPU work and inference is GPU work, so they want different pools. Split
them into two stages: tokenize with the default CPU fan-out, then hand the token column
to a GPU stage with its own `concurrency`. The engine overlaps them, so the tokenizer for
batch *n+1* runs while the GPU chews on batch *n*.

```python
# docs: skip
scored = (
    bt.read.parquet("s3://bucket/corpus.parquet")
    .map_batches(HFTokenizer(), output_columns=["id", "text", "input_ids"])  # CPU
    .ml.infer(Classifier, num_gpus=1, concurrency=4)                          # GPU
)
```

:::{tip}
Better still: tokenize once, write the token ids to Parquet, and let every subsequent
epoch and every subsequent experiment read them. Tokenization is deterministic; running
it every epoch is pure waste.
:::

## See also

- [Preprocessors](preprocessors.md): the fit/transform contract and the rest of the family.
- [LLM inference](llm.md): generation over the tokens, and sequence packing in context.
- [Data loaders](data-loaders.md): getting the token column into a training loop.
- [Distributed training](distributed-training.md): the loader that reads the tokens you
  wrote out.
- [UDFs](../user-guide/udfs.md): the class-per-worker contract the tokenizer stage rests
  on.
- [Arrow memory](../deep-dives/arrow-memory.md): what a `List<Int64>` column costs, and
  why the boundary stays zero-copy.
- [Feature pipeline](../examples/ml/feature-pipeline.md): tokenization inside a larger
  preprocessing job.
- [ML API](../api/ml.md): the `Tokenizer`, `pack_sequences`, and `LabelEncoder` reference.

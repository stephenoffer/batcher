# Tokenization

This page covers turning text into token ids, packing them for pretraining, and encoding text that is really a category.

Tokenizing in the training loop is the classic way to leave a GPU idle. A tokenizer is CPU work, embarrassingly parallel, and it produces a column, so it belongs in the data pipeline. Run it once as an engine stage, write the result out, and the training loop reads token ids straight off disk.

The right tool depends on what the text actually is, as the following table shows:

| The column holds | Reach for | What you get |
| --- | --- | --- |
| Documents to feed a model | {py:class}`Tokenizer <batcher.ml.preprocessors.Tokenizer>` around a fast tokenizer | a `List<Int64>` column of token ids |
| Documents for causal-LM pretraining | `pack_sequences` over that token column | dense fixed-length blocks, nothing padded |
| A category, not a document | {py:class}`LabelEncoder <batcher.ml.preprocessors.LabelEncoder>` | one integer per distinct value, learned on the train split |

## The Tokenizer preprocessor

`Tokenizer(column, tokenizer, output_column=None)` takes either a plain `str -> list` callable or a HuggingFace-style tokenizer, meaning a callable object that also carries `.encode`. It's a {py:class}`Preprocessor <batcher.ml.preprocessors.Preprocessor>` with the standard `fit`, `transform` and `fit_transform` contract. There's nothing to learn, so `fit` only marks the object fitted. You still have to call it: `transform` without it raises {py:exc}`PlanError <batcher.PlanError>`.

What you pass decides how the tokenizer is driven. A HuggingFace tokenizer is called **once per Arrow batch** over the whole list of texts, which is where its Rust fast path lives, and that batched call is what unlocks `max_length`, `truncation`, `padding` and `attention_mask_column`. A plain `str -> list` callable is applied per string, and passing any of those four arguments with one raises. Null texts never reach either kind of tokenizer and stay null in the output.

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

The real thing. Pass the tokenizer straight in.

```python
# docs: skip
import batcher as bt
from batcher.ml import Tokenizer
from transformers import AutoTokenizer

hf = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

ds = bt.read.parquet("s3://bucket/corpus.parquet")
tok = Tokenizer(
    "text",
    hf,
    output_column="input_ids",
    max_length=512,
    truncation=True,
    attention_mask_column="attention_mask",
)
tok.fit_transform(ds).write.parquet("s3://bucket/tokens.parquet")
```

:::

:::{tab-item} A tokenizer you build yourself

When the tokenizer needs constructor arguments a preprocessor can't carry, or you want to tokenize and do something else in the same pass, write the UDF yourself. Make it a class, so the model loads once per worker instead of once per batch.

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
A tokenizer you construct yourself has to be constructed **once per worker**, which means a class, not a lambda. A slow tokenizer re-created per batch becomes the bottleneck of the whole job, and the GPU ends up waiting on the CPU stage that was supposed to feed it.
:::

The parallelism knobs live on {py:meth}`map_batches <batcher.Dataset.map_batches>`, not on the preprocessor. `num_workers` defaults to `"auto"`, which fans the calls across every local core. A fast tokenizer releases the GIL, so threads are the right pool, while a pure-Python tokenizer needs `multiprocessing=True` for real parallelism. `Tokenizer.transform` calls `map_batches` with the defaults, so write the UDF yourself when you need to change them.

## Token ids are a list column

The output is `List<Int64>`, an ordinary Arrow column. The whole expression surface applies, so length statistics are one aggregate instead of a Python pass over the corpus.

```python
from batcher import col

lengths = tokenized.select(n=col("tokens").list.len())
print(lengths.to_pydict())
# {'n': [2, 3]}

print(lengths.describe().to_pydict()["n"][:4])
# [2.0, 0.0, 2.5, 0.7071067811865476]
```

:::{important}
Look at that distribution before you set `max_length`. Truncation is silent. Nothing raises, and a corpus where many documents lost their tail trains happily on the first 512 tokens of each. Make that decision deliberately.
:::

Filtering by length is a predicate:

```python
print(tokenized.filter(col("tokens").list.len() >= 3).to_pydict()["id"])
# [2]
```

## Sequence packing for pretraining

A causal LM trains on fixed-length sequences. Padding each document up to `seq_len` wastes GPU time in proportion to the padding, and on a corpus of short documents that's most of it. `pack_sequences` concatenates documents end to end, inserts an EOS token after each, and cuts the stream into `seq_len` blocks.

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

Three documents of 3, 2 and 4 tokens, plus one EOS each, became three dense sequences of exactly 4 tokens with nothing padded. The second sequence holds the end of document 2 and the start of document 3. That's the point of packing, and it's also the problem the next section solves.

The following diagram sets that result beside padding the same three documents, with the segment lengths the next section adds:

![Three documents of 3, 2 and 4 tokens: 1, 2, 3, then 4, 5, then 6, 7, 8, 9. Padding each document to 4 gives the rows 1, 2, 3, pad, then 4, 5, pad, pad, then 6, 7, 8, 9, so 3 of the 12 slots are padding. pack_sequences with seq_len=4 and eos_token=0 gives the rows 1, 2, 3, EOS, then 4, 5, EOS, 6, then 7, 8, 9, EOS, with no padding and an EOS at each seam. The segment lengths are 4 for the first row, 3 and 1 for the second, where document 2 and its EOS end and document 3 begins, and 4 for the third.](/_static/diagrams/sequence_packing.svg)

`drop_remainder=True`, the default, discards the tail that doesn't fill a block, because that tail is the one place padding would enter the run. Set it to `False` to keep the tail, padded with `pad_token`, which defaults to `eos_token` or to 0 when there's none. `rows_per_batch` controls how many packed sequences come back per output batch. The packed column is a `FixedSizeList<Int64>[seq_len]`, which {py:meth}`iter_torch_batches <batcher.api.dataset.ml.DatasetML.iter_torch_batches>` turns into an `(n, seq_len)` tensor with no reshape.

Packing carries state across batches: a document cut at the end of one sequence continues into the next. So consume the result sequentially, and don't run packing as a parallel `map_batches`, which would cut the stream at nondeterministic places. Order matters too. Shuffle documents *before* packing, not after.

### Telling the model where the documents end

A packed sequence holds several unrelated documents, and plain causal attention lets every token attend straight across the joins. The data can't show the cost, because the packed column is exactly as wide either way and nothing in it says which token started a document. The EOS token marks the seams for a reader, not for the attention mask. The model pays instead.

`boundaries_column` emits the segment lengths inside each sequence:

```python
packed = list(
    pack_sequences(
        corpus.iter_batches(),
        token_column="tokens",
        seq_len=4,
        eos_token=0,
        boundaries_column="seq_lens",
    )
)
print(packed[0].to_pydict())
# {'tokens': [[1, 2, 3, 0], [4, 5, 0, 6], [7, 8, 9, 0]],
#  'seq_lens': [[4], [3, 1], [4]]}
```

The lengths in each row sum to `seq_len`. A cumulative sum of one row is the `cu_seqlens` that FlashAttention's variable-length path takes, and the same list restarts position ids per document. A document that straddles a cut contributes a segment on each side, which is what a block-diagonal mask wants, since inside one sequence the piece really is contiguous. With `drop_remainder=False`, the final sequence's padding is its own segment, so every row still sums to its width and the padding can be masked by length instead of by scanning for a pad token. Omit `boundaries_column` and the output schema doesn't change.

`pack_sequences` operates on a batch iterator rather than a {py:class}`Dataset <batcher.Dataset>`, so it composes with anything that yields batches. It holds one buffer of tokens at a time, so even a trillion-token corpus packs in bounded memory.

## Encoding labels and categories

Text that is a *label* rather than a document doesn't want a tokenizer. `LabelEncoder` maps each distinct value to an integer, learned by a `fit` over the training split.

```python
from batcher.ml import LabelEncoder

labelled = bt.from_pydict({"sentiment": ["pos", "neg", "pos", "neu"]})
enc = LabelEncoder("sentiment").fit(labelled)
print(enc.transform(labelled).to_pydict())
# {'sentiment': [2, 0, 2, 1]}
print(enc.classes_)
# ['neg', 'neu', 'pos']
```

A value unseen at fit time maps to `unknown_value`, `-1` by default, instead of raising, so a new category in production doesn't take the job down. Fit on the training split only. Fitting on train and test together leaks the test distribution into the encoding.

## Where the work runs

Tokenization is CPU work and inference is GPU work, so they want different pools. Split them into two stages: tokenize with the default CPU fan-out, then hand the token column to a GPU stage with its own `concurrency`. The engine prefetches between stages, so the tokenizer for batch *n+1* runs while the GPU works on batch *n*.

```python
# docs: skip
scored = (
    bt.read.parquet("s3://bucket/corpus.parquet")
    .map_batches(HFTokenizer(), output_columns=["id", "text", "input_ids"])  # CPU
    .ml.infer(Classifier, num_gpus=1, concurrency=4)  # GPU
)
```

:::{tip}
Better still, tokenize once, write the token ids to Parquet, and let every later epoch and experiment read them. Tokenization is deterministic, so running it every epoch is pure waste.
:::

## See also

- {doc}`Preprocessors </ml/preparing/preprocessors/index>`: the fit and transform contract, and the rest of the family.
- {doc}`LLM inference </ml/retrieval/llm/index>`: generating text with a model once the corpus is ready.
- {doc}`Data loaders </ml/training/data-loaders>`: getting the token column into a training loop.
- {doc}`Preparing a training corpus </ml/training/training-corpus>`: mixing, filtering, and decontaminating text before you tokenize it.
- {doc}`Chunking documents for RAG </ml/preparing/multimodal/pipelines>`: splitting long documents into windows before embedding.
- {doc}`Distributed training </ml/training/distributed-training>`: the loader that reads the tokens you wrote out.
- {doc}`UDFs </user-guide/transform/columns/udfs>`: the class-per-worker contract the tokenizer stage rests on.
- {doc}`Arrow memory </architecture/deep-dives/memory/arrow-memory>`: what a `List<Int64>` column costs, and why the boundary stays zero-copy.
- {doc}`Feature pipeline </cookbook/ml/pipelines/features/feature-pipeline>`: tokenization inside a larger preprocessing job.
- {doc}`ML API </api/models/ml>`: the `Tokenizer`, `pack_sequences`, and `LabelEncoder` reference.

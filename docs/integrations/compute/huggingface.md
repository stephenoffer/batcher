# Hugging Face

Two separate integrations share a name. The **datasets** side is ingestion: a `datasets.Dataset` is
an Arrow table underneath, so `bt.from_huggingface` takes that table directly and no data is
converted. The **models** side is inference: `ds.ml.infer` and `ds.ml.embed` take a Hub model id
and load it once per worker.

Ingestion needs `pip install 'batcher-engine[huggingface]'`. The model paths need `transformers` or
`sentence-transformers` (`batcher-engine[st]`) respectively.

| | |
| --- | --- |
| **Datasets in** | `bt.from_huggingface(hf)`, or `bt.read.parquet("hf://...")` |
| **Models** | `ds.ml.infer(model_id, ...)`, `ds.ml.embed(model_id, ...)` |
| **Write** | Not supported. Batcher does not push datasets to the Hub. |
| **Extra** | `pip install 'batcher-engine[huggingface]'`; `transformers` or `batcher-engine[st]` for models |
| **Parallelism** | `from_huggingface` is one in-memory source. `hf://` Parquet splits per row group. |

## Datasets in

```python
# docs: skip
import batcher as bt
from datasets import load_dataset

hf = load_dataset("imdb", split="train")
reviews = bt.from_huggingface(hf)
print(reviews.filter(bt.col("label") == 1).count())
```

`from_huggingface` reaches for the dataset's underlying `pa.Table` (`hf.data.table`) and wraps it
as an in-memory source. That is genuinely zero-copy: the same buffers, no re-encoding. It is
exactly what `bt.from_arrow` does with a table you already have, which is how the path can be
demonstrated without the Hub:

```python
import pyarrow as pa

import batcher as bt

# The Arrow table a `datasets.Dataset` is holding.
table = pa.table({"text": ["good", "bad", "great"], "label": [1, 0, 1]})
reviews = bt.from_arrow(table)
print(reviews.filter(bt.col("label") == 1).select("text").to_pydict())
# {'text': ['good', 'great']}
```

:::{important}
The consequence of "it takes the table" is that the corpus is **already in memory**. `datasets`
memory-maps its Arrow files, so this is cheaper than it sounds, but it is still a single-process
handle to the whole thing, not a streaming, larger-than-memory, distributable source.
:::

For a corpus that size, land it once and read it back:

```python
import os
import tempfile

work = tempfile.mkdtemp()
corpus = os.path.join(work, "reviews")
reviews.write.parquet(corpus)

# From here it is a normal source: split-parallel, prunable, distributable.
print(bt.read.parquet(corpus).count())
# 3
```

That Parquet directory is what you point a training job at. A `bt.read.parquet` scan splits per row
group, prunes columns and predicates at the file level, and fans out across a cluster. An
in-memory HF table can do none of it.

## Hub datasets are mostly Parquet

Most Hub datasets are stored as Parquet, and Batcher's filesystem resolver falls back to fsspec for
any scheme it does not know natively. With `huggingface_hub` installed, that includes `hf://`, so
you can skip `datasets` entirely and read the files:

```python
# docs: skip
import batcher as bt

ds = bt.read.parquet("hf://datasets/stanfordnlp/imdb/plain_text/train-00000-of-00001.parquet")
```

This is the better path when you want *part* of a large dataset. Projection and predicate pushdown
apply, so a filtered read of two columns fetches two columns, where `load_dataset` downloads the
split.

## Models

::::{tab-set}

:::{tab-item} `ds.ml.infer`

`ds.ml.infer(model_id, column=...)` runs a `transformers` pipeline over the dataset. The model
loads once per worker (it goes through the class-based `map_batches` path) and the prediction is
appended as a column. `task=` picks the pipeline kind when it cannot be inferred from the model.

```python
# docs: skip
import batcher as bt

scored = (
    bt.read.parquet("s3://lake/reviews/*.parquet")
    .ml.infer(
        "distilbert-base-uncased-finetuned-sst-2-english",
        column="text",
        output_column="sentiment",
        batch_size=64,
        num_gpus=1,
        concurrency=8,
        model_memory_gb=1.5,
    )
)
scored.write.parquet("s3://lake/reviews_scored")
```
:::

:::{tab-item} `ds.ml.embed`

`ds.ml.embed(model_id, column=...)` is the same shape for a sentence-transformers model, appending
a vector column.

```python
# docs: skip
import batcher as bt

vectors = (
    bt.read.parquet("s3://lake/reviews/*.parquet")
    .ml.embed(
        "sentence-transformers/all-MiniLM-L6-v2",
        column="text",
        output_column="embedding",
        batch_size=64,
        num_gpus=1,
        concurrency=8,
    )
)
vectors.write.parquet("s3://lake/reviews_embedded")
```
:::

::::

Upstream reading and filtering stay on CPU workers while the model sits on GPU actors. That is the
point of `num_gpus` + `concurrency`, and it is what keeps the GPUs fed rather than waiting on a
scan.

:::{dropdown} A Hugging Face tokenizer as a Batcher preprocessor
A Hugging Face tokenizer drops into the preprocessor family, since `Tokenizer` accepts anything
with `.encode`:

```python
# docs: skip
from transformers import AutoTokenizer

from batcher.ml.preprocessors import Tokenizer

hf_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = Tokenizer("text", hf_tok, output_column="input_ids").fit_transform(reviews)
```
:::

## Failure modes worth knowing

:::{warning}
**`load_dataset(..., streaming=True)` is not a streaming source here.** An HF `IterableDataset` has
no materialized Arrow table, so the adapter falls back to iterating it into one, which materializes
the whole thing. If you wanted streaming, you did not get it. Read the Parquet files instead.
:::

**Nested and non-Arrow-native features.** Image, Audio, and `ClassLabel` features are Arrow structs
of paths or bytes, not decoded tensors. They arrive as structs, and decoding is a `map_batches`
stage (or `bt.read.images` if you have the paths). Nothing decodes implicitly.

**A model id per worker is a download per worker.** The first batch on a cold cluster pulls the
weights on every worker. Pre-bake the model into the image or warm the HF cache on a shared volume;
otherwise your job's first minute is a rate-limited stampede on `huggingface.co`. Set `HF_TOKEN`
for gated models. Every worker needs it, so it belongs in the `runtime_env`.

:::{important}
**Do not pass a plain function to a GPU stage.** A function is rebuilt per batch, so the weights
reload per batch. Pass a class, or a model id, which becomes one. Batcher warns; heed the warning.
:::

## See also

- {doc}`Inference </ml/inference/inference>`: the actor pool, batching, GPU placement.
- {doc}`Reading data </user-guide/moving-data/reading-data>`: the readers and how they split.
- {doc}`Text embeddings </cookbook/ml/pipelines/text/text-embeddings>`: `ds.ml.embed` over a real corpus.
- {doc}`LLM batch scoring </cookbook/ml/pipelines/text/llm-batch-scoring>`: the same actor pool, a bigger model.
- {doc}`ML API </api/models/ml>`: `infer`, `embed`, `map_batches`, the preprocessors.
- {doc}`PyTorch </integrations/compute/pytorch>`: tensors, DDP ingest, model-once-per-worker.

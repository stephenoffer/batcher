# Hugging Face

This page covers Hugging Face datasets and models. Two separate integrations share the name. The datasets side is ingestion: a `datasets.Dataset` is an Arrow table underneath, so {py:func}`bt.from_huggingface <batcher.from_huggingface>` takes that table directly and no data is
converted. The models side is inference: {py:meth}`ds.ml.infer <batcher.api.dataset.ml.DatasetML.infer>` and {py:meth}`ds.ml.embed <batcher.api.dataset.ml.DatasetML.embed>` take a Hub model id
and load it once per worker.

Ingestion needs `pip install 'batcher-engine[huggingface]'`. `ds.ml.infer` needs `batcher-engine[transformers]`, and `ds.ml.embed` with a model id needs `batcher-engine[st]` for sentence-transformers.

The following table summarizes the integration:

| | |
| --- | --- |
| Datasets in | {py:func}`bt.from_huggingface(hf) <batcher.from_huggingface>`, or {py:meth}`bt.read.parquet("hf://...") <batcher.api.io_namespace.reader.Reader.parquet>` |
| Models | `ds.ml.infer(model_id, ...)`, `ds.ml.embed(model_id, ...)` |
| Write | Not supported. Batcher does not push datasets to the Hub. |
| Extra | `batcher-engine[huggingface]` for datasets, `[transformers]` or `[st]` for models |
| Parallelism | {py:func}`from_huggingface <batcher.from_huggingface>` is one in-memory source. `hf://` Parquet splits per row group. |

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
as an in-memory source. It's zero-copy: the same buffers, with no re-encoding. It's exactly what {py:func}`bt.from_arrow <batcher.from_arrow>` does with a table you already have, which is how the path can be
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
Taking the table means the corpus is already in memory. `datasets`
memory-maps its Arrow files, so this is cheaper than it sounds, but it's still a single-process handle to the whole corpus, not a streaming, distributable source.
:::

For a large corpus, land it once as Parquet and read it back:

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
in-memory Hugging Face table can do none of that.

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

scored = bt.read.parquet("s3://lake/reviews/*.parquet").ml.infer(
    "distilbert-base-uncased-finetuned-sst-2-english",
    column="text",
    output_column="sentiment",
    batch_size=64,
    num_gpus=1,
    concurrency=8,
    model_memory_gb=1.5,
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

vectors = bt.read.parquet("s3://lake/reviews/*.parquet").ml.embed(
    "sentence-transformers/all-MiniLM-L6-v2",
    column="text",
    output_column="embedding",
    batch_size=64,
    num_gpus=1,
    concurrency=8,
)
vectors.write.parquet("s3://lake/reviews_embedded")
```
:::

::::

Upstream reading and filtering stay on CPU workers while the model sits on GPU actors. That is the
point of `num_gpus` + `concurrency`, and it is what keeps the GPUs fed rather than waiting on a
scan.

:::{dropdown} A Hugging Face tokenizer as a Batcher preprocessor
A Hugging Face tokenizer drops into the preprocessor family, since {py:class}`Tokenizer <batcher.ml.preprocessors.Tokenizer>` accepts anything
with `.encode`:

```python
# docs: skip
from transformers import AutoTokenizer

from batcher.ml.preprocessors import Tokenizer

hf_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = Tokenizer("text", hf_tok, output_column="input_ids").fit_transform(reviews)
```
:::

## Requirements and limitations

`load_dataset(..., streaming=True)` isn't a streaming source here. An `IterableDataset` has no materialized Arrow table, so the adapter iterates it into one, which materializes the whole dataset. Read the Parquet files instead.

`Image` and `Audio` features are Arrow structs of paths or bytes, not decoded tensors, and they arrive as structs. Decoding is a `map_batches` stage, or {py:meth}`bt.read.images <batcher.api.io_namespace.reader.Reader.images>` if you have the paths. Nothing decodes implicitly. A `ClassLabel` arrives as its integer ids.

A model id is a download per worker. The first batch on a cold cluster pulls the
weights on every worker. Pre-bake the model into the image or warm the HF cache on a shared volume;
otherwise the job's first minute is a rate-limited rush of downloads from `huggingface.co`. Set `HF_TOKEN` for gated models. Every worker needs it, so it belongs in the `runtime_env`.

Don't pass a plain function to a GPU stage. A function is rebuilt per batch, so the weights reload per batch. Pass a class, or a model id, which becomes one.

## See also

- {doc}`Inference </ml/inference/inference>`: the actor pool, batching, GPU placement.
- {doc}`Reading data </user-guide/moving-data/reading-data>`: the readers and how they split.
- {doc}`Text embeddings </cookbook/ml/pipelines/text/text-embeddings>`: `ds.ml.embed` over a real corpus.
- {doc}`LLM batch scoring </cookbook/ml/pipelines/text/llm-batch-scoring>`: the same actor pool, a bigger model.
- {doc}`ML API </api/models/ml>`: `infer`, `embed`, `map_batches`, the preprocessors.
- {doc}`PyTorch </integrations/compute/pytorch>`: tensors, DDP ingest, model-once-per-worker.

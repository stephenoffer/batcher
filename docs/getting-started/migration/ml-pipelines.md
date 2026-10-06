# Batch inference and ML pipelines

This page covers the ML half of a port: running a model over batches, feeding a trainer, and writing results back out. Your model code keeps working, and the data work around it runs in the same optimized engine as every other query.

## Running a model over batches

{py:meth}`ds.map_batches(fn) <batcher.Dataset.map_batches>` runs a function over Arrow batches. Pass a class and the model loads once per worker:

```python
import pyarrow.compute as pc

import batcher as bt


class Scorer:
    def __init__(self):
        self.weight = 2.0  # load the model once per worker

    def __call__(self, batch):
        return batch.append_column("score", pc.multiply(batch.column("x"), self.weight))


print(bt.from_pydict({"x": [1, 2, 3]}).map_batches(Scorer).to_pydict())
# {'x': [1, 2, 3], 'score': [2.0, 4.0, 6.0]}
```

On a GPU, `num_gpus=` and `concurrency=` size an actor pool. The entry points below cover the common ML shapes:

| Task | Batcher | Note |
|------|---------|------|
| Map a model over batches | {py:meth}`ds.map_batches(Model, ...) <batcher.Dataset.map_batches>` | class = model loaded once per worker |
| Batch inference | `ds.ml.infer(model, num_gpus=, concurrency=)` | CPU readers feed GPU actors |
| Embeddings | {py:meth}`ds.ml.embed(model) <batcher.api.dataset.ml.DatasetML.embed>` / {py:func}`batcher.ml.embed(...) <batcher.ml.embed>` | text or image to a vector column |
| LLM generation | {py:func}`batcher.ml.llm_generate(..., engine=vllm_engine("...")) <batcher.ml.llm_generate>` | the engine does its own continuous batching |
| Distributed training feed | {py:meth}`ds.ml.stream_loader(world_size=, rank=, ...) <batcher.api.dataset.ml.DatasetML.stream_loader>` | deterministic, balanced, resumable |
| Per-op metrics | {py:meth}`ds.stats() <batcher.Dataset.stats>` | measured rows, time, bytes, and bottleneck |
| Bounded output files | {py:meth}`ds.write.parquet(max_rows_per_file=) <batcher.api.io_namespace.writer.Writer.parquet>` | honored even with `partition_by` |
| Resumable writes | `ds.write.parquet(resume=True)` | skips committed shards on re-run |

Batch size adapts toward throughput under a VRAM cap, and there's no object-store fraction to set, because bulk data never enters the Ray object store.

Splits are hash-based, so they don't depend on how the data is partitioned:

```python
data = bt.from_pydict({"x": list(range(100))})
train, test = data.ml.train_test_split(test_size=0.2, seed=0)
print(train.count() + test.count())
# 100
```

## Where the time goes

`ds.stats()` runs the query and reports measured rows, wall time, peak bytes, and spill per operator, plus the bottleneck:

```python
import batcher as bt
from batcher import col

ds = bt.from_pydict({"city": ["NYC", "LA", "NYC", "SF"], "amount": [10, 20, 30, 40]})
stats = ds.filter(col("amount") > 15).group_by("city").agg(total=col("amount").sum()).stats()
print(stats.rows, stats.bottleneck is not None)
# 3 True
```

## Writing results back out

Batch writes are atomic and resumable, so a preempted job re-runs without losing or duplicating data:

```python
import tempfile

import batcher as bt

out = tempfile.mkdtemp()  # a fresh directory, so the demo replaces nothing of yours
ds = bt.from_pydict({"v": list(range(1000))})
ds.write.parquet(out, max_rows_per_file=400)  # 3 part files
ds.write.parquet(out, max_rows_per_file=400, resume=True)  # skips committed
print(bt.read.parquet(out).count())
# 1000
```

## Feeding a distributed trainer

{py:meth}`stream_loader <batcher.api.dataset.ml.DatasetML.stream_loader>` feeds DDP, FSDP, or DeepSpeed. Every rank gets the same number of batches in a seed-reproducible order independent of world size, so a job resumes on a differently sized cluster with no repeated or skipped samples. Disable the framework's own sampler.

```python
# docs: skip  (requires torch; shown for reference)
loader = ds.ml.stream_loader(batch_size=256, world_size=8, rank=0, epoch=0, seed=1)
for batch in loader:  # {column: torch.Tensor}, this rank's shard
    train_step(batch)
```

## Offline LLM generation

Offline LLM batch inference wraps a text-generation engine such as vLLM (`batcher-engine[vllm]`), built once per worker:

```python
# docs: skip  (requires a GPU + batcher-engine[vllm]; shown for reference)
from batcher.ml import llm_generate, vllm_engine

for out in llm_generate(
    ds.iter_batches(),
    vllm_engine("meta-llama/Llama-3.1-8B-Instruct", max_model_len=4096),
    prompt_column="question",
    template="Answer concisely. Q: {question}",
):
    ...
```

## See also

- {doc}`/ml/index`: the ML guides in full, from preprocessing to serving.
- {doc}`/ml/inference/inference`: batch inference, GPU pools, and adaptive batch sizing.
- {doc}`/ml/training/data-loaders`: `stream_loader` and the distributed training feed.
- {doc}`/getting-started/migration/ray-data`: the Ray Data verbs, including class-based UDFs.

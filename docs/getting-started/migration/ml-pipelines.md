# Batch inference and ML pipelines

This page covers the ML half of a port: running a model over batches, feeding a
distributed trainer, and writing results back out. The relational work around the model
goes through the same optimizer and resource manager as any other query.

## Batch inference and ML pipelines

`ds.map_batches(fn)` runs a function over Arrow batches, and `ds.ml.infer(model)` and
`ds.ml.embed(model)` run a model. Pass a class instead of an instance and the model
loads once per worker, with `num_gpus=` and `concurrency=` for GPU actor pools. The
relational work around the model goes through the same optimizer (Kyber) and resource
manager (Carbonite) as any other query, so it's planned and sized for you rather than
executed as written.

The entry points below cover the common ML shapes.

| Task | Batcher | Note |
|------|---------|------|
| Map a model over batches | `ds.ml.map_batches(Model, ...)` | class = model loaded once per worker |
| Batch inference | `ds.ml.infer(model, num_gpus=, concurrency=)` | CPU readers feed GPU actors |
| Embeddings | `ds.ml.embed(model)` / `batcher.ml.embed(...)` | text or image to a vector column |
| LLM generation | `batcher.ml.llm_generate(..., engine=vllm_engine("..."))` | engine self-batches; no outer PID |
| Distributed training feed | `ds.ml.stream_loader(world_size=, rank=, ...)` | deterministic, balanced, resumable |
| Per-op metrics | `ds.stats()` | measured rows, time, bytes, and bottleneck |
| Bounded output files | `ds.write.parquet(max_rows_per_file=)` | honored even with `partition_by` |
| Resumable writes | `ds.write.parquet(resume=True)` | skips committed shards on re-run |

Settings other engines make you tune by hand are measured defaults here. Batch size
adapts toward throughput under a VRAM cap, `num_gpus` adapts to observed GPU
utilization, and there's no object-store proportion to set, because the data plane
bypasses it. For timings, run `python benchmarks/run.py`, which checks every result
against DuckDB and Polars before it reports a number.

`ds.stats()` answers "where is my time going". It runs the query and reports measured
rows, wall time, peak bytes, and spill per operator, plus which one was the bottleneck:

```python
import batcher as bt
from batcher import col

ds = bt.from_pydict({"city": ["NYC", "LA", "NYC", "SF"], "amount": [10, 20, 30, 40]})
stats = ds.filter(col("amount") > 15).group_by("city").agg(total=col("amount").sum()).stats()
print(stats.rows, stats.bottleneck is not None)
# 3 True
```

Batch writes are atomic and resumable, so a job killed by a spot preemption re-runs
without losing or duplicating data, and `max_rows_per_file` bounds each output file:

```python
import batcher as bt

ds = bt.from_pydict({"v": list(range(1000))})
ds.write.parquet("/tmp/bt_resume_demo", max_rows_per_file=400)            # 3 part files
ds.write.parquet("/tmp/bt_resume_demo", max_rows_per_file=400, resume=True)  # skips committed
print(bt.read.parquet("/tmp/bt_resume_demo").count())
# 1000
```

Feeding a distributed PyTorch trainer, whether DDP, FSDP, or DeepSpeed, uses
`stream_loader`. It gives every rank the same number of batches in a seed-reproducible
order that's independent of world size, so a job can resume on a differently sized
cluster with no repeated or skipped samples. Disable the framework's own sampler,
because `stream_loader` is the single shard authority.

```python
# docs: skip  (requires torch; shown for reference)
loader = ds.ml.stream_loader(batch_size=256, world_size=8, rank=0, epoch=0, seed=1)
for batch in loader:          # {column: torch.Tensor}, this rank's shard
    train_step(batch)
```

Offline LLM batch inference wraps any text-generation engine, such as vLLM behind the
`batcher-engine[vllm]` extra. The engine is built once per worker, and `template` and
`parse_json` handle prompt templating and structured-output parsing:

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

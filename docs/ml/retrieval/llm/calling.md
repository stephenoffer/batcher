# Calling a model

This page covers the three shapes a generation call takes: the one-line `ds.ml.generate`, the streaming `llm_generate` iterator, and the load-once class UDF you compose into your own pipeline. All three run the same code underneath, so they can't produce different columns for the same input.

## On a Dataset

{py:meth}`ds.ml.generate(...) <batcher.api.dataset.ml.DatasetML.generate>` is the form to start with. It returns a new lazy {py:class}`Dataset <batcher.Dataset>` with the generated column appended. Placement uses the same `num_gpus`, `concurrency`, and `accelerator_type` GPU-actor scheduling as {py:meth}`ds.ml.infer <batcher.api.dataset.ml.DatasetML.infer>` and {py:meth}`ds.ml.embed <batcher.api.dataset.ml.DatasetML.embed>`.

```python
# docs: skip
import batcher as bt
from batcher.ml import vllm_engine

engine = vllm_engine("meta-llama/Llama-3-8B-Instruct", chat=True, sampling={"max_tokens": 256})
answers = (
    bt.read.parquet("s3://bucket/questions.parquet")
    .ml.generate(engine, prompt_column="question", num_gpus=1)
    .write.parquet("s3://bucket/answers.parquet")
)
```

An *engine* is a zero-argument callable that returns a `list[str] -> list[str]` function. Nothing more is required, so a deterministic stub can stand in for the model and a generation pipeline becomes testable with no GPU:

```python
import batcher as bt

shout = lambda: lambda prompts: [p.upper() for p in prompts]
print(bt.from_pydict({"q": ["hi"]}).ml.generate(shout, prompt_column="q").to_pydict())
# {'q': ['hi'], 'response': ['HI']}
```

## Trace each request to its row

`request_id_column=` gives every request a stable id and records it beside the output, so a remote call stays traceable to the row that produced it through retries and reconciliation. When the data already has a column of that name, its values are the ids. Otherwise each id is derived from what the row sends, a SHA-256 of the rendered prompt, its per-row overrides and its image, and appended. The derivation depends on nothing about the run, so a retried batch or a re-run job sends the same id for the same row. Two rows that send identical requests share an id, which is what an idempotency key means. Fold a source key into the hash with `request_id_key=` when they should differ.

```python
import batcher as bt

echo = lambda: lambda requests: [r["prompt"].upper() for r in requests]
ds = bt.from_pydict({"order": [7, 8], "q": ["refund?", "refund?"]})
out = ds.ml.generate(echo, prompt_column="q", request_id_column="rid", request_id_key="order")
rows = out.to_pydict()
print(rows["response"], rows["rid"][0] != rows["rid"][1])
# ['REFUND?', 'REFUND?'] True
```

`http_engine` sends the id in the `X-Client-Request-Id` header, the one OpenAI documents for a client-supplied request id, and every retry of the request carries the same value. Set `request_id_header="Idempotency-Key"` for a gateway that deduplicates on one. The other engines record the id without sending it. With ids on, each request reaches the engine as a `{"prompt": ..., "request_id": ...}` dict, so a hand-written engine must accept the dict form, as it must for any per-row column.

:::{warning}
The request-id header is not yet verified against a live OpenAI-compatible provider; see `tests/PENDING_VERIFICATION.md`. It is tested against a local HTTP server, retries included.
:::

## Chat models need the chat template

`vllm_engine(chat=True)` sends each row as a conversation through `LLM.chat`, so vLLM applies the model's own chat template. Set it for any instruction-tuned or chat model.

Without it the request takes the completion path. That path is right for a base model and wrong for a tuned one, which then answers a prompt in a format it was never trained on. The output degrades and nothing fails. Leaving `chat` unset keeps the completion path but logs a warning when the model turns out to ship a chat template, because that combination is almost always this mistake. An explicit `chat=False` counts as a decision and stays silent, which is what constrained-choice classification wants.

`system=` adds a system turn to every conversation:

```python
# docs: skip
engine = vllm_engine(
    "meta-llama/Llama-3-8B-Instruct",
    chat=True,
    system="Answer in one sentence.",
    sampling={"temperature": 0.2, "top_p": 0.9, "stop": ["\n\n"]},
)
```

Vision models take their image through the completion path, so `image_column` needs `chat=False`.

## The streaming form

```python
# docs: skip
import batcher as bt
from batcher.ml import llm_generate, vllm_engine

ds = bt.read.parquet("s3://bucket/questions.parquet")
engine = vllm_engine("meta-llama/Llama-3-8B", sampling={"max_tokens": 256, "temperature": 0.0})
answers = llm_generate(ds.iter_batches(), engine, prompt_column="question")
```

`llm_generate` is an iterator transform. It takes an iterable of Arrow batches and an engine factory, and yields each batch with `output_column` appended, `"response"` by default, in input order. The factory runs once, so the model loads once. By default the row boundaries of every batch come back unchanged, generated by that single engine, which is the same shape `ds.ml.generate` produces.

Inside a call, the engine does its own continuous batching. Batcher hands it the whole batch, dispatched in length-sorted order and put back in row order afterwards, and imposes no outer batch size that would fight the engine's scheduler.

Two options add scheduling on top, and each has a cost. `target_batch_rows` re-chunks the stream toward that many rows per engine call and hill-climbs the size for throughput, which helps when your batches are far smaller than the engine can fill and costs the memory of holding that many rows. `num_workers` builds more engines *in this process*. Leave it at 1 for a GPU-resident engine, because each worker calls the factory again and 2 loads two full copies of the weights onto the same device. Raise it for a network-bound engine such as `http_engine`, whose workers wait on sockets rather than hold a model.

The prompt comes from `prompt_column`, or from a `template` that formats any of the row's columns. {doc}`prompts` covers both.

The result is an iterator of Arrow batches, so it composes with the rest of the engine. Write it straight back out, or feed it to another stage:

```python
# docs: skip
import pyarrow as pa

batches = llm_generate(ds.iter_batches(), engine, prompt_column="question")
table = pa.Table.from_batches(batches)
bt.from_arrow(table).write.parquet("s3://bucket/answers.parquet")
```

## The class-UDF form

`llm_udf(engine_factory, prompt_column=...)` returns a *class* that appends the generated column to each batch. It does the same columnar work as `llm_generate`, packaged so `map_batches` owns the scheduling. `ds.ml.generate` is exactly this: it builds the UDF and hands it to `map_batches`, which is how generation gets `num_gpus`, `concurrency`, and `accelerator_type` without a second scheduler.

Reach for it when you want the GPU-actor machinery around a generation step you compose yourself. Pass the class, never an instance. `map_batches` constructs it once per worker, and the constructor is where the engine is built. A plain function would rebuild the engine and reload the model on every batch.

```python
# docs: skip
from batcher.ml import llm_udf, vllm_engine

udf = llm_udf(
    vllm_engine("meta-llama/Llama-3-8B"),
    prompt_column="question",
    output_column="answer",
    usage=True,  # also append prompt_tokens / completion_tokens
)
answered = ds.map_batches(udf, num_gpus=1, concurrency=4)
```

It takes the same row-level options as `llm_generate`: `template`, `image_column`, `adapter_column`, `max_tokens_column`, `temperature_column`, `few_shot`, `parse_json`, `usage`, `finish_reason`, `logprobs`, `dedup`, `skip_null_prompts`, `request_id_column`, and `request_id_key`. It doesn't take `num_workers` or `target_batch_rows`, because `map_batches` supplies the pool.

## See also

- {doc}`prompts`: templates, null prompts, conversation columns, and the context window.
- {doc}`engines`: which engine to pick, throughput, and sizing a model across GPUs.
- {doc}`/ml/retrieval/llm-outputs`: parse what the model returned into typed columns.
- {doc}`/ml/preparing/tokenization`: tokenizing a corpus and packing it into fixed-length pretraining sequences.

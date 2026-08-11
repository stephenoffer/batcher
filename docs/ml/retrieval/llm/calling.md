# Calling a model

The four shapes a generation call takes, from the one-liner to the class UDF.

## On a Dataset

{py:meth}`ds.ml.generate(...) <batcher.api.dataset.ml.DatasetML.generate>` is the Dataset-native form. It returns a new lazy {py:class}`Dataset <batcher.Dataset>` with
the generated column appended, and reuses the same `num_gpus`, `concurrency`, and
`accelerator_type` GPU-actor scheduling as {py:meth}`ds.ml.infer <batcher.api.dataset.ml.DatasetML.infer>` and {py:meth}`ds.ml.embed <batcher.api.dataset.ml.DatasetML.embed>`.

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

An *engine* is only a zero-arg callable returning a `list[str] -> list[str]` function,
so a deterministic stub can stand in for a model, which makes a generation pipeline
testable with no GPU:

```python
import batcher as bt

shout = lambda: (lambda prompts: [p.upper() for p in prompts])
print(bt.from_pydict({"q": ["hi"]}).ml.generate(shout, prompt_column="q").to_pydict())
# {'q': ['hi'], 'response': ['HI']}
```

## Chat models need the chat template

`vllm_engine(chat=True)` sends each row as a conversation through `LLM.chat`, so vLLM
applies the model's own chat template. **Set this for any instruction-tuned or chat
model.** The default, `chat=False`, is the completion path, which is right for a base
model. It skips the template, and a tuned model then answers a prompt in a format it was
never trained on. The output degrades, and nothing signals it. `system=` adds a system
turn to every conversation.

```python
# docs: skip
engine = vllm_engine(
    "meta-llama/Llama-3-8B-Instruct",
    chat=True,
    system="Answer in one sentence.",
    sampling={"temperature": 0.2, "top_p": 0.9, "stop": ["\n\n"]},
)
```

Vision models take their image through the completion path, so `image_column` needs
`chat=False`.

## Sequence packing (pretraining ingest)

A pretraining batch is `seq_len` tokens wide, and documents are not. Padding each
document to the context length wastes the padding, and the GPU computes attention over it.
`pack_sequences` lays the tokenized documents end to end, separated by an EOS token, and
cuts the stream every `seq_len` tokens, so every position holds a real token.

```python
import pyarrow as pa
from batcher.ml import pack_sequences

batch = pa.RecordBatch.from_pydict({"tokens": [[1, 2, 3], [4, 5], [6, 7, 8]]})
print(list(pack_sequences([batch], seq_len=4))[0].column("tokens").to_pylist())
# [[1, 2, 3, 4], [5, 6, 7, 8]]
```

Every position in a packed sequence holds a real token, so the number of sequences a
corpus produces falls in proportion to how much padding the unpacked form carried. The
shorter the documents are relative to the context length, the larger that saving.

Packing is sequential and stateful. A document that does not fit is carried into the next
sequence rather than padded, so it transforms a *batch stream* instead of running as a
`map_batches`. A parallel per-batch map would cut the stream in a nondeterministic place.
Shuffle before packing, not after. The output is a `FixedSizeList<Int64>[seq_len]` column,
which {py:meth}`iter_torch_batches <batcher.api.dataset.ml.DatasetML.iter_torch_batches>` turns into an `(n, seq_len)` tensor with no reshape at the
edge.

The trailing partial sequence is dropped by default. Set `drop_remainder=False` to pad it
with `pad_token` instead, so every emitted batch keeps one schema.

## The streaming form

```python
# docs: skip
import batcher as bt
from batcher.ml import llm_generate, vllm_engine

ds = bt.read.parquet("s3://bucket/questions.parquet")
engine = vllm_engine("meta-llama/Llama-3-8B", sampling={"max_tokens": 256, "temperature": 0.0})
answers = llm_generate(ds.iter_batches(), engine, prompt_column="question")
```

`llm_generate` is an iterator transform. It takes an iterable of Arrow batches and an
engine factory, and yields each batch with `output_column` appended, defaulting to
`"response"`, in input order. The factory is a zero-arg callable run once per worker,
so the model is loaded once and reused. Throughput comes from two layers. `num_workers`
engine copies run in parallel, and inside each, the engine batches the requests it is
handed. Batcher reshapes the incoming morsels into request lists of about
`target_batch_rows` and lets the engine's own continuous batching schedule them across
its accelerators. There is no outer latency controller, because the engine owns its
batching. The prompt comes from `prompt_column` directly, or from a `template` that
formats any of the row's columns into a prompt.

Because the result is an iterator of Arrow batches, it composes with the rest of the
engine. Write it straight back out, or feed it into another stage:

```python
# docs: skip
import pyarrow as pa

batches = llm_generate(ds.iter_batches(), engine, prompt_column="question")
table = pa.Table.from_batches(batches)
bt.from_arrow(table).write.parquet("s3://bucket/answers.parquet")
```

## The class-UDF form

`llm_udf(engine_factory, prompt_column=...)` returns a **class** that appends the
generated column to each batch. It does the same columnar work as `llm_generate`,
packaged so `map_batches` can own the scheduling. `ds.ml.generate` is exactly this. It
builds the UDF and hands it to `map_batches`, which is how generation inherits
`num_gpus`, `concurrency`, and `accelerator_type` instead of carrying a second scheduler.

Reach for it directly when you want the GPU-actor machinery around a generation step you
are composing yourself. Pass the class, never an instance. `map_batches` constructs it
once per worker, and that constructor is where the engine is built. A plain function
would rebuild the engine and reload the model on every batch.

```python
# docs: skip
from batcher.ml import llm_udf, vllm_engine

udf = llm_udf(
    vllm_engine("meta-llama/Llama-3-8B"),
    prompt_column="question",
    output_column="answer",
    usage=True,  # also append prompt_tokens / completion_tokens
)
answered = ds.ml.map_batches(udf, num_gpus=1, concurrency=4)
```

It takes the same `template`, `image_column`, `adapter_column`, `parse_json`, and `usage`
options as `llm_generate`, minus the pool knobs, which `map_batches` supplies.

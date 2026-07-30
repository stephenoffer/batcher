# LLM inference

Offline text generation over millions of rows. The engine loads once per worker and
does its own continuous batching, so Batcher feeds it whole request lists and handles
the surrounding columnar work of building prompts from row columns and parsing structured
output.

## On a Dataset

`ds.ml.generate(...)` is the Dataset-native form. It returns a new lazy `Dataset` with
the generated column appended, and reuses the same `num_gpus`, `concurrency`, and
`accelerator_type` GPU-actor scheduling as `ds.ml.infer` and `ds.ml.embed`.

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
which `iter_torch_batches` turns into an `(n, seq_len)` tensor with no reshape at the
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

## Building prompts from columns

When the prompt is more than a single column, pass a `template`, which is a `str.format`
string over the row's columns. `prompt_column` is then ignored, and each row's prompt
is `template.format(**row)`, so any combination of columns assembles the request
without a per-row Python loop in your code.

```python
# docs: skip
from batcher.ml import llm_generate, vllm_engine

engine = vllm_engine("meta-llama/Llama-3-8B", sampling={"max_tokens": 128})
summaries = llm_generate(
    ds.iter_batches(),
    engine,
    template="Summarize the following {category} review in one sentence:\n\n{text}",
    output_column="summary",
)
```

A shared instruction prefix, whether a system prompt baked into the template or the same
leading text on every row, is encoded once by the engine when prefix caching is on.
`vllm_engine` enables prefix caching by default, so a long fixed preamble costs little
across millions of rows.

To build the prompt column as its own expression before generation, `bt.render_template` fills
named `{placeholder}` slots from columns, `bt.wrap_tag` surrounds a field in `<tag>...</tag>` for a
structured prompt, and `bt.truncate_to_token_budget` trims a column to fit the context window. They
are row-wise string builders that run in the data plane.

```python
import batcher as bt

rows = bt.from_pydict({"topic": ["comets"], "doc": ["a very long document..."]})
built = rows.select(
    prompt=bt.render_template(
        "Summarize {t} using {d}",
        t=bt.col("topic"),
        d=bt.wrap_tag(bt.truncate_to_token_budget("doc", budget=1000), "doc"),
    )
)
print(built.to_pydict()["prompt"][0][:24])
# Summarize comets using <
```

## Engines

An **`Engine`** is a callable that maps `list[str]` prompts to a list of completions,
one per prompt, in input order. That is the whole contract. An **`EngineFactory`** is a
zero-argument callable returning an `Engine`, called once per worker so the model loads
a single time.

Keeping the contract that small is what makes the backends interchangeable. A local vLLM
engine holding weights on a GPU and a remote OpenAI-compatible HTTP endpoint both
satisfy it, so `ds.ml.generate`, `llm_generate`, and `llm_udf` take either without
knowing which they got. It is also why the stub above works. A lambda returning a lambda
is a legal `EngineFactory`, which is how you test a generation pipeline with no GPU.

```python
engine_factory = lambda: (lambda prompts: [p.upper() for p in prompts])
print(engine_factory()(["a", "b"]))
# ['A', 'B']
```

An engine may also set `last_usage`, one `(prompt_tokens, completion_tokens)` pair per
request, which is what `usage=True` reads to append the token-count columns.
`vllm_engine` and `http_engine` both do.

| Engine | Use |
| --- | --- |
| `vllm_engine(model, *, sampling, guided_json, guided_regex, lora_path, **engine_kwargs)` | Local vLLM on a GPU. `sampling` (max tokens, temperature, etc.), `guided_json` / `guided_regex` for structured output, `lora_path` for an adapter, and `engine_kwargs` for tensor parallelism, quantization, and the rest of vLLM's engine options. Needs `batcher-engine[vllm]`. |
| `http_engine(base_url, model, *, api_key, system, chat=True, max_tokens=512, temperature=0.0, on_error="raise", timeout=60.0)` | An OpenAI-compatible HTTP endpoint (vLLM server, llama.cpp, a hosted API). Applies the chat template server-side; retries on rate limits. `on_error="null"` skips a failed row instead of failing the batch. |
| `anthropic_engine(model, *, api_key, system, max_tokens=1024, temperature=None, on_error="raise", concurrency=8)` | A hosted Claude model over the Anthropic Messages API. Same interchangeable engine contract; `temperature` is omitted unless set (some models reject it). Reads `$ANTHROPIC_API_KEY` when `api_key` is unset. |

`vllm_engine` is the high-throughput path. The GPU stays saturated because vLLM
batches continuously across in-flight requests. It enables **prefix caching** and
**chunked prefill** by default, both of which help offline batch throughput and
time-to-first-token. Any value you pass in `engine_kwargs` overrides the default. Use
`sampling` for decoding parameters such as `temperature`, `top_p`, `max_tokens`, `stop`,
`seed`, and `n`. Use `lora_path` to serve a LoRA adapter on top of the base model, and
`engine_kwargs` for `max_model_len`, `gpu_memory_utilization`, `tensor_parallel_size`,
and the rest of vLLM's engine options.

```python
# docs: skip
from batcher.ml import vllm_engine

engine = vllm_engine(
    "meta-llama/Llama-3-70B",
    sampling={"temperature": 0.7, "top_p": 0.9, "max_tokens": 512},
    tensor_parallel_size=4,          # shard the model across 4 GPUs
    gpu_memory_utilization=0.92,
    quantization="awq",
)
```

`http_engine` offloads the model entirely. Throughput is then bounded by the endpoint,
and `num_workers` controls how many concurrent request streams you open against it.
With `chat=True`, the default, the server applies the model's chat template, so a plain
prompt is wrapped as a user message. Pass `system=...` to prepend a system message.

```python
# docs: skip
from batcher.ml import llm_generate, http_engine

engine = http_engine(
    "https://api.example.com/v1",
    "gpt-4o-mini",
    api_key="sk-...",
    system="You are a precise data labeler.",
    max_tokens=64,
)
labeled = llm_generate(ds.iter_batches(), engine, prompt_column="text", num_workers=8)
```

`anthropic_engine` is the same offloaded shape for a hosted Claude model. It speaks the
Anthropic Messages API rather than the OpenAI one, but the engine contract is identical,
so it drops into `ds.ml.generate` unchanged. Leave `temperature` unset unless the target
model accepts it.

```python
# docs: skip
from batcher.ml import anthropic_engine

engine = anthropic_engine(
    "claude-haiku-4-5",
    system="You are a precise data labeler.",
    max_tokens=64,
)
labeled = ds.ml.generate(engine, prompt_column="text", concurrency=8)
```

## Batching and throughput

Two knobs control the request flow. `num_workers` sets how many engine copies run in
parallel. Each loads the model once, so for `vllm_engine` size it to the GPUs you have,
or to the model replicas that fit. `target_batch_rows` sets how many requests Batcher
hands the engine at a time, and the engine's continuous batching then schedules them
across its accelerator. Do **not** try to micro-manage an outer batch size for vLLM. Its
scheduler already interleaves prefill and decode, and a fixed outer batch would fight
it.

```python
# docs: skip
answers = llm_generate(
    ds.iter_batches(),
    engine,
    prompt_column="question",
    num_workers=4,           # 4 model replicas in parallel
    target_batch_rows=512,   # requests handed to each engine call
)
```

## Vision-language models

Pass an `image_column` holding raw bytes or a decoded `(H, W, 3)` tensor for a multimodal
model. Each request becomes prompt plus image, and the engine must be vision-capable.
`vllm_engine` on a multimodal model handles it. A null image row falls back to a
text-only request.

```python
# docs: skip
import batcher as bt
from batcher.ml import llm_generate, vllm_engine

ds = bt.read.images("s3://bucket/photos/", decode=True)  # an "image" tensor column
engine = vllm_engine("llava-hf/llava-1.5-7b-hf")
captions = llm_generate(
    ds.iter_batches(),
    engine,
    prompt_column="instruction",
    image_column="image",
    output_column="caption",
)
```

## Text embeddings

`ds.ml.embed` with a sentence-transformers model id embeds a text `column`, loading
the model once per worker and scheduling it across GPU actors. It is the
retrieval-pipeline companion to {doc}`vector search <vector-search>`. It appends one
fixed-width vector column named by `output_column` and keeps the dataset lazy.
`num_gpus` reserves an accelerator fraction per worker, `concurrency` sets the worker
count as an int or an autoscaling `(min, max)` range, and `batch_size` controls how many
texts go through the model at once. Needs `batcher-engine[st]`.

```python
# docs: skip
import batcher as bt

ds = bt.read.parquet("s3://bucket/docs.parquet")
vectors = ds.ml.embed("sentence-transformers/all-MiniLM-L6-v2", column="text", num_gpus=1)
vectors.write.lance("s3://bucket/vectors.lance")
```

For a model Batcher does not wrap directly, `ds.ml.embed` also takes any load-once
callable or class that maps a batch to vectors, and `ds.ml.infer` does the same for
general model scoring. Both accept `concurrency`, `num_gpus`, and `batch_size` the
same way.

## The rest of this topic

Generation is half the job. The other half is turning what the model produced into
columns you can query, and measuring whether it was any good:

- {doc}`llm-outputs`: parsing generated strings into typed columns, and constrained
  (guided) decoding.
- {doc}`llm-evaluation`: scoring generations against a reference, and the single-scan
  monitors for running generation at scale.

## See also

- {doc}`Inference <inference>`: the general batch-inference and embedding path.
- {doc}`Serving <serving>`: expose a model behind an endpoint.
- {doc}`GPU scheduling <gpu>`: how `num_gpus` and `concurrency` map to actors.
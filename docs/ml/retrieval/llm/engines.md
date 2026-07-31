# Engines and throughput

Which backend runs the model, what it costs, and the two neighboring modalities.

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

Both hosted engines also take `requests_per_minute` and `tokens_per_minute`, covered below.

### Staying inside a provider's quota

Retrying a 429 is recovery, not control. A fleet that only retries still sends the burst that
caused the rejection, then sends it again, so throughput settles below the quota while every
worker spends its time asleep. Some providers also count rejected requests against the quota,
which makes the retries fund their own starvation.

`http_engine` and `anthropic_engine` take `requests_per_minute` and `tokens_per_minute`. Each
worker holds a token bucket refilled at that rate, and a request waits for capacity *before*
going out. The send rate is then smooth at the quota rather than a sawtooth under it. The token
dimension counts the prompt plus the reply the request reserved with `max_tokens`, because that
is what a provider counts.

```python
# docs: skip
from batcher.ml import http_engine

engine = http_engine(
    "https://api.example.com/v1",
    "some-model",
    api_key="...",
    concurrency=16,
    requests_per_minute=500,     # per worker, not per fleet
    tokens_per_minute=400_000,
)
```

The limit is **per worker**, deliberately: coordinating a fleet-wide limiter would put a
synchronous round trip in front of every request. Divide the account quota by the number of
workers and leave headroom, because a provider measures arrival at its edge, where two workers'
bursts can coincide. Keep `retries` on as well — a limiter smooths your own send rate, it
cannot see the other traffic on the account.

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
retrieval-pipeline companion to {doc}`vector search </ml/retrieval/vector-search>`. It appends one
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

- {doc}`/ml/retrieval/llm-outputs`: parsing generated strings into typed columns, and constrained
  (guided) decoding.
- {doc}`/ml/retrieval/llm-evaluation`: scoring generations against a reference, and the single-scan
  monitors for running generation at scale.

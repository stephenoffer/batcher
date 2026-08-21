# Engines and throughput

Which backend runs the model, what it costs, and the two neighboring modalities.

## Engines

An **`Engine`** is a callable that maps `list[str]` prompts to a list of completions,
one per prompt, in input order. That is the whole contract. An **`EngineFactory`** is a
zero-argument callable returning an `Engine`, called once per worker so the model loads
a single time.

Keeping the contract that small is what makes the backends interchangeable. A local vLLM
engine holding weights on a GPU and a remote OpenAI-compatible HTTP endpoint both
satisfy it, so {py:meth}`ds.ml.generate <batcher.api.dataset.ml.DatasetML.generate>`, `llm_generate`, and `llm_udf` take either without
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
| `sglang_engine(model, *, chat, system, sampling, json_schema, regex, ebnf, lora_paths, **engine_kwargs)` | Local SGLang on a GPU. Same shape as `vllm_engine`; its RadixAttention cache reuses whatever prefix the rows happen to share rather than one configured prefix, which is the win on templated batches. `max_tokens` is accepted as an alias for SGLang's `max_new_tokens`, so a `sampling` dict moves between backends unchanged. Needs `batcher-engine[sglang]`. |
| `http_engine(base_url, model, *, api_key, system, chat=True, max_tokens=512, temperature=0.0, on_error="raise", timeout=60.0)` | An OpenAI-compatible HTTP endpoint (vLLM server, llama.cpp, a hosted API). Applies the chat template server-side; retries on rate limits. `on_error="null"` skips a failed row instead of failing the batch. |
| `anthropic_engine(model, *, api_key, system, max_tokens=1024, temperature=None, on_error="raise", concurrency=8)` | A hosted Claude model over the Anthropic Messages API. Same interchangeable engine contract; `temperature` is omitted unless set (some models reject it). Reads `$ANTHROPIC_API_KEY` when `api_key` is unset. |
| `bedrock_engine(model, *, region, system, max_tokens=1024, temperature=None, additional_model_request_fields, on_error="raise", concurrency=8)` | Any model on AWS Bedrock, through the Converse API. One request shape across Claude, Llama, Mistral, Titan and Nova, so changing model family is a string change. Credentials come from the standard AWS chain, so a worker on EC2 or EKS needs none passed in. Needs `boto3`. |
| `gemini_engine(model, *, api_key, base_url, system, max_tokens=1024, response_schema, safety_settings, on_error="raise", concurrency=8)` | A Gemini model over `generateContent`. Point `base_url` at a Vertex AI endpoint and pass an OAuth token to use Vertex instead. `response_schema` constrains the generation to a schema rather than asking for JSON. Reads `$GEMINI_API_KEY` or `$GOOGLE_API_KEY` when `api_key` is unset. |

Both hosted engines also take `requests_per_minute` and `tokens_per_minute`, covered below.

### Choosing between vLLM and SGLang

Both are local, GPU-resident, and interchangeable through the same `EngineFactory` contract,
so this is a measurement rather than a commitment. The difference that matters offline is what
each caches.

`vllm_engine` caches a prefix you configure. `sglang_engine` keeps every prompt's KV cache in a
radix tree keyed by token prefix, so two rows that happen to share an opening share its
prefill, and a third that shares a longer opening shares more. Batch inference is usually the
*same template* over different rows, which is exactly the shape that pays: a shared system
message, a shared instruction, a shared few-shot block, and a few hundred varying characters at
the end.

```python
# docs: skip
import batcher as bt

engine = bt.ml.sglang_engine(
    "meta-llama/Llama-3.1-8B-Instruct",
    chat=True,
    system="Answer with a single sentence.",
    sampling={"max_tokens": 128},
    tp_size=2,
)
answered = ds.ml.generate(engine, prompt_column="question", num_gpus=2)
```

Structured output is one keyword rather than a separate object. Pass `json_schema=` to
constrain generation to a schema, `regex=` to a pattern, or `ebnf=` to a grammar; exactly one
of the three, because a generation is constrained by one grammar and silently preferring one
over another produces output shaped by a rule you did not pick.

```python
# docs: skip
engine = bt.ml.sglang_engine(
    "meta-llama/Llama-3.1-8B-Instruct",
    chat=True,
    json_schema={"type": "object", "properties": {"sentiment": {"type": "string"}}},
)
typed = ds.ml.generate(engine, prompt_column="review", parse_json=True)
```

Rows sharing a prefix hit the tree more often when they are adjacent, so ordering the input by
the varying part is worth trying on a large run.

### Reaching a cloud provider's own API

`http_engine` covers everything that speaks the OpenAI chat-completions shape, which is most
self-hosted servers and several hosted APIs. Three large providers do not, and each has its own
engine for the same reason: the request body differs, and nothing else does.

```python
# docs: skip
import batcher as bt

# AWS: one Converse request shape across every model family Bedrock hosts.
claude = bt.ml.bedrock_engine("anthropic.claude-sonnet-4-20250514-v1:0", region="us-west-2")
llama = bt.ml.bedrock_engine("meta.llama3-1-70b-instruct-v1:0", region="us-west-2")

# Google: the same engine reaches Vertex AI through `base_url` and an OAuth token.
gemini = bt.ml.gemini_engine("gemini-2.0-flash")

answered = ds.ml.generate(claude, prompt_column="question", concurrency=16)
```

Bedrock takes credentials from the standard AWS chain, so a worker running under an instance
role or IRSA needs nothing passed in. `additional_model_request_fields` forwards whatever the
underlying model accepts that Converse does not name, which is how you reach a Claude thinking
block or a Llama repetition penalty without leaving the uniform request shape.

Gemini constrains structured output with `response_schema=`, which is stronger than asking for
JSON in the prompt: the decoding itself is constrained, so the result parses.

```python
# docs: skip
engine = bt.ml.gemini_engine(
    "gemini-2.0-flash",
    response_schema={
        "type": "object",
        "properties": {"sentiment": {"type": "string"}, "score": {"type": "number"}},
    },
)
typed = ds.ml.generate(engine, prompt_column="review", parse_json=True)
```

Both take the same `requests_per_minute` and `tokens_per_minute` caps as the other hosted
engines, and both honor a `Retry-After` the provider sends rather than guessing a backoff.

### Pointing `http_engine` at a self-hosted server

The engines above load a model *into the worker*. This is the other deployment: the model
already runs behind an HTTP server you operate, and the worker is a client of it. Reach for
this when the server is shared with other consumers, or when it is scaled and scheduled
separately from the batch job.

`http_engine` is not tied to OpenAI. It posts to `{base_url}/chat/completions`, so it drives
any server that speaks the OpenAI chat-completions shape, and most modern inference servers
do. There is no separate engine to pick and nothing to subclass: set `base_url` to the
server's `/v1` root and `model` to the name that server serves under.

| Server | `base_url` |
| --- | --- |
| vLLM (`vllm serve`) | `http://<host>:8000/v1` |
| SGLang (`python -m sglang.launch_server`) | `http://<host>:<port>/v1`, the port you passed `--port` |
| Ollama | `http://<host>:11434/v1` |
| Hugging Face TGI (Messages API, 1.4.0 and later) | `http://<host>:8080/v1` |
| llama.cpp (`llama-server`) | `http://<host>:8080/v1` |

```python
# docs: skip
import batcher as bt

engine = bt.ml.http_engine("http://localhost:11434/v1", "llama3.1", api_key="ollama")
scored = ds.ml.generate(engine, prompt_column="question", concurrency=16)
```

The ports above are each server's own default, so check the one your deployment actually
binds. Anything the server accepts but the OpenAI schema does not name goes through
`extra_body`, which is passed into the request body untouched.

```python
# docs: skip
engine = bt.ml.http_engine(
    "http://localhost:30000/v1",
    "Qwen/Qwen2.5-7B-Instruct",
    extra_body={"separate_reasoning": True},
)
```

:::{note}
A served endpoint and a local `vllm_engine` are not the same throughput story. The server
already batches across every client that is talking to it, so the lever is `concurrency`
(how many requests this worker keeps in flight), not an outer batch size. The
"Batching and throughput" section below covers the difference.
:::

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

## Sizing a model across several GPUs

`tensor_parallel_size` shards one model's weights over a group of GPUs so a model too large
for one card runs at all. It is the setting that most often costs a job hours, in three
distinct ways, and Batcher checks all three on the worker before the engine builds rather than
after it has downloaded the weights.

The first is a group wider than the devices the worker holds. A stage scheduled with one GPU
and told to build a four-way group does not raise. The engine waits for peers that were never
scheduled, holding its slot, and the job reads as hung rather than misconfigured. It is the
easiest mistake to make on a multi-GPU node, because the node has the cards the degree implies
and the task does not. Give the stage the GPUs the degree needs, through the `num_gpus` of the
map stage that runs it.

The second is a group too small to hold the model. Batcher reads the model's weight footprint
from the repository's metadata, which costs one request and no download, and says so before
the tens of gigabytes move. Sizing on weights alone is not enough on its own: a group where
the weights just fit leaves no room for the key/value cache, and an engine with no cache does
not fail, it admits one sequence, preempts it, recomputes it, and reports a third of the
throughput the hardware can do.

The third is the interconnect. The same `tensor_parallel_size=2` is nearly free on an NVLink
card and a measured 30-50% throughput loss on a PCIe-only one such as an L4 or an L40S,
because every forward all-reduces across the group. Batcher warns when the degree is above one
on a PCIe-only card, and separately when a card that *supports* NVLink has its links reported
down, which is a node fault rather than a setting: the group looks free on paper while every
collective runs over PCIe. Check {py:func}`bt.accelerators() <batcher.accelerators>` and drain the node if the links do not
come back.

None of these change the degree. The penalty is hardware-specific and the fix differs per
case, so Batcher reports and you decide.

```{tip}
Tensor parallelism divides the key/value cache as well as the weights, so the concurrency a
group reaches is more than proportional to its size. Prefer the smallest group that holds the
model, and spend the remaining GPUs on more replicas: a replica serves its own sequences at
full rate, while a wider group than the model needs pays an all-reduce on every layer of every
token for memory nobody uses.
```

## Where model weights are cached

The default HuggingFace cache lives under `$HOME`, which on a GPU node is usually a small
container overlay shared with the image. A node running eight GPU workers is eight processes
each wanting the same tens of gigabytes there, and the failure is an out-of-space error partway
through a download rather than anything that names the cause. When no cache location is
configured, Batcher points the cache at the node's measured local disk before the first model
loads, so the workers on a node share one copy on the filesystem that has room for it.

Set any of `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `HF_HOME`, or `TRANSFORMERS_CACHE` and
that choice is used untouched. An image that has already baked models into the default cache is
also left alone, because moving the cache there would force the download it was meant to avoid.

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

{py:meth}`ds.ml.embed <batcher.api.dataset.ml.DatasetML.embed>` with a sentence-transformers model id embeds a text `column`, loading
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
callable or class that maps a batch to vectors, and {py:meth}`ds.ml.infer <batcher.api.dataset.ml.DatasetML.infer>` does the same for
general model scoring. Both accept `concurrency`, `num_gpus`, and `batch_size` the
same way.

## The rest of this topic

Generation is half the job. The other half is turning what the model produced into
columns you can query, and measuring whether it was any good:

- {doc}`/ml/retrieval/llm-outputs`: parsing generated strings into typed columns, and constrained
  (guided) decoding.
- {doc}`/ml/retrieval/llm-evaluation`: scoring generations against a reference, and the single-scan
  monitors for running generation at scale.

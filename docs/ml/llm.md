# LLM inference

`llm_generate` runs offline text generation over millions of rows. The engine loads
once per worker and does its own continuous batching, so Batcher feeds it whole
request lists and handles the surrounding columnar work: building prompts from row
columns and parsing structured output.

```python
# docs: skip
import batcher as bt
from batcher.ml import llm_generate, vllm_engine

ds = bt.read.parquet("s3://bucket/questions.parquet")
engine = vllm_engine("meta-llama/Llama-3-8B", sampling={"max_tokens": 256, "temperature": 0.0})
answers = llm_generate(ds.iter_batches(), engine, prompt_column="question")
```

`llm_generate` is an iterator transform: it takes an iterable of Arrow batches and an
engine factory, and yields each batch with `output_column` appended (default
`"response"`), in input order. The factory is a zero-arg callable run once per worker,
so the model is loaded once and reused. Throughput comes from two layers: `num_workers`
engine copies run in parallel, and inside each, the engine batches the requests it is
handed. Batcher reshapes the incoming morsels into request lists of about
`target_batch_rows` and lets the engine's own continuous batching schedule them across
its accelerators — there is no outer latency controller, because the engine owns its
batching. The prompt comes from `prompt_column` directly, or from a `template` that
formats any of the row's columns into a prompt.

Because the result is an iterator of Arrow batches, it composes with the rest of the
engine — write it straight back out, or feed it into another stage:

```python
# docs: skip
import pyarrow as pa

batches = llm_generate(ds.iter_batches(), engine, prompt_column="question")
table = pa.Table.from_batches(batches)
bt.from_arrow(table).write.parquet("s3://bucket/answers.parquet")
```

## Building prompts from columns

When the prompt is more than a single column, pass a `template` — a `str.format`
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

A shared instruction prefix (a system prompt baked into the template, or the same
leading text on every row) is encoded once by the engine when prefix caching is on —
which `vllm_engine` enables by default — so a long fixed preamble costs little across
millions of rows.

## Engines

| Engine | Use |
| --- | --- |
| `vllm_engine(model, *, sampling, guided_json, guided_regex, lora_path, **engine_kwargs)` | Local vLLM on a GPU. `sampling` (max tokens, temperature, etc.), `guided_json` / `guided_regex` for structured output, `lora_path` for an adapter, and `engine_kwargs` for tensor parallelism, quantization, and the rest of vLLM's engine options. Needs `batcher-engine[vllm]`. |
| `http_engine(base_url, model, *, api_key, system, chat=True, max_tokens=512, temperature=0.0, timeout=60.0)` | An OpenAI-compatible HTTP endpoint (vLLM server, llama.cpp, a hosted API). Applies the chat template server-side; retries on rate limits. |

`vllm_engine` is the high-throughput path — the GPU stays saturated because vLLM
batches continuously across in-flight requests. It enables **prefix caching** and
**chunked prefill** by default (both throughput/TTFT wins for offline batch); any value
you pass in `engine_kwargs` overrides the default. Use `sampling` for decoding
parameters (`temperature`, `top_p`, `max_tokens`, `stop`, `seed`, `n`), `lora_path` to
serve a LoRA adapter on top of the base model, and `engine_kwargs` for `max_model_len`,
`gpu_memory_utilization`, `tensor_parallel_size`, `quantization`, and the rest of
vLLM's engine options.

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

`http_engine` offloads the model entirely; throughput is then bounded by the endpoint,
and `num_workers` controls how many concurrent request streams you open against it.
With `chat=True` (the default) the server applies the model's chat template, so a plain
prompt is wrapped as a user message — pass `system=...` to prepend a system message.

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

## Batching and throughput

Two knobs control the request flow. `num_workers` sets how many engine copies run in
parallel — each loads the model once, so for `vllm_engine` size it to the GPUs you have
(or the model replicas that fit). `target_batch_rows` sets how many requests Batcher
hands the engine at a time; the engine's continuous batching then schedules them across
its accelerator. Do **not** try to micro-manage an outer batch size for vLLM — its
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

## Structured output

Constrain generation to a JSON schema so every row is parseable, then parse it into a
struct column. `guided_json` on the engine forces the model's decoding to the schema,
and `parse_json=True` on `llm_generate` parses each output into a struct; a row that
fails to parse gets a null rather than failing the batch. Pair the two — guided
decoding makes the output well-formed, and `parse_json` turns it into typed columns
you can query downstream.

```python
# docs: skip
from batcher.ml import llm_generate, vllm_engine

schema = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "confidence": {"type": "number"},
    },
    "required": ["label"],
}
engine = vllm_engine("meta-llama/Llama-3-8B", guided_json=schema)
classified = llm_generate(
    ds.iter_batches(),
    engine,
    prompt_column="text",
    output_column="result",
    parse_json=True,         # "result" becomes a struct: {label, confidence}
)
```

For a fixed pattern rather than a full schema, `guided_regex` constrains the output to
a regular expression (e.g. `r"\d{4}-\d{2}-\d{2}"` for a date).

## Vision-language models

Pass an `image_column` (raw bytes or a decoded `(H, W, 3)` tensor) for a multimodal
model. Each request becomes prompt plus image, and the engine must be vision-capable —
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
the model once per worker and scheduling it across GPU actors — the retrieval-pipeline
companion to [vector search](multimodal.md). It appends one fixed-width vector column
(named by `output_column`) and keeps the dataset lazy. `num_gpus` reserves accelerator
fraction per worker, `concurrency` sets the worker count (or an autoscaling
`(min, max)` range), and `batch_size` controls how many texts go through the model at
once. Needs `batcher-engine[st]`.

```python
# docs: skip
import batcher as bt

ds = bt.read.parquet("s3://bucket/docs.parquet")
vectors = ds.ml.embed("sentence-transformers/all-MiniLM-L6-v2", column="text", num_gpus=1)
vectors.write.lance("s3://bucket/vectors.lance")
```

For a model Batcher does not wrap directly, `ds.ml.embed` also takes any load-once
callable or class that maps a batch to vectors, and `ds.ml.infer` does the same for
general model scoring — both accept `concurrency`, `num_gpus`, and `batch_size` the
same way.

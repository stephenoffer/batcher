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
engine factory, and yields each batch with `output_column` appended, in input order.
The factory is a zero-arg callable run once per worker, so the model is loaded once and
reused. Throughput comes from two layers: `num_workers` engine copies run in parallel,
and inside each, the engine batches the requests it is handed. Batcher reshapes the
incoming morsels into request lists of about `target_batch_rows` and lets the engine's
own continuous batching schedule them across its accelerators — there is no latency
controller, because the engine owns its batching. The prompt comes from `prompt_column`
directly, or from a `template` that formats any of the row's columns into a prompt.

## Engines

| Engine | Use |
| --- | --- |
| `vllm_engine(model, *, sampling, guided_json, guided_regex, lora_path, **engine_kwargs)` | Local vLLM on a GPU. `sampling` (max tokens, temperature, etc.), `guided_json` / `guided_regex` for structured output, `lora_path` for an adapter, and `engine_kwargs` for tensor parallelism, quantization, and the rest of vLLM's engine options. Needs `batcher-engine[vllm]`. |
| `http_engine(base_url, model, *, api_key, system, chat=True, max_tokens=512, temperature=0.0, timeout=60.0)` | An OpenAI-compatible HTTP endpoint (vLLM server, llama.cpp, a hosted API). Applies the chat template server-side; retries on rate limits. |

`vllm_engine` is the high-throughput path — the GPU stays saturated because vLLM
batches continuously across in-flight requests. `http_engine` offloads the model
entirely; throughput is then bounded by the endpoint, and `num_workers` controls how
many concurrent request streams you open against it.

## Structured output

Constrain generation to a JSON schema so every row is parseable, then parse it into a
struct column. `guided_json` forces the model's decoding to the schema, and
`parse_json=True` parses each output into a struct; a row that fails to parse gets a
null rather than failing the batch.

```python
# docs: skip
from batcher.ml import llm_generate, vllm_engine

engine = vllm_engine("my-model", guided_json={"type": "object", "properties": {"label": {"type": "string"}}})
classified = llm_generate(batches, engine, prompt_column="text", parse_json=True)
```

## Vision-language models

Pass an `image_column` (raw bytes or a decoded `(H, W, 3)` tensor) for a multimodal
model. Each request becomes prompt plus image, and the engine must be vision-capable —
`vllm_engine` on a multimodal model handles it.

```python
# docs: skip
from batcher.ml import llm_generate, vllm_engine

engine = vllm_engine("llava-hf/llava-1.5-7b-hf")
captions = llm_generate(batches, engine, prompt_column="instruction", image_column="image")
```

## Text embeddings

`ds.ml.embed_text` embeds a text column with a sentence-transformers model, loaded
once per worker and scheduled across GPU actors — the retrieval-pipeline companion to
[vector search](multimodal.md). It appends one fixed-width vector column (named by
`output_column`) and keeps the dataset lazy. `num_gpus` reserves accelerator fraction
per worker, `concurrency` sets the worker count (or an autoscaling `(min, max)` range),
and `batch_size` controls how many texts go through the model at once. Needs
`batcher-engine[st]`.

```python
# docs: skip
import batcher as bt

ds = bt.read.parquet("s3://bucket/docs.parquet")
vectors = ds.ml.embed_text("text", "sentence-transformers/all-MiniLM-L6-v2", num_gpus=1)
vectors.write.lance("s3://bucket/vectors.lance")
```

For a model Batcher does not wrap directly, `ds.ml.embed` takes any load-once callable
or class that maps a batch to vectors, and `ds.ml.infer` does the same for general
model scoring — both accept `concurrency`, `num_gpus`, and `batch_size` the same way.

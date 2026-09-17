# LLM inference

This section covers running a language model over a column: the generation call, the prompts and conversations around it, and the engines and throughput decisions underneath.

The workload is offline generation over millions of rows: summarizing tickets, labeling a corpus, extracting fields from documents, synthesizing training data. An LLM engine already does the hard GPU work, continuous batching and a KV cache, so Batcher doesn't try to replace it. It loads the engine once per worker, feeds it whole request lists without imposing an outer batch size that would fight its scheduler, and keeps the columnar work on either side in the data plane: building prompts from row columns before the call, and parsing and scoring the output after it.

## The call

{py:meth}`ds.ml.generate <batcher.api.dataset.ml.DatasetML.generate>` appends a generated column. The engine is a factory, a zero-argument callable that returns a function from a list of prompts to a list of completions, which is why the example below runs without a GPU:

```python
import batcher as bt


def engine_factory():
    return lambda prompts: [f"{len(p)} chars" for p in prompts]


tickets = bt.from_pydict(
    {
        "id": [1, 2, 3],
        "product": ["router", "modem", "router"],
        "body": ["no signal", "slow", "no signal"],
    }
)
out = tickets.ml.generate(
    engine_factory,
    prompt_column="body",
    template="Classify this {product} ticket: {body}",
    dedup=True,
)
print(out.sort("id").to_pydict()["response"])
# ['38 chars', '32 chars', '38 chars']
```

`template` builds each prompt from the row's own columns. `dedup=True` sends each distinct prompt to the engine once and copies the result to every row that repeats it, which pays off with deterministic decoding over a corpus full of duplicates. For a real model, replace the factory with `bt.ml.vllm_engine("<model-id>", chat=True)` and add `num_gpus=1`.

## Built for a million-row job

Per-row controls ride in columns, so one pass can mix very different requests. `max_tokens_column` gives a 16-token classification and a 2,000-token summary their own budgets, `temperature_column` does the same for sampling, and `adapter_column` routes each row to its own LoRA adapter on a single vLLM engine. `usage=True`, `finish_reason=True` and `logprobs=True` append token counts, truncation flags and the model's confidence as columns, so cost accounting and routing uncertain rows to review are queries rather than log scraping.

Failures stay contained. `max_errored_rows` drops a bounded number of failing rows instead of failing the job, `max_retries` retries a batch whose engine call raises, and every hosted engine takes `requests_per_minute` and `tokens_per_minute` to stay under a provider's rate limit.

On an 8xT4 Ray cluster with warm engine pools, HF gpt2 generated at 814.8 prompts per second with full output agreement. {doc}`/benchmarks/results/ai-and-gpu` has the measurement.

## Engines

The following table lists the engine factories in `batcher.ml`. All of them satisfy the same contract, so switching backends is a one-argument change:

| Factory | Runs | Extra |
|---|---|---|
| {py:func}`vllm_engine <batcher.ml.vllm_engine>` | vLLM on local GPUs, with guided JSON, regex, choice and grammar output, and multi-LoRA | `vllm` |
| {py:func}`sglang_engine <batcher.ml.sglang_engine>` | SGLang on local GPUs, whose radix cache reuses any shared prompt prefix | `sglang` |
| {py:func}`http_engine <batcher.ml.http_engine>` | Any OpenAI-compatible endpoint, such as a vLLM server or llama.cpp | none |
| {py:func}`anthropic_engine <batcher.ml.anthropic_engine>` | Claude models over the Anthropic Messages API | none |
| {py:func}`bedrock_engine <batcher.ml.bedrock_engine>` | Any model on AWS Bedrock, through the Converse API | `aws` |
| {py:func}`gemini_engine <batcher.ml.gemini_engine>` | Gemini models, or Vertex AI through `base_url` | none |

## In this section

The following table lists the pages in this section:

| Page | Covers |
|---|---|
| {doc}`/ml/retrieval/llm/calling` | The shapes a generation call takes: on a Dataset, chat templates, sequence packing, the streaming form, and the class UDF. |
| {doc}`/ml/retrieval/llm/prompts` | Rows with no prompt, building prompts from columns, reading a conversation column, and staying inside the context window. |
| {doc}`/ml/retrieval/llm/engines` | Each engine in depth, choosing between vLLM and SGLang, provider quotas, sizing a model across GPUs, and vision-language models. |

## See also

- {doc}`/ml/retrieval/llm-outputs`: parse generated text into typed columns, with {py:meth}`ds.ml.extract <batcher.api.dataset.ml.DatasetML.extract>` and {py:meth}`ds.ml.classify <batcher.api.dataset.ml.DatasetML.classify>`.
- {doc}`/ml/retrieval/llm-evaluation`: score generations with and without a reference.
- {doc}`/ml/retrieval/rag`: put retrieval in front of generation.
- {doc}`/ml/inference/gpu`: how `num_gpus` and `concurrency` map to actors.
- {doc}`/ml/training/serving`: call a model served elsewhere.

```{toctree}
:hidden:

calling
prompts
engines
```

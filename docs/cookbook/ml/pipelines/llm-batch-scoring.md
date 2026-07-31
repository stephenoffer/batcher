# LLM batch scoring

You have two million support tickets and you want a severity label and a refund amount on
each one. The model part is easy. What breaks these jobs is the column you get back: a
string. `"Positive."`, `"positive"`, and `"The sentiment is positive."` are three
categories to a `GROUP BY`, and a struct type inferred per batch will blow up at concat
time, after the GPU work is paid for.

## Label a column, with the domain pinned

`ds.ml.classify(engine, labels=[...])` resolves the model's answer against the declared
label set and **nulls anything that does not resolve to exactly one**. The output column's
domain is the list you passed, not whatever the model felt like saying.

An engine is a zero-argument callable returning a `list[str] -> list[str]` function.
That contract is why a stub stands in for a model with no GPU, and why the pipeline you
test locally is the pipeline that runs on the cluster.

```python
import batcher as bt

tickets = bt.from_pydict(
    {
        "id": [1, 2, 3, 4],
        "body": [
            "The checkout page 500s on every card.",
            "Love the new dashboard, thanks!",
            "Charged twice for the same order, want a refund.",
            "How do I change my avatar?",
        ],
    }
)


def stub_engine():  # an EngineFactory: built once per worker
    def engine(prompts):
        out = []
        for p in prompts:
            text = p.lower()
            if "500" in text or "charged twice" in text:
                out.append("Urgent.")
            else:
                out.append("routine")
        return out

    return engine


triaged = tickets.ml.classify(
    stub_engine, labels=["urgent", "routine"], prompt_column="body", output_column="severity"
)
print(triaged.to_pydict()["severity"])
# ['urgent', 'routine', 'urgent', 'routine']
```

`"Urgent."` came back with punctuation and capitalization and still landed as `urgent`.
Taken verbatim it would have been a fourth category nobody expected.

## Swap the stub for a real engine

The stub and the model are the same call. Only the engine and the placement keywords change,
and the choice below is about where the tokens are decoded, not about how the pipeline is
written.

::::{tab-set}
:::{tab-item} vLLM, on your GPUs

Set `chat=True` for any instruction-tuned model. The default completion path skips the chat
template, and a tuned model answering an untemplated prompt degrades silently.

```python
# docs: skip
from batcher.ml import vllm_engine

engine = vllm_engine(
    "meta-llama/Llama-3-8B-Instruct",
    chat=True,
    system="You are a precise data labeler.",
    sampling={"temperature": 0.0, "max_tokens": 8},
)
triaged = tickets.ml.classify(
    engine,
    labels=["urgent", "routine"],
    prompt_column="body",
    output_column="severity",
    num_gpus=1,
    concurrency=4,
)
```
:::

:::{tab-item} A hosted endpoint

`http_engine` swaps the local model for an OpenAI-compatible endpoint. Nothing else about
the pipeline changes: same `classify`, same `extract`, same joins. Throughput is then
bounded by the endpoint rather than by your GPUs, and `concurrency` sets how many request
streams you open against it.

```python
# docs: skip
from batcher.ml import http_engine

engine = http_engine("https://api.example.com/v1", "gpt-4o-mini", api_key="sk-...", max_tokens=64)
triaged = tickets.ml.classify(
    engine, labels=["urgent", "routine"], prompt_column="body", concurrency=8
)
```
:::
::::

Warm pools matter most on the local path: the model loads once per session and is reused
across calls, which is what keeps gpt2 batch generation at 814.8 prompt/s on 8xT4 rather than
paying a 7-second load on every execution.

## Typed columns, not JSON blobs

`ds.ml.extract(engine, schema=...)` appends one typed Arrow column per declared field. The
**declaration** decides the type, not what the model emitted in a given batch.

:::{warning}
That is the difference from `generate(parse_json=True)`, whose struct type is inferred per
batch: ask for `{vendor, total}`, have the model omit `total` on one batch, and the two
batches carry incompatible struct types and the scan dies at concat, after the GPU work is
already paid for.
:::

A row that will not parse degrades to nulls in its own columns. One bad generation over a
million rows costs one row, and it is countable.

| Call | What it appends | What pins the type |
| --- | --- | --- |
| `ds.ml.classify(engine, labels=[...])` | one column whose domain is the label list | the `labels` you declared; anything that does not resolve to exactly one is null |
| `ds.ml.extract(engine, schema={...})` | one typed Arrow column per field | the `schema` you declared; an unparseable row is nulls in its own columns |
| `ds.ml.generate(engine, ...)` | the raw text the model produced | nothing, because with `parse_json=True` the struct type is inferred per batch |

```python
import batcher as bt
from batcher import col

notes = bt.from_pydict(
    {
        "id": [1, 2, 3],
        "note": [
            "Paid 42 USD to Acme on Tuesday",
            "Invoice from Globex for 1200",
            "no idea what this is",
        ],
    }
)


def extractor():
    def engine(prompts):
        out = []
        for p in prompts:
            if "Acme" in p:
                out.append('{"vendor": "Acme", "total": "42"}')
            elif "Globex" in p:
                out.append('{"vendor": "Globex", "total": 1200}')
            else:
                out.append("I could not find a vendor.")  # unparseable
        return out

    return engine


parsed = notes.ml.extract(
    extractor, schema={"vendor": "string", "total": "float64"}, prompt_column="note"
)
result = parsed.to_pydict()
print(result["vendor"], result["total"])
# ['Acme', 'Globex', None] [42.0, 1200.0, None]
print(parsed.filter(col("total").is_null()).count())
# 1
```

Note the coercion: `"42"` arrived as a *string* and landed in a `float64` column. The
schema wins.

Pair `extract` with guided decoding so the rows parse in the first place. `json_schema`
turns the same declaration into a JSON Schema vLLM can constrain against:

```python
# docs: skip
from batcher.ml import json_schema, vllm_engine

schema = {"vendor": "string", "total": "float64"}
engine = vllm_engine("meta-llama/Llama-3-8B-Instruct", chat=True, guided_json=json_schema(schema))
invoices = (
    bt.read.parquet("s3://bucket/invoices.parquet")
    .ml.extract(engine, schema=schema, prompt_column="body", num_gpus=1, concurrency=8)
    .write.parquet("s3://bucket/invoices_typed.parquet")
)
```

## Do not pay twice for the same prompt

:::{tip}
A production corpus is not a set. The same product description, the same boilerplate
footer, the same auto-generated ticket text appears thousands of times, and every copy is
a full decode on a GPU you are renting by the hour. Score the *distinct* prompts and join
the answers back.
:::

This is ordinary relational work, which is the point: the LLM stage is another
operator, so the optimizer and the join sit on either side of it.

```python
import batcher as bt

boilerplate = "reset my password"
outage = "The checkout page 500s on every card."
raw = bt.from_pydict(
    {
        "row_id": [1, 2, 3, 4, 5],
        "body": [boilerplate, boilerplate, outage, boilerplate, outage],
    }
)

unique_prompts = raw.select("body").distinct()
answered = unique_prompts.ml.classify(
    stub_engine, labels=["urgent", "routine"], prompt_column="body", output_column="severity"
)
scored = raw.join(answered, on="body", how="left")
print(unique_prompts.count(), raw.count())
# 2 5
print(sorted(zip(scored.to_pydict()["row_id"], scored.to_pydict()["severity"])))
# [(1, 'routine'), (2, 'routine'), (3, 'urgent'), (4, 'routine'), (5, 'urgent')]
```

Two generations instead of five. On a real corpus the ratio is usually far worse than
that, and this one rewrite is the cheapest optimization available to an LLM ETL job.
A shared instruction prefix is the other one: `vllm_engine` enables prefix caching by
default, so a long fixed preamble in a `template` is encoded once rather than per row.

## Prompts built from several columns

`template` is a `str.format` string over the row's columns, so the prompt assembles in the
engine and `prompt_column` is ignored.

:::{dropdown} Templating a prompt, and costing the job with `usage=True`

```python
# docs: skip
summaries = tickets.ml.generate(
    engine,
    template="Summarize this {product} ticket in one sentence:\n\n{body}",
    prompt_column="body",
    output_column="summary",
    usage=True,  # also append prompt_tokens / completion_tokens
    num_gpus=1,
)
```

`usage=True` gives you the token counts as columns, which is how you cost the job: a
`sum()` over `completion_tokens` is a bill.
:::

## See also

- {doc}`LLM inference </ml/retrieval/llm/index>`: engines, structured output, sequence packing.
- {doc}`Batch scoring </ml/inference/batch-scoring>`: the general model-over-a-table surface.
- {doc}`Text embeddings </cookbook/ml/pipelines/text-embeddings>`: the retrieval half of a RAG pipeline.
- {doc}`RAG index </cookbook/ml/pipelines/rag-index>`: generation over retrieved context.
- {doc}`ML API reference </api/models/ml>`: `classify`, `extract`, `generate`, `vllm_engine`,
  `http_engine`, `json_schema`.
- {doc}`AI and GPU benchmarks </benchmarks/ai-and-gpu>`: the LLM batch-inference
  throughput, and the hardware ceiling behind it.
- {doc}`HuggingFace integration </integrations/compute/huggingface>`: model ids, tokenizers, and
  datasets.
- {doc}`GPU execution </deep-dives/distribution/gpu-execution>`: why the pool is warm and the stages
  overlap.

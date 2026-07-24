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

## Turning generated strings into typed columns

Generation gives you a *string*. No analyst can filter, join, or aggregate a string, so
turning it into a column is the actual ETL step, and it is where these pipelines break.
Two Dataset methods do it.

`ds.ml.extract(engine, schema=...)` appends one **typed** column per declared field. The
declaration decides the Arrow type, not whatever the model happened to emit:

```python
import batcher as bt

notes = bt.from_pydict({"note": ["Paid 42 USD to Acme"]})
stub = lambda: (lambda ps: ['{"vendor": "Acme", "total": "42"}'] * len(ps))
print(notes.ml.extract(stub, schema={"vendor": "string", "total": "float64"}, prompt_column="note").to_pydict())
# {'note': ['Paid 42 USD to Acme'], 'vendor': ['Acme'], 'total': [42.0]}
```

This is why `extract` exists rather than `generate(parse_json=True)`. `parse_json` infers
the struct type from whatever came back **in that batch**. Ask for `{label, score}`, have
the model omit `score` on one batch, and the two batches carry incompatible struct types.
The scan then dies at concat time with the GPU work already paid for. A declared schema
pins every batch to the same types and makes the missing value a null. In the example
above, `"42"` came back as a *string* and landed in a `float64` column, because values are
coerced per row.

Failures degrade one row, never the batch. An unparseable response, a missing key, or a
value that will not coerce becomes null, and the damage is countable:

```python
# docs: skip
bad = extracted.filter(bt.col("total").is_null()).count()
```

`ds.ml.classify(engine, labels=[...])` labels each row with exactly one of `labels`. A
model asked for `"positive"` will answer `"Positive."` or `"The sentiment is positive."`.
Taken verbatim those give a category column with a long tail that never groups together.
`classify` resolves the answer against the declared set and **nulls anything else**, so the
column's domain is exactly `labels`:

```python
import batcher as bt

reviews = bt.from_pydict({"review": ["loved it", "awful"]})
stub = lambda: (lambda ps: ["Positive." if "loved" in p else "negative" for p in ps])
print(reviews.ml.classify(stub, labels=["positive", "negative"], prompt_column="review").to_pydict())
# {'review': ['loved it', 'awful'], 'label': ['positive', 'negative']}
```

Pair `extract` with guided decoding so that every row parses in the first place.
`json_schema(schema)` builds the JSON Schema for you:

```python
# docs: skip
from batcher.ml import json_schema, vllm_engine

schema = {"vendor": "string", "total": "float64"}
engine = vllm_engine("meta-llama/Llama-3-8B", guided_json=json_schema(schema))
invoices = bt.read.parquet("s3://bucket/invoices.parquet").ml.extract(
    engine, schema=schema, prompt_column="body", num_gpus=1
)
```

Both lower to `map_batches`, so they are linear maps. They stream, they distribute across
GPU actors, and they compose with the rest of the engine the way any other projection
does.

### Parsing without a second model call

`extract` and `classify` call a model to reshape the text. When the fragment you want is
already *in* the generated string, a regex expression pulls it out in the same scan with no
GPU and no second inference pass. These are ordinary scalar functions, so they vectorize,
push down, and compose with any other expression. Each returns an empty string where the
fragment is absent, so a malformed row degrades to a filterable empty rather than an error.

`extract_json` and `extract_json_array` recover the JSON a model wrapped in prose, which is
the common case that breaks a bare `json.loads`. `extract_code_block` drops the triple-backtick
fences and language tag from a returned snippet. `extract_first_number` parses the first
numeric span to a float, for a model asked to score or count in free text.

```python
import batcher as bt

out = bt.from_pydict(
    {
        "reply": [
            'Sure! Here is the data: {"vendor": "Acme", "total": 42}',
            "The score is 87 out of 100.",
        ]
    }
)
print(
    out.select(
        obj=bt.extract_json("reply"),
        score=bt.extract_first_number("reply"),
    ).to_pydict()
)
# {'obj': ['{"vendor": "Acme", "total": 42}', ''], 'score': [42.0, 87.0]}
```

Reasoning models fence their chain of thought. `extract_reasoning` reads the `<think>...</think>`
trace and `strip_reasoning` removes it to leave the user-facing answer. `extract_tag` reads any
named XML-style tag, the convention prompts use to mark a final answer:

```python
traces = bt.from_pydict(
    {"out": ["<think>2+2 is 4</think><answer>4</answer>"]}
)
print(
    traces.select(
        why=bt.extract_reasoning("out"),
        answer=bt.extract_tag("out", "answer"),
        clean=bt.strip_reasoning("out"),
    ).to_pydict()
)
# {'why': ['2+2 is 4'], 'answer': ['4'], 'clean': ['<answer>4</answer>']}
```

`extract_after` and `extract_between` slice around literal markers, for the `Answer:` and
delimiter conventions that few-shot prompts create. `extract_choice` reads a standalone
multiple-choice letter, and `is_refusal` flags the common refusal phrasings so you can measure
a refusal rate or filter them out before scoring:

```python
graded = bt.from_pydict(
    {
        "resp": [
            "Reasoning aside, the answer is C.",
            "I'm sorry, I can't help with that.",
        ]
    }
)
print(
    graded.select(
        choice=bt.extract_choice("resp"),
        refused=bt.is_refusal("resp"),
    ).to_pydict()
)
# {'choice': ['C', ''], 'refused': [False, True]}
```

## Structured output

Constrain generation to a JSON schema so every row is parseable, then parse it into a
struct column. `guided_json` on the engine forces the model's decoding to the schema,
and `parse_json=True` on `llm_generate` parses each output into a struct. A row that
fails to parse gets a null rather than failing the batch. Prefer `ds.ml.extract` above
when the fields are known, because it pins the Arrow types. Pair the two so that guided
decoding makes the output well-formed and `parse_json` turns it into typed columns you
can query downstream.

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
a regular expression such as `r"\d{4}-\d{2}-\d{2}"` for a date.

Guided decoding is not always available, so measure whether the output actually held its shape.
`bt.valid_json_rate` is the strict JSON-mode compliance rate (the whole output parses as JSON),
`bt.json_present_rate` is the lenient rate (a JSON object is recoverable from surrounding prose), and
`bt.tagged_answer_rate` is the compliance rate for a tag-delimited format. Watch them per model or
per prompt version to catch a format regression before the parser starts nulling rows.

Benchmark harnesses grade a specific answer shape, and an output without it is ungradeable rather
than wrong. `bt.numeric_answer_rate` is the fraction with a parseable number (math and counting
tasks), `bt.choice_answer_rate` the fraction with a standalone multiple-choice letter, and
`bt.boxed_answer_rate` the fraction with a LaTeX `\boxed{}` answer (the MATH convention). A low rate
points at the prompt, not the model's reasoning.

```python
outs = bt.from_pydict({"o": ['{"label": "yes"}', "Sure! {\"label\": \"no\"}", "I refuse"]})
print(
    outs.agg(
        strict=bt.valid_json_rate("o"),
        lenient=bt.json_present_rate("o"),
    ).to_pydict()
)
# {'strict': [0.3333333333333333], 'lenient': [0.6666666666666666]}
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
retrieval-pipeline companion to [vector search](vector-search.md). It appends one
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

## Scoring generations against a reference

Evaluating generations is comparing a generated column to a gold column, and the lexical-overlap
metrics for that are expressions that aggregate to a corpus score in one scan — no Python loop
over examples. `bt.exact_match` is the strict character-for-character rate; `bt.normalized_exact_match`
applies SQuAD normalization first (lowercase, drop articles and punctuation), so casing and a
trailing period do not count against a correct answer.

```python
import batcher as bt

evals = bt.from_pydict(
    {"answer": ["The capital is Paris.", "It is Rome"], "gold": ["Paris", "London"]}
)
print(evals.agg(em=bt.normalized_exact_match("answer", "gold")).to_pydict())
```

For free-form answers where neither exact match nor a single word is right, the token metrics
compare the *sets* of words (repeats counted once): `bt.token_set_precision`, `bt.token_set_recall`,
`bt.token_set_f1` (the balanced default), and `bt.token_set_jaccard`. `bt.length_ratio` reports how
verbose the output is relative to the reference, which catches a model that systematically over- or
under-generates. Every one composes with `group_by` to score per model, per prompt template, or per
slice in the same pass.

```python
scored = bt.from_pydict(
    {"model": ["a", "a", "b", "b"],
     "answer": ["the quick brown fox", "yes", "a slow brown fox", "no"],
     "gold": ["a fast brown fox", "yes", "the brown fox", "yes"]}
)
print(scored.group_by("model").agg(f1=bt.token_set_f1("answer", "gold")).sort("model").to_pydict())
```

These are set-based by design, so they are stable and fast rather than a multiset BLEU/ROUGE score;
each metric's docstring states this so it is never confused with one.

The token-set metrics split on whitespace, which fails on a language that does not put spaces
between words. `bt.char_ngram_f1` scores the overlap of *character* n-grams instead, the idea behind
chrF, so it works on Chinese, Japanese, or heavily inflected output with no tokenizer.
`bt.char_ngram_precision`, `bt.char_ngram_recall`, and `bt.char_ngram_jaccard` are the matching
directional and set-similarity views.

```python
cjk = bt.from_pydict({"pred": ["東京都"], "gold": ["東京市"]})
print(cjk.agg(chrf=bt.char_ngram_f1("pred", "gold", n=2)).to_pydict())
# {'chrf': [0.5]}
```

### Scoring generations without a reference

Most generations arrive with no gold answer to compare against, and the questions you still want
answered are about the output itself: is it diverse or repeating, how long is it, and how often is
it empty, a refusal, or cut off. These metrics take one output column and aggregate to a corpus
number, so they run over a million generations in one scan and break down per model or per day with
`group_by`.

`bt.distinct_token_ratio` is the Distinct-1 diversity score, the cheap detector of a model
degenerating into repetition. `bt.mean_output_tokens` tracks verbosity and sizes the token bill.
`bt.empty_generation_rate`, `bt.refusal_rate`, and `bt.truncation_rate` are the three failure rates
worth a dashboard: silent empty outputs, declined answers, and responses that stop mid-sentence.

```python
gens = bt.from_pydict(
    {
        "out": [
            "The capital of France is Paris.",
            "yes yes yes yes yes",
            "I'm sorry, I can't help with that.",
            "The list of steps is as follows",
        ]
    }
)
print(
    gens.agg(
        diversity=bt.distinct_token_ratio("out"),
        refused=bt.refusal_rate("out"),
        truncated=bt.truncation_rate("out"),
    ).to_pydict()
)
# {'diversity': [0.8], 'refused': [0.25], 'truncated': [0.5]}
```

They are lexical heuristics, so read them as monitors that catch a regression between runs, not as
ground-truth judgments of a single generation.

Before a run rather than after it, the token aggregates size the bill and the capacity.
`bt.total_token_estimate` sums the corpus token estimate for a cost number, `bt.token_budget_exceed_rate`
is the fraction of rows that will overflow a given context window, and `bt.token_estimate_quantile`
is the length tail that sizes the window. All take either the prompt or the output column.

```python
reqs = bt.from_pydict({"prompt": ["short one", "a considerably longer prompt string here"]})
print(
    reqs.agg(
        total=bt.total_token_estimate("prompt"),
        over=bt.token_budget_exceed_rate("prompt", budget=5),
    ).to_pydict()
)
# {'total': [12], 'over': [0.5]}
```

A second set of monitors watches for output a model *should not* produce at scale. `bt.all_caps_rate`
and `bt.repeated_punctuation_rate` catch shouting and degenerate punctuation, `bt.non_ascii_rate`
flags encoding or language drift, `bt.url_rate` surfaces hallucinated links or prompt injection, and
`bt.code_block_rate` catches a code block leaking into a prose task. `bt.long_output_rate` and
`bt.short_output_rate` bound the length distribution, and `bt.mean_sentence_count` and
`bt.mean_word_length` track structural and lexical drift.

```python
outputs = bt.from_pydict(
    {"out": ["STOP.", "see https://spam.example", "a normal, useful answer here"]}
)
print(
    outputs.agg(
        shouting=bt.all_caps_rate("out"),
        links=bt.url_rate("out"),
    ).to_pydict()
)
# {'shouting': [0.3333333333333333], 'links': [0.3333333333333333]}
```

### More output monitors

Four further families of single-scan monitors cover the rest of what a generation-at-scale team
watches, and all compose with `group_by`.

For a RAG pipeline, compare the answer column against its retrieved context. `bt.answer_groundedness`
is the share of the answer's tokens the context supports, `bt.context_utilization` the share of the
context the answer drew on, `bt.unsupported_token_rate` the hallucination-proxy complement, and
`bt.fully_grounded_rate` the fraction of answers entirely supported. `bt.citation_rate` tracks how
often the model cited a source at all.

For reading level, `bt.automated_readability_index` is the ARI grade, with `bt.mean_words_per_sentence`,
`bt.mean_chars_per_word`, `bt.long_word_rate`, and `bt.mean_paragraph_count` as the complexity drivers
behind it.

For degeneration, `bt.distinct_char_ngram_ratio` and its complement `bt.char_repetition_rate` catch a
model looping at the character level, `bt.word_type_token_ratio` at the word level,
`bt.repeated_line_rate` catches duplicated lines, and `bt.compression_ratio_proxy` is a cheap
gzip-style repetition score.

For safety, `bt.email_rate`, `bt.phone_rate`, and `bt.pii_rate` flag leaked contact details,
`bt.ssn_like_rate` and `bt.credit_card_like_rate` catch structured identifiers, and
`bt.contains_any_rate` is a configurable blocklist monitor over a list of terms.

For formatting, `bt.heading_rate`, `bt.bullet_list_rate`, `bt.numbered_list_rate`,
`bt.markdown_link_rate`, `bt.table_rate`, and `bt.code_block_present_rate` check whether the model
produced the Markdown elements a task asked for.

For tone, `bt.question_rate` catches a model deflecting an answer task with a question,
`bt.exclamation_rate` and `bt.politeness_rate` track register, `bt.hedge_rate` flags uncertainty,
`bt.first_person_rate` measures first-person voice, and `bt.contains_phrase_rate` is a configurable
phrase monitor.

For language, `bt.cjk_rate`, `bt.cyrillic_rate`, and `bt.arabic_rate` flag unexpected scripts,
`bt.emoji_rate` catches emoji spam, and `bt.latin_only_rate` is the clean-ASCII-output rate.

## Next steps

- [Inference](inference.md): the general batch-inference and embedding path.
- [Serving](serving.md): expose a model behind an endpoint.
- [GPU scheduling](gpu.md): how `num_gpus` and `concurrency` map to actors.

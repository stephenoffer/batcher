# Parsing LLM output

Generation gives you a string. This page covers turning that string into typed columns
you can filter, join, and aggregate, which is the step where these pipelines usually
break.

Read {doc}`/ml/retrieval/llm/index` first for how the generation itself runs.

## Extracting typed columns

No analyst can filter, join, or aggregate a string, so turning it into a column is the
actual ETL step. Two Dataset methods do it.

{py:meth}`ds.ml.extract(engine, schema=...) <batcher.api.dataset.ml.DatasetML.extract>` appends one **typed** column per declared field. The
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

A field can declare a string, a number, a boolean, or a **date, time, timestamp, or binary**
type. A model answers a date question with an ISO string, because that is what dates look
like in the text it learned from, so that string becomes a real `date32` column you can
compare, filter, and group by:

```python
import batcher as bt

invoices = bt.from_pydict({"note": ["Invoice 7 dated 2024-01-05, total 42 USD"]})
stub = lambda: (lambda ps: ['{"due": "2024-01-05", "total": 42}'] * len(ps))
typed = invoices.ml.extract(
    stub, schema={"due": "date32", "total": "float64"}, prompt_column="note"
)
print(typed.schema.field("due").type)
# date32[day]
print(typed.to_pydict()["due"])
# [datetime.date(2024, 1, 5)]
```

A dtype that no JSON output can fill, such as `duration` or `interval`, is rejected when you
declare it. That is deliberate: the alternative is a column of nulls handed back after the
generation is already paid for, under a schema that looks exactly right.

Failures degrade one row, never the batch. An unparseable response, a missing key, or a
value that will not coerce becomes null, and the damage is countable:

```python
# docs: skip
bad = extracted.filter(bt.col("total").is_null()).count()
```

{py:meth}`ds.ml.classify(engine, labels=[...]) <batcher.api.dataset.ml.DatasetML.classify>` labels each row with exactly one of `labels`. A
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

## Parsing without a second model call

`extract` and `classify` call a model to reshape the text. When the fragment you want is
already *in* the generated string, a regex expression pulls it out in the same scan with no
GPU and no second inference pass. These are ordinary scalar functions, so they vectorize,
push down, and compose with any other expression. Each returns an empty string where the
fragment is absent, so a malformed row degrades to a filterable empty rather than an error.

{py:func}`extract_json <batcher.extract_json>` and {py:func}`extract_json_array <batcher.extract_json_array>` recover the JSON a model wrapped in prose, which is
the common case that breaks a bare `json.loads`. {py:func}`extract_code_block <batcher.extract_code_block>` drops the triple-backtick
fences and language tag from a returned snippet. {py:func}`extract_first_number <batcher.extract_first_number>` parses the first
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

Reasoning models fence their chain of thought. {py:func}`extract_reasoning <batcher.extract_reasoning>` reads the `<think>...</think>`
trace and {py:func}`strip_reasoning <batcher.strip_reasoning>` removes it to leave the user-facing answer. {py:func}`extract_tag <batcher.extract_tag>` reads any
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

{py:func}`extract_after <batcher.extract_after>` and {py:func}`extract_between <batcher.extract_between>` slice around literal markers, for the `Answer:` and
delimiter conventions that few-shot prompts create. {py:func}`extract_choice <batcher.extract_choice>` reads a standalone
multiple-choice letter, and {py:func}`is_refusal <batcher.is_refusal>` flags the common refusal phrasings so you can measure
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

Three more read the answer conventions that {doc}`llm-evaluation` already measures the
*compliance* of. {py:func}`bt.extract_boxed <batcher.extract_boxed>` reads the LaTeX `\boxed{}` a math benchmark grades on, and
{py:func}`bt.extract_last_number <batcher.extract_last_number>` reads the conclusion of a reasoning chain — which is not the same as
{py:func}`bt.extract_first_number <batcher.extract_first_number>`, because a model that reasons before answering emits its intermediate
quantities first.

```python
math_answers = bt.from_pydict({"out": ["12 apples minus 4 leaves \\boxed{8}"]})
print(
    math_answers.select(
        boxed=bt.extract_boxed("out"),
        first=bt.extract_first_number("out"),
        last=bt.extract_last_number("out"),
    ).to_pydict()
)
# {'boxed': ['8'], 'first': [12.0], 'last': [8.0]}
```

{py:func}`bt.extract_citations <batcher.extract_citations>` returns every `[n]` marker as a list, which is what turns a citation rate
into a citation *check*: set-subtract the retrieved passage ids and whatever is left is a
reference to a source that was never retrieved.

```python
answered = bt.from_pydict(
    {"answer": ["backed by [1] and [9]"], "retrieved": [["1", "2"]]}
)
print(
    answered.select(
        fabricated=bt.extract_citations("answer").list.set_difference(bt.col("retrieved"))
    ).to_pydict()
)
# {'fabricated': [['9']]}
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
{py:func}`bt.valid_json_rate <batcher.valid_json_rate>` is the strict JSON-mode compliance rate (the whole output parses as JSON),
{py:func}`bt.json_present_rate <batcher.json_present_rate>` is the lenient rate (a JSON object is recoverable from surrounding prose), and
{py:func}`bt.tagged_answer_rate <batcher.tagged_answer_rate>` is the compliance rate for a tag-delimited format. Watch them per model or
per prompt version to catch a format regression before the parser starts nulling rows.

Benchmark harnesses grade a specific answer shape, and an output without it is ungradeable rather
than wrong. {py:func}`bt.numeric_answer_rate <batcher.numeric_answer_rate>` is the fraction with a parseable number (math and counting
tasks), {py:func}`bt.choice_answer_rate <batcher.choice_answer_rate>` the fraction with a standalone multiple-choice letter, and
{py:func}`bt.boxed_answer_rate <batcher.boxed_answer_rate>` the fraction with a LaTeX `\boxed{}` answer (the MATH convention). A low rate
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

## See also

- {doc}`/ml/retrieval/llm/index`: running the generation these outputs come from.
- {doc}`/ml/retrieval/llm-evaluation`: scoring the parsed results.
- {doc}`/user-guide/transform/columns/expression-accessors`: the {py:class}`.json <batcher.plan.expr_ir.namespaces.collections._JsonNamespace>` accessor these methods build on.

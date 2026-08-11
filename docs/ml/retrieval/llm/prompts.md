# Prompts and conversations

Building the input, and reading a conversation column back out.

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

To build the prompt column as its own expression before generation, {py:func}`bt.render_template <batcher.render_template>` fills
named `{placeholder}` slots from columns, {py:func}`bt.wrap_tag <batcher.wrap_tag>` surrounds a field in `<tag>...</tag>` for a
structured prompt, and {py:func}`bt.truncate_to_token_budget <batcher.truncate_to_token_budget>` trims a column to fit the context window. They
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

{py:func}`bt.tagged_fields <batcher.tagged_fields>` is the multi-field form of `bt.wrap_tag`: one delimited block per column, so
a value containing punctuation or newlines cannot be mistaken for the next section.
{py:func}`bt.join_context <batcher.join_context>` is the step between retrieval and generation, folding a list column of
retrieved passages into one block and dropping the empty entries a short retrieval leaves.

```python
rag = bt.from_pydict(
    {"q": ["Why is the sky blue?"], "hits": [["Rayleigh scattering.", "", "Shorter wavelengths."]]}
)
print(
    rag.select(
        prompt=bt.tagged_fields(
            question=bt.col("q"), context=bt.join_context(bt.col("hits"))
        )
    ).to_pydict()["prompt"][0]
)
```

For a completions endpoint or a local engine that wants a rendered string rather than a message
list, {py:func}`bt.chatml_prompt <batcher.chatml_prompt>` renders a row's turns in the ChatML format and {py:func}`bt.instruction_prompt <batcher.instruction_prompt>`
in the Alpaca instruction layout. Check which template your model was trained on: a mismatch
degrades quality quietly rather than raising.

### Reading a conversation column

A chat log, a fine-tuning set, and an agent trace all arrive the same way: a list of
`{role, content}` structs per row. Every question you want to ask of that column looks like a
per-row loop, and none of it needs to be — a list of structs is a columnar value, and the
{py:class}`.list <batcher.plan.expr_ir.namespaces.collections._ListNamespace>` higher-order functions reach inside it.

{py:func}`bt.conversation_turns <batcher.conversation_turns>` counts messages, optionally for one role. Counting the assistant's
turns is the more useful form: a log full of user messages with no answers is a collection
failure, not a short conversation, and length alone cannot tell them apart.

{py:func}`bt.ends_with_role <batcher.ends_with_role>` is the completeness check for a fine-tuning corpus. A conversation ending
on a user turn was cut off — the export stopped mid-exchange, or the reply failed and was never
written — and training on it teaches the model to stop where an answer should start. Those rows
look identical to complete ones by every other measure.

```python
chats = bt.from_pydict(
    {
        "msgs": [
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
            [{"role": "user", "content": "hi"}],
        ]
    }
)
print(
    chats.select(
        turns=bt.conversation_turns("msgs"),
        replies=bt.conversation_turns("msgs", "assistant"),
        complete=bt.ends_with_role("msgs"),
    ).to_pydict()
)
# {'turns': [2, 1], 'replies': [1, 0], 'complete': [True, False]}
```

{py:func}`bt.last_message <batcher.last_message>` pulls one turn out. Without a role it is how the conversation ended; with
`"user"` it is the request the final answer responded to, and with `"assistant"` the answer
itself — which is the pair every generation metric on {doc}`/ml/retrieval/llm-evaluation` wants. A
conversation with no message of that role yields null rather than an empty string, so those
rows stay countable instead of scoring as empty answers.

{py:func}`bt.render_messages <batcher.render_messages>` flattens the whole exchange to text, one message per line prefixed by its
role, for a spot check or a lexical metric over the conversation rather than its last turn.

```python
print(chats.select(text=bt.render_messages("msgs")).to_pydict()["text"][0])
# user: hi
# assistant: hello
```

The `role` and `content` field names are parameters, because the convention is not universal
and renaming a struct field to fit a hard-coded assumption is a materialization nobody should
have to pay for.

### Staying inside the context window

Overrunning a context window rarely raises. The serving stack truncates the prompt, or leaves so
few tokens that the answer stops mid-sentence, and the run finishes looking successful.

{py:func}`bt.prompt_token_estimate <batcher.prompt_token_estimate>` prices an assembled prompt from its parts before the concatenation
exists as a column, which is what you want to route long rows to a larger-window model or to
sort a batch by length so a continuous-batching engine packs it well. {py:func}`bt.fits_context <batcher.fits_context>` is the
filter, and its `reserve_output` is the part a bare length check misses: a prompt that fills the
window exactly cannot be answered at all.

```python
docs = bt.from_pydict({"sys": ["Be brief."], "body": ["a" * 4000]})
print(
    docs.select(
        tokens=bt.prompt_token_estimate(bt.col("sys"), bt.col("body")),
        ok=bt.fits_context("body", window=1000, reserve_output=256),
    ).to_pydict()
)
```

When a row does not fit, `bt.truncate_to_token_budget` cuts the tail and {py:func}`bt.truncate_middle <batcher.truncate_middle>`
keeps both ends, replacing the middle with a marker. Prefer the middle cut for a contract, a
transcript, or a log, where the last paragraph is often the one holding the answer.

```python
long_doc = bt.from_pydict({"body": ["START " + "x" * 200 + " END"]})
trimmed = long_doc.select(t=bt.truncate_middle("body", budget=8)).to_pydict()["t"][0]
print(trimmed.startswith("START"), trimmed.endswith("END"))
# True True
```

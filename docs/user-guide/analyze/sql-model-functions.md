# Model and AI functions in SQL

Batcher's SQL surface carries two kinds of table function that call a model: `ML_PREDICT` for a fitted traditional model, and `AI_GENERATE` and `AI_EXTRACT` for a language model. This page covers both, including why each is written in the `FROM` clause rather than in the `SELECT` list. For the rest of the SQL surface, see {doc}`sql`.

Every example here runs against a {py:obj}`bt.Session <batcher.Session>`, which holds the catalog a query resolves names against:

```python
import batcher as bt

s = bt.Session()
```

## Scoring a fitted model

A fitted model registers on a session too, with {py:meth}`register_model <batcher.Session.register_model>`, and `ML_PREDICT` then scores a relation inside the query. The prediction is an ordinary column, so the rest of the statement filters, joins and aggregates over it without leaving SQL:

```python
from batcher.ml import LinearRegression

train = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 8.0]})
s.register_model("doubler", LinearRegression(features=["x"], target="y").fit(train))
s.register("points", bt.from_pydict({"x": [5.0, 10.0]}))

print(
    s.sql("SELECT COUNT(*) AS n FROM ML_PREDICT(points, doubler) WHERE prediction > 15").to_pydict()
)
# {'n': [1]}
```

Scoring stays inside the plan, so the model runs where the data is rather than pulling rows back to the driver. A saved model can be named by quoted path instead of registering it first; see the {doc}`SQL API </api/relational/sql>`.

## Generative AI functions

`ML_PREDICT` covers the *traditional* model. `AI_GENERATE` is the generative half: a language
model asked to write from a text column, with `ai_query` and `ai_complete` accepted as aliases
so a query ported from Databricks or Snowflake runs as written. `AI_EXTRACT` is the same shape
for pulling typed fields out of the text.

An engine is registered in Python and named in SQL, never written inline. It carries an
endpoint, credentials and sampling settings, so a quoted engine argument is refused rather than
becoming a way to put an API key in query text:

```python
s = bt.Session()
s.register(
    "reviews", bt.from_pydict({"id": [1, 2, 3], "body": ["love it", "broke fast", "it is fine"]})
)


def shouty():
    # Stands in for `bt.ml.http_engine(...)` / `bt.ml.vllm_engine(...)` so this page needs
    # no model: any zero-argument callable returning `list[str] -> list[str]` is an engine.
    return lambda prompts: [p.upper() for p in prompts]


s.register_engine("shouty", shouty)

print(
    s.sql(
        "SELECT id, response FROM AI_GENERATE(reviews, shouty, prompt_column => 'body')"
    ).to_pydict()
)
# {'id': [1, 2, 3], 'response': ['LOVE IT', 'BROKE FAST', 'IT IS FINE']}
```

The relation and the engine are positional; everything that changes the answer is a named
setting. `AI_GENERATE` takes `prompt_column` (required), `template` and `output_column`.
`AI_EXTRACT` takes `prompt_column` and a `schema` written as a column definition list, and
appends one typed column per field:

```python
import json


def grader():
    return lambda prompts: [
        json.dumps({"label": "positive" if "love" in p else "negative"}) for p in prompts
    ]


s.register_engine("grader", grader)

print(
    s.sql(
        "SELECT label, COUNT(*) AS n FROM AI_EXTRACT(reviews, grader,"
        " prompt_column => 'body', schema => ['label string'])"
        " GROUP BY label ORDER BY label"
    ).to_pydict()
)
# {'label': ['negative', 'positive'], 'n': [2, 1]}
```

The generated column is an ordinary column, so the rest of the statement groups, filters and
joins over it without leaving SQL.

### Why these read as tables rather than as functions in the SELECT list

Every warehouse writes its AI call in the `SELECT` list and this does not, for the reason that
also makes `ML_PREDICT` a table function. A Batcher scalar function lowers to an expression
evaluated per row in Rust, and a language-model call is neither expressible there nor wanted
per row: the whole point of the inference path is that an engine loads once per worker and
sees a batch at a time. Writing the call in `FROM` says that rather than hiding it.

### What is not translated

`AI_CLASSIFY` is not. Its grammar is fixed at three arguments, and a relational form needs
four: the relation, the engine, the text column and the labels. Use `AI_EXTRACT` with a
one-field schema, or {py:meth}`ds.ml.classify <batcher.api.dataset.ml.DatasetML.classify>` on
the `Dataset`. `AI_EMBED`, `AI_SIMILARITY`, `AI_AGG` and `AI_FORECAST` are likewise
DataFrame-side; each reports where its capability lives rather than failing as an unknown
table.

The full set is always available on the `Dataset`, where these lower to anyway:
{py:meth}`ds.ml.generate <batcher.api.dataset.ml.DatasetML.generate>`,
{py:meth}`ds.ml.classify <batcher.api.dataset.ml.DatasetML.classify>`,
{py:meth}`ds.ml.extract <batcher.api.dataset.ml.DatasetML.extract>` and
{py:meth}`ds.ml.embed <batcher.api.dataset.ml.DatasetML.embed>`. See
{doc}`the LLM engines page </ml/retrieval/llm/engines>` for the engines they take, and
{doc}`batch inference </ml/inference/index>` for batching, GPU sizing and error handling.

## See also

- {doc}`SQL </user-guide/analyze/sql>`: the rest of the SQL surface, the supported subset, and the deliberate differences from DuckDB.
- {doc}`SQL API </api/relational/sql>`: the {py:class}`Session <batcher.Session>` methods, including naming a saved model by path.
- {doc}`Batch inference </ml/inference/index>`: the same models scored from the DataFrame API.

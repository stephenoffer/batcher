# User-defined functions

This page covers user-defined functions in Batcher: running your own Python over whole Arrow batches, per group, per row, or from SQL.

The first rule of a UDF in Batcher is not to write one. An expression such as `bt.col("x") * 2` or {py:meth}`.str.contains(...) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.contains>` lowers to Rust, runs vectorized over Arrow, and can be JIT-compiled. A Python UDF is none of those things. The optimizer also can't see through your function, so it won't push a filter past it or prune a column it might read. Reach for a UDF when the expression language genuinely has no answer, and when you do, hand it whole batches. A batch UDF still runs zero-copy over Arrow, in parallel across cores or a cluster, so the engine around it keeps its speed.

| Form | What the engine sees | Cost per row |
| --- | --- | --- |
| An {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` | a plan node it can push, prune, fuse, and JIT | vectorized in Rust |
| `map_batches(fn)` | an opaque stage over an Arrow batch | whatever `fn` does, once per batch |
| `map` / `flat_map` | an opaque stage over a Python dict per row | a Python object per row |

Before you write one, ask these questions in order. The first "yes" names the form, and each form has its own section below.

![Deciding whether a UDF is needed and which form to write, as four questions in order. First, if an expression such as bt.col, .str, .dt or .list can say it, write the expression, which is vectorized in Rust and needs no UDF. Second, if the work is per group, such as a session, a series or a document, use group_by(k).map_groups(fn), which hands every row of one group to each call. Third, if the function must see one row at a time, use map, flat_map or ml.filter, which build a Python object per row. Fourth, if there is costly setup such as a model, a client or a connection, pass a class to map_batches, which is built once per worker. Pass the class rather than an instance when construction must happen on the worker, such as for a CUDA context. Otherwise use map_batches with a plain function, called per batch. A batch function receives what batch_format asks for: a RecordBatch for "pyarrow", the default, a dict of ndarrays for "numpy", a DataFrame for "pandas", or a dict of tensors for "torch". The conversion happens around the call only, and the engine boundary stays Arrow.](/_static/diagrams/udf_choice.svg)

:::{tip}
`select` down to the columns the UDF reads *before* the UDF stage. The optimizer cannot prune across an opaque function, so anything still in the batch at that point is decoded, carried, and handed to Python whether `fn` looks at it or not.
:::

## Setup

```python
import batcher as bt
import pyarrow as pa
import pyarrow.compute as pc

ds = bt.from_pydict(
    {
        "text": ["a,b", "c", "d,e,f"],
        "price": [10.0, 20.0, 30.0],
        "qty": [1, 2, 3],
    }
)
```

## One function per Arrow batch

`fn` receives a `pyarrow.RecordBatch` and returns one. Everything inside should be vectorized Arrow compute. You are writing the *body* of a columnar operator, not a row loop.

```python
def add_total(batch):
    total = pc.multiply(batch.column("price"), pc.cast(batch.column("qty"), "float64"))
    return batch.append_column("total", total)


with_total = ds.map_batches(add_total, output_columns=["text", "price", "qty", "total"])
print(with_total.select("text", "total").to_pydict())
# {'text': ['a,b', 'c', 'd,e,f'], 'total': [10.0, 40.0, 90.0]}
```

`output_columns` declares the result columns when `fn` changes them. Omit it and later operations still believe the old schema, so a `select` on your new column fails at plan time. Declare it whenever the columns differ from the input. A list declares names only. A `pyarrow.Schema` declares the types too, which {ref}`Declare the output types <udf-declare-output-types>` covers.

`num_workers` defaults to `"auto"`, which fans the per-batch calls across local cores. That helps only if `fn` releases the GIL, which Arrow, NumPy, and torch all do. For a CPU-bound pure-Python `fn`, pass `multiprocessing=True`. Your script still needs an `if __name__ == "__main__":` guard, because a worker process starts by importing it.

A lambda or a closure is fine there as long as `cloudpickle` is installed, which is how the function reaches a worker that cannot import it by name. Without `cloudpickle` only a module-level function or a picklable callable object can cross, and anything else stays on threads with a warning saying so.

## Declare what you read

`input_columns` tells the optimizer which columns `fn` actually reads, so projection pushdown can prune the scan to those and skip decoding the rest. On a wide Parquet table that is the difference between reading a handful of columns and reading all of them.

```python
priced = ds.map_batches(
    lambda b: b.append_column("cheap", pc.less(b.column("price"), 25.0)),
    input_columns=["price"],
    output_columns=["text", "price", "qty", "cheap"],
)
print(priced.select("price", "cheap").to_pydict())
# {'price': [10.0, 20.0, 30.0], 'cheap': [True, True, False]}
```

:::{warning}
`input_columns` is a *declaration to the optimizer*, not a filter on the batch `fn` receives. Naming a column does not hide the others, and leaving one out does not merely cost you nothing. The column you failed to declare can be pruned out of the scan from under the function, and `fn` then reads a column that is not there. That is a correctness bug, not a slow query. Leave `input_columns=None`, the default, if you are not sure. That keeps every column alive.
:::

## A class loads once per worker

:::{tip}
A plain function is re-created on every batch. A class is instantiated *once per worker* and then called per batch, which is the difference between loading a model once per batch and loading it once per worker. Nothing else in this API buys as much for one word of typing.
:::

```python
class Splitter:
    def __init__(self, sep):
        self.sep = sep

    def __call__(self, batch):
        counts = [len(v.split(self.sep)) for v in batch.column("text").to_pylist()]
        return batch.append_column("parts", pa.array(counts, pa.int64()))


print(
    ds.map_batches(Splitter(","), output_columns=["text", "price", "qty", "parts"])
    .select("text", "parts")
    .to_pydict()
)
# {'text': ['a,b', 'c', 'd,e,f'], 'parts': [2, 1, 3]}
```

Pass the class itself, as in `map_batches(Classifier, num_gpus=1)`, when construction needs to happen inside the worker. That is the case for anything holding a CUDA context. The engine warns you if a GPU stage gets a bare function, because that is the single most expensive mistake in this API. See {doc}`inference </ml/inference/inference>`.

A model class almost never takes zero arguments, so `fn_constructor_args` and `fn_constructor_kwargs` supply them. The class is still built once per worker, so this is not the same as passing an instance:

```python
print(
    ds.map_batches(
        Splitter,
        fn_constructor_args=(",",),
        output_columns=["text", "price", "qty", "parts"],
    )
    .select("parts")
    .to_pydict()
)
# {'parts': [2, 1, 3]}
```

Use `fn_args` and `fn_kwargs` for arguments that vary per call rather than per worker. They arrive after the batch, as `fn(batch, *fn_args, **fn_kwargs)`. `fn_args` and `fn_constructor_args` must be a tuple or list and `fn_kwargs` and `fn_constructor_kwargs` a dict with string keys. Any other shape raises a `PlanError` naming the parameter when you define the stage, not in a worker on its first batch. The same holds for `input_columns`, `preserves_columns`, `num_workers` below 1, and a non-numeric `max_concurrency`.

If the class holds a resource that must be released, give it a `close` method. Batcher calls it when the worker is done with the model, which is where a GPU allocation or an HTTP session goes back. Under `collect(distributed=True)` the model lives on a long-lived actor, and Batcher asks each actor to close its models before it shuts the actor pool down. That shutdown waits a bounded time for `close` to return and then ends the actor regardless, so a `close` that hangs can't wedge the job.

## Receive NumPy, pandas, or torch batches

`batch_format` converts around the call only. The engine boundary stays Arrow.

```python
def scale(batch):  # batch is {column: ndarray}
    return {"price": batch["price"] * 2.0, "qty": batch["qty"]}


print(
    ds.select("price", "qty")
    .map_batches(scale, batch_format="numpy", output_columns=["price", "qty"])
    .to_pydict()
)
# {'price': [20.0, 40.0, 60.0], 'qty': [1, 2, 3]}
```

A NumPy, pandas, or torch batch is a zero-copy view of Arrow memory, so it is read-only. Writing into it in place raises NumPy's `assignment destination is read-only`, and Batcher adds a note to that error naming the fix. Pass `zero_copy_batch=False` to hand `fn` a writable copy, paid only by the stages that ask for it:

```python
def cap(batch):
    batch["price"][batch["price"] > 15] = 15.0  # writes into the batch
    return batch


print(ds.select("price").map_batches(cap, batch_format="numpy", zero_copy_batch=False).to_pydict())
# {'price': [10.0, 15.0, 15.0]}
```

## Per-row functions, when you must

`map` takes `fn(row_dict) -> row_dict` and `flat_map` returns any number of rows per input row. The rows are built inside the worker, never in the driver, so the hot-path rule holds. But you are paying Python-object cost per row, and it shows.

Declare `input_columns` here if you declare it anywhere. A batch callback pays for an undeclared column once, when it is decoded. A row callback pays twice, because every one of those columns is also boxed into a Python object for every row.

The output columns are every key any row returned, in the order they first appear. A row without a key is null in that column, the same rule `map_batches` applies across batches. A key that an earlier row had and a later one dropped is logged as a warning, because that is usually a rename or a typo rather than an optional field:

```python
print(ds.map(lambda row: {"short": row["text"]} if len(row["text"]) < 3 else {"long": row["text"]}).to_pydict())
# {'long': ['a,b', None, 'd,e,f'], 'short': [None, 'c', None]}
```

A column your callback adds is unknown to the plan until you declare it with `output_columns`, as with `map_batches`. Referencing it downstream without the declaration raises a `ColumnNotFoundError` whose message says to pass `output_columns=[...]`.

::::{tab-set}
:::{tab-item} flat_map

```python
print(
    ds.select("text").flat_map(lambda row: [{"tok": t} for t in row["text"].split(",")]).to_pydict()
)
# {'tok': ['a', 'b', 'c', 'd', 'e', 'f']}
```

:::

:::{tab-item} The expression form

```python
print(ds.select(tok=bt.col("text").str.split(",")).explode("tok").to_pydict())
# {'tok': ['a', 'b', 'c', 'd', 'e', 'f']}
```

:::
::::

### A Python predicate

{py:meth}`ds.filter <batcher.Dataset.filter>` keeps the rows for which `fn(row_dict)` is true. Reach for it when the condition genuinely cannot be written as an expression, such as a call into a library or a model's verdict. {py:meth}`ds.filter <batcher.Dataset.filter>` stays vectorized in Rust and is the right answer everywhere else.

```python
import pyarrow.compute as pc

print(ds.filter(lambda batch: pc.greater(pc.count_substring(batch["text"], ","), 0)).to_pydict())
# {'text': ['a,b', 'd,e,f'], 'price': [10.0, 30.0], 'qty': [1, 3]}
```

The predicate's answers become one Arrow boolean mask, so the surviving rows keep their exact types and no value makes the round trip back through Python. Dropping rows also changes no column, which the stage declares, so a cheap expression filter written after it is still pushed underneath and runs first.

Declare `input_columns` here. Building a Python dict per row is the entire cost of a row predicate, and declaring what it reads narrows that dict to those columns. Every column still comes out, because the output is the input masked. Reading a column you did not declare raises rather than working by accident.

```python
print(ds.filter(lambda batch: batch["text"].str.contains(","), batch_format="pandas", input_columns=["text"]).to_pydict())
# {'text': ['a,b', 'd,e,f'], 'price': [10.0, 30.0], 'qty': [1, 3]}
```

Same rows, and the expression form builds no Python object per row. Check for an expression before you write the loop.

## Async functions

An `async def` function suits an I/O-bound callback, such as a request to a model endpoint. Batcher awaits its calls concurrently on one event loop instead of holding a thread per request. Four verbs accept one, and `max_concurrency` bounds a different unit on each:

| Verb | What `max_concurrency` bounds |
| --- | --- |
| `map_batches(fn)` | Batches in flight at once. |
| `filter(fn)` | Batches in flight at once. |
| `map(fn)` | Rows in flight at once within one batch. |
| `flat_map(fn)` | Rows in flight at once within one batch. |

`0`, the default, picks a default bound. The bound applies within one worker, so a cluster runs up to `max_concurrency` times the worker count. A quota that spans every worker, such as a provider's requests per minute, belongs on the engine making the call rather than on the stage.

```python
import asyncio


async def lookup(batch):
    await asyncio.sleep(0)  # stands in for a request to a remote service
    return batch.append_column("found", pc.greater(batch.column("qty"), 1))


print(ds.select("qty").map_batches(lookup, max_concurrency=4, output_columns=["qty", "found"]).to_pydict())
# {'qty': [1, 2, 3], 'found': [False, True, True]}
```

## Bundle a function with its options

`@bt.udf` bundles a function with its `map_batches` options so the transform is a reusable, named thing you apply to a dataset. Options go on the decorator, so the call site stays clean.

```python
@bt.udf(output_columns=["text", "price", "qty", "discounted"])
def discount(batch):
    return batch.append_column("discounted", pc.multiply(batch.column("price"), 0.9))


print(discount(ds).select("price", "discounted").to_pydict())
# {'price': [10.0, 20.0, 30.0], 'discounted': [9.0, 18.0, 27.0]}
```

`@bt.udf(per_row=True)` wraps a `fn(row) -> row` callback the same way.

A decorated function stays an ordinary Python function. Call it on a batch to test it without building a dataset, pass it to `map_batches` by hand, or reuse it inside another UDF:

```python
import pyarrow as pa

print(discount(pa.record_batch({"price": [10.0]})).column("discounted").to_pylist())
# [9.0]
```

Use `options` to run the same function at a second scale rather than defining it twice. The original is unchanged, so a local smoke test and a cluster run share one definition:

```python
big = discount.options(batch_size=4096)
print(big(ds).select("discounted").to_pydict())
# {'discounted': [9.0, 18.0, 27.0]}
```

## What your function may return

The default `batch_format="pyarrow"` hands your function a `RecordBatch`. It may return any of the following, so a model wrapper does not have to convert its framework's output before Batcher sees it:

| Return value | Use it when |
|---|---|
| `pyarrow.RecordBatch` or `Table` | The function already works in Arrow. |
| `{"col": values}` dict | You are building columns from scratch, including NumPy arrays. |
| `pandas.DataFrame` or `polars.DataFrame` | The transform is easier to write in a frame library. |
| A list or generator of any of the above | One input batch expands into several output batches. |

The generator form is what a row-expanding stage wants, such as decoding a video into frames or fanning one prompt out into several completions. Yield a batch per unit of work instead of concatenating everything first:

```python
def explode_chars(batch):
    for text in batch.column("text").to_pylist():
        yield {"ch": list(text)}


print(bt.from_pydict({"text": ["ab", "cd"]}).map_batches(explode_chars).to_pydict())
# {'ch': ['a', 'b', 'c', 'd']}
```

Returning a list of *row* dicts is rejected, because that is {py:meth}`ds.flat_map <batcher.Dataset.flat_map>`, which declares the row-at-a-time cost rather than hiding it.

## Keep the output schema stable across batches

Your function is called once per batch, and Batcher reconciles the batches it returns into one result. A column a later batch adds is filled with nulls for the earlier rows. That is deliberate. It lets a stage whose output grows a field, such as an LLM returning structured output, concatenate instead of failing at the merge:

```python
def gains_a_field(batch):
    yield {"a": [1]}
    yield {"a": [2], "extra": [9]}


print(bt.from_pydict({"x": [1]}).map_batches(gains_a_field).to_pydict())
# {'a': [1, 2], 'extra': [None, 9]}
```

The reverse is almost always a bug. A column missing from a later batch is kept and null-filled, so a function that renames or drops one returns a result that is mostly null:

```python
def renames_halfway(batch):
    yield {"a": [1, 2]}
    yield {"b": [3, 4]}


print(bt.from_pydict({"x": [1]}).map_batches(renames_halfway).to_pydict())
# {'a': [1, 2, None, None], 'b': [None, None, 3, 4]}
```

Batcher logs a warning naming the dropped columns, because the result otherwise looks complete. It is a warning rather than an error because a column appearing and a column disappearing are the same operation at the schema level, and the first is supported.

Build the output columns once, outside any per-batch branching, so every return path has the same keys. A function that decides its columns inside an `if` is the shape this catches.

(udf-declare-output-types)=
### Declare the output types

A list in `output_columns` names the columns and leaves their types to whatever `fn` returns. Pass a `pyarrow.Schema` instead and the types are part of the declaration. `map`, `flat_map`, and `map_batches` all accept one. Every output batch is cast to it, and a value that does not fit raises a `SchemaError` naming the column. A field declared `nullable=False` refuses a null the same way. A tensor column is declared with Arrow's own `pa.fixed_shape_tensor(...)` type.

The declaration does two things a list can't. `schema` answers from it without calling `fn`. And an empty input still returns the declared types, where a list leaves a column no row reached as Arrow `null`:

```python
calls = []


def priced(batch):
    calls.append(batch.num_rows)
    return {"cents": pc.cast(pc.multiply(batch.column("price"), 100), "int64")}


cents = pa.schema([("cents", pa.int64())])
print(ds.map_batches(priced, output_columns=cents).schema, calls)
# cents: int64 []
print(ds.filter(bt.col("price") > 1000).map_batches(priced, output_columns=cents).collect().schema)
# cents: int64
```

The types come back the way the engine normalizes them for every operator above the stage. A declared `int32` reads back as `int64`, a `float32` as `float64`, and a dictionary column as its value type.

Without a declared schema, `schema` has to ask `fn`. It calls a batch function on an empty batch. A `map` or `flat_map` function is never called on an empty batch, so it gets one input row, and so does any batch stage beneath it. Either way a class `fn` is still constructed, which for a model means loading it. A column that comes back Arrow `null` from the empty batch is asked about on one row as well. The probe sees at most one row, so a type that depends on the values across rows, such as a tensor column that is ragged only because its shapes differ, is certain only when you declare it. The callable form of `filter` is the exception, because dropping rows keeps the input's columns and types, so `schema` reads them from the input without calling the predicate.

## One call per group

{py:meth}`map_groups <batcher.GroupBy.map_groups>` hands your function every row of one group and no row of another. It is the Batcher spelling of pandas `groupby().apply()`, Polars {py:meth}`group_by().map_groups() <batcher.Dataset.group_by>`, and Spark `applyInPandas`, and it is what per-entity work needs: a user's session sequence, a device's time series, a document's chunks.

```python
sales = bt.from_pydict(
    {
        "region": ["west", "east", "west", "east"],
        "amount": [10.0, 5.0, 7.0, 3.0],
    }
)


def spread(group):  # group is a RecordBatch of one region's rows
    amounts = group.column("amount").to_pylist()
    return {
        "region": [group.column("region")[0].as_py()],
        "spread": [max(amounts) - min(amounts)],
    }


print(
    sales.group_by("region")
    .map_groups(spread, output_columns=["region", "spread"])
    .sort("region")
    .to_pydict()
)
# {'region': ['east', 'west'], 'spread': [2.0, 3.0]}
```

Pass `batch_format="pandas"` to receive each group as a `DataFrame`, which is the `applyInPandas` shape. The conversion happens per group, so the frame holds the group's rows.

:::{warning}
Do not call `map_batches` straight after `group_by`. It sees whatever batches the engine produces, and a group is not confined to one of them, so your function runs on *fragments* of a group and returns a wrong answer rather than an error.
:::

Two cases do not need a callback at all. A plain reduction is `.agg(...)`, which runs in Rust. Broadcasting a group statistic back onto every row is a window:

```python
print(
    sales.window(partition_by=["region"], functions={"total": ("sum", "amount")})
    .sort("region", "amount")
    .to_pydict()["total"]
)
# [8.0, 8.0, 17.0, 17.0]
```

Prefer both of those when they fit. `map_groups` materializes one group at a time, so a single key holding hundreds of millions of rows needs a reduction rather than a callback. Row order within a group is not guaranteed either, so sort inside the function when it matters.

Two keywords make the callback safe on real data. `output_schema` declares the result's schema. An input with no groups never calls the function, so without a schema there is nothing to learn the result's columns from, and the empty result has no columns at all. With one, it is an empty table of that schema, and every result is cast to it.

```python
import pyarrow as pa

schema = pa.schema([("region", pa.string()), ("spread", pa.float64())])
none = sales.filter(bt.col("amount") > 100).group_by("region")
print(none.map_groups(spread, output_schema=schema).collect().schema)
# region: string
# spread: double
```

`max_group_rows` and `max_group_bytes` refuse a group over either limit before the function sees it, with an `ExecutionError` naming the group's key and size. The engine has already assembled the group by then, so the limits protect the Python side, the conversion to `batch_format` and whatever your function builds, rather than engine memory.

:::{note}
`map_groups` builds an aggregation followed by a `map_batches`, so whether {py:meth}`collect(distributed=True) <batcher.Dataset.collect>` accepts the plan is the same question as for {py:meth}`ds.group_by("k").agg(...).map_batches(fn) <batcher.Dataset.group_by>`. Check it on your plan before relying on it.
:::

## UDFs in SQL

{py:func}`bt.register_function(name, fn, result_type=...) <batcher.register_function>` makes a Python function callable from {py:func}`bt.sql <batcher.sql>`. The vectorized form, which is the default, receives whole Arrow arrays.

```python
bt.register_function("bump", lambda a: pc.add(a, 100), result_type="int64")
print(bt.sql("SELECT bump(qty) AS q FROM t", t=ds).to_pydict())
# {'q': [101, 102, 103]}
```

Scalar SQL functions do not work inside `GROUP BY` keys, aggregate arguments, or `ORDER BY`. Compute them in a subquery or a projected alias first. For a function that transforms a whole table, register it with `table=True` and it follows the `map_batches` contract, forwarding any `map_batches` option you pass alongside it.

There is no aggregate form. An aggregate has to be mergeable, built from a partial, a combine and a finalize, so that one machine and a hundred produce the same answer. A Python callable over one batch cannot supply that. Use `ds.group_by(...).agg(...)` for a built-in aggregate, or `map_groups` for arbitrary Python over each group.

An option the call form cannot honor is rejected at registration rather than ignored, so a misspelled keyword fails where you wrote it.

A per-row function receives `None` for a NULL argument by default, and must handle it. DuckDB's default is the opposite, NULL in and NULL out. Pass `null_handling="default"` to get DuckDB's behavior: a row with any NULL argument answers NULL and `fn` never sees it, in the vectorized form too. The parameter takes DuckDB's name and values, and the default `"special"` keeps NULLs flowing to `fn`.

```python
s = bt.Session()
s.register("n", bt.from_pydict({"x": [1, None, 3]}))
s.register_function("inc", lambda v: v + 1, vectorized=False, result_type="int64", null_handling="default")
print(s.sql("SELECT inc(x) AS y FROM n").to_pydict())
# {'y': [2, None, 4]}
```

A registered function exists only inside a SQL query, run through {py:obj}`bt.sql <batcher.sql>`, `Session.sql`, or {py:obj}`Dataset.sql <batcher.Dataset.sql>`. {py:func}`bt.call_function <batcher.call_function>` names the built-in function library and raises for it, because an expression has no relation for the function's `map_batches` stage to run over. In the DataFrame API, apply the same callable with `map_batches`.

## Taking it to a cluster

Distributing a UDF stage, surviving a batch that raises, and the idempotency a preempted worker demands are all on {doc}`Running a UDF at scale <udfs-at-scale>`.

## See also

- {doc}`Expressions </user-guide/transform/columns/expressions>`: check here first, because the expression usually exists.
- {doc}`Running a UDF at scale <udfs-at-scale>`: distributing a UDF stage, the `max_errored_rows` budget, and idempotency under retry.
- {doc}`Inference </ml/inference/inference>`: the class-per-worker pattern with a real model.
- {doc}`Explain plans </user-guide/operate/tuning/explain-plans>`: see what a UDF does to the plan the optimizer builds.
- {doc}`Expression evaluation </architecture/deep-dives/query/expression-evaluation>`: what an expression gets that a UDF cannot, meaning vectorization, fusion, and the JIT.
- {doc}`Arrow memory </architecture/deep-dives/memory/arrow-memory>`: why `fn` is handed a zero-copy `RecordBatch` and what happens when you convert it.
- {doc}`Expressions API </api/relational/expressions>`: the method surface to check before you write a function.
- {doc}`Feature pipeline </cookbook/ml/pipelines/features/feature-pipeline>`: batch functions and expressions side by side in one job.
- {doc}`/cookbook/ml/inference/batch_inference`: a model over every row without a Python loop, as a script.

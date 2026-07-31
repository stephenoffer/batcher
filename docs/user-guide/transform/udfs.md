# User-defined functions

The first rule of a UDF in Batcher is not to write one. An expression such as
`bt.col("x") * 2` or `.str.contains(...)` lowers to Rust, runs vectorized over Arrow, and
can be JIT-compiled. A Python UDF is none of those things. The optimizer also can't see
through your function, so it won't push a filter past it or prune a column it might read.
Reach for a UDF when the expression language genuinely has no answer, and when you do,
hand it whole batches.

| Form | What the engine sees | Cost per row |
| --- | --- | --- |
| An `Expr` | a plan node it can push, prune, fuse, and JIT | vectorized in Rust |
| `map_batches(fn)` | an opaque stage over an Arrow batch | whatever `fn` does, once per batch |
| `map` / `flat_map` | an opaque stage over a Python dict per row | a Python object per row |

:::{tip}
`select` down to the columns the UDF reads *before* the UDF stage. The optimizer cannot
prune across an opaque function, so anything still in the batch at that point is decoded,
carried, and handed to Python whether `fn` looks at it or not.
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

## map_batches: one function, one Arrow batch

`fn` receives a `pyarrow.RecordBatch` and returns one. Everything inside should be
vectorized Arrow compute. You are writing the *body* of a columnar operator, not a row
loop.

```python
def add_total(batch):
    total = pc.multiply(batch.column("price"), pc.cast(batch.column("qty"), "float64"))
    return batch.append_column("total", total)


with_total = ds.map_batches(add_total, output_columns=["text", "price", "qty", "total"])
print(with_total.select("text", "total").to_pydict())
# {'text': ['a,b', 'c', 'd,e,f'], 'total': [10.0, 40.0, 90.0]}
```

`output_columns` declares the result schema when `fn` changes it. Omit it and later
operations still believe the old schema, so a `select` on your new column fails at plan
time. Declare it whenever the columns differ from the input.

`num_workers` defaults to `"auto"`, which fans the per-batch calls across local cores.
That helps only if `fn` releases the GIL, which Arrow, NumPy, and torch all do. For a CPU-bound
pure-Python `fn`, pass `multiprocessing=True`. Then `fn` must be importable and your
script needs an `if __name__ == "__main__":` guard, because the workers re-import it.

## Declare what you read with input_columns

`input_columns` tells the optimizer which columns `fn` actually reads, so projection
pushdown can prune the scan to those and skip decoding the rest. On a wide Parquet table
that is the difference between reading a handful of columns and reading all of them.

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
`input_columns` is a *declaration to the optimizer*, not a filter on the batch `fn`
receives. Naming a column does not hide the others, and leaving one out does not merely
cost you nothing: the column you failed to declare can be pruned out of the scan from
under the function, and `fn` then reads a column that is not there. That is a correctness
bug, not a slow query. Leave `input_columns=None` (the default) if you are not sure,
which keeps every column alive.
:::

## A class loads once per worker

:::{tip}
A plain function is re-created on every batch. A class is instantiated **once per
worker** and then called per batch, which is the difference between loading a model once
per batch and loading it once per worker. This is the single highest-leverage line in the
API.
:::

```python
class Splitter:
    def __init__(self, sep):
        self.sep = sep

    def __call__(self, batch):
        counts = [len(v.split(self.sep)) for v in batch.column("text").to_pylist()]
        return batch.append_column("parts", pa.array(counts, pa.int64()))


print(ds.map_batches(Splitter(","), output_columns=["text", "price", "qty", "parts"])
      .select("text", "parts").to_pydict())
# {'text': ['a,b', 'c', 'd,e,f'], 'parts': [2, 1, 3]}
```

Pass the class itself, as in `map_batches(Classifier, num_gpus=1)`, when construction
needs to happen inside the worker. That is the case for anything holding a CUDA context. The
engine warns you if a GPU stage gets a bare function, because that is the single most
expensive mistake in this API. See {doc}`inference </ml/inference/inference>`.

A model class almost never takes zero arguments, so `fn_constructor_args` and
`fn_constructor_kwargs` supply them. The class is still built once per worker, so this is
not the same as passing an instance:

```python
print(ds.map_batches(
    Splitter,
    fn_constructor_args=(",",),
    output_columns=["text", "price", "qty", "parts"],
).select("parts").to_pydict())
# {'parts': [2, 1, 3]}
```

Use `fn_args` and `fn_kwargs` for arguments that vary per call rather than per worker.
They arrive after the batch, as `fn(batch, *fn_args, **fn_kwargs)`.

If the class holds a resource that must be released, give it a `close` method. Batcher
calls it when the worker is done with the model, which is where a GPU allocation or an
HTTP session goes back.

## batch_format: numpy, pandas, torch

`batch_format` converts around the call only. The engine boundary stays Arrow.

```python
def scale(batch):  # batch is {column: ndarray}
    return {"price": batch["price"] * 2.0, "qty": batch["qty"]}


print(ds.select("price", "qty")
      .map_batches(scale, batch_format="numpy", output_columns=["price", "qty"])
      .to_pydict())
# {'price': [20.0, 40.0, 60.0], 'qty': [1, 2, 3]}
```

## Per-row functions, when you must

`map` takes `fn(row_dict) -> row_dict` and `flat_map` returns any number of rows per
input row. The rows are built inside the worker, never in the driver, so the hot-path
rule holds. But you are paying Python-object cost per row, and it shows.

::::{tab-set}
:::{tab-item} flat_map

```python
print(ds.select("text").flat_map(lambda row: [{"tok": t} for t in row["text"].split(",")])
      .to_pydict())
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

Same rows, and the expression form builds no Python object per row. Check for an
expression before you write the loop.

## @udf: a function bundled with its config

`@bt.udf` bundles a function with its `map_batches` options so the transform is a
reusable, named thing you apply to a dataset. Options go on the decorator, so the call
site stays clean.

```python
@bt.udf(output_columns=["text", "price", "qty", "discounted"])
def discount(batch):
    return batch.append_column("discounted", pc.multiply(batch.column("price"), 0.9))


print(discount(ds).select("price", "discounted").to_pydict())
# {'price': [10.0, 20.0, 30.0], 'discounted': [9.0, 18.0, 27.0]}
```

`@bt.udf(per_row=True)` wraps a `fn(row) -> row` callback the same way.

A decorated function stays an ordinary Python function. Call it on a batch to test it
without building a dataset, pass it to `map_batches` by hand, or reuse it inside another
UDF:

```python
import pyarrow as pa

print(discount(pa.record_batch({"price": [10.0]})).column("discounted").to_pylist())
# [9.0]
```

Use `options` to run the same function at a second scale rather than defining it twice.
The original is unchanged, so a local smoke test and a cluster run share one definition:

```python
big = discount.options(batch_size=4096)
print(big(ds).select("discounted").to_pydict())
# {'discounted': [9.0, 18.0, 27.0]}
```

## What your function may return

The default `batch_format="pyarrow"` hands your function a `RecordBatch`. It may return
any of the following, so a model wrapper does not have to convert its framework's output
before Batcher sees it:

| Return value | Use it when |
|---|---|
| `pyarrow.RecordBatch` or `Table` | The function already works in Arrow. |
| `{"col": values}` dict | You are building columns from scratch, including NumPy arrays. |
| `pandas.DataFrame` or `polars.DataFrame` | The transform is easier to write in a frame library. |
| A list or generator of any of the above | One input batch expands into several output batches. |

The generator form is what a row-expanding stage wants, such as decoding a video into
frames or fanning one prompt out into several completions. Yield a batch per unit of work
instead of concatenating everything first:

```python
def explode_chars(batch):
    for text in batch.column("text").to_pylist():
        yield {"ch": list(text)}


print(bt.from_pydict({"text": ["ab", "cd"]}).map_batches(explode_chars).to_pydict())
# {'ch': ['a', 'b', 'c', 'd']}
```

Returning a list of *row* dicts is rejected, because that is `ds.ml.flat_map`, which
declares the row-at-a-time cost rather than hiding it.

## map_groups: one call per group

`map_groups` hands your function every row of one group and no row of another. It is the
Batcher spelling of pandas `groupby().apply()`, Polars `group_by().map_groups()`, and Spark
`applyInPandas`, and it is what per-entity work needs: a user's session sequence, a device's
time series, a document's chunks.

```python
sales = bt.from_pydict({
    "region": ["west", "east", "west", "east"],
    "amount": [10.0, 5.0, 7.0, 3.0],
})


def spread(group):  # group is a RecordBatch of one region's rows
    amounts = group.column("amount").to_pylist()
    return {
        "region": [group.column("region")[0].as_py()],
        "spread": [max(amounts) - min(amounts)],
    }


print(sales.group_by("region")
      .map_groups(spread, output_columns=["region", "spread"])
      .sort("region").to_pydict())
# {'region': ['east', 'west'], 'spread': [2.0, 3.0]}
```

Pass `batch_format="pandas"` to receive each group as a `DataFrame`, which is the
`applyInPandas` shape. The conversion happens per group, so the frame holds the group's rows.

:::{warning}
Do not call `map_batches` straight after `group_by`. It sees whatever batches the engine
produces, and a group is not confined to one of them, so your function runs on *fragments*
and returns a wrong answer rather than an error. Measured over 50,000 rows and 20 keys,
every one of the 20 keys spanned more than one batch.
:::

Two cases do not need a callback at all. A plain reduction is `.agg(...)`, which runs in
Rust. Broadcasting a group statistic back onto every row is a window:

```python
print(sales.window(partition_by=["region"], functions={"total": ("sum", "amount")})
      .sort("region", "amount").to_pydict()["total"])
# [8.0, 8.0, 17.0, 17.0]
```

Prefer both of those when they fit. `map_groups` materializes one group at a time, so a
single key holding hundreds of millions of rows needs a reduction rather than a callback.
Row order within a group is not guaranteed either, so sort inside the function when it
matters.

:::{note}
`map_groups` builds an aggregation followed by a `map_batches`, so whether
`collect(distributed=True)` accepts the plan is the same question as for
`ds.group_by("k").agg(...).map_batches(fn)`. Check it on your plan before relying on it.
:::

## Tolerating dirty data

A single malformed record should not kill a six-hour job. With `max_errored_rows` set,
a batch whose `fn` raises is bisected to isolate the offending rows, and those rows are
*dropped* up to the budget. Past the budget the error propagates, so a genuine bug on
clean data still fails fast.

```python
raw = bt.from_pydict({"s": ["1", "2", "oops", "4"]})


def parse(batch):
    return pa.RecordBatch.from_pydict(
        {"n": [int(v) for v in batch.column("s").to_pylist()]}
    )


print(raw.map_batches(parse, output_columns=["n"], max_errored_rows=10).to_pydict())
# {'n': [1, 2, 4]}
```

:::{important}
Default is 0 (strict). Set it deliberately and keep it small. A budget of 1,000,000
silently deleted rows is not resilience, it is a deletion policy nobody agreed to.
:::

## UDFs in SQL

`bt.register_function(name, fn, result_type=...)` makes a Python function callable from
`bt.sql`. The vectorized form (the default) receives whole Arrow arrays.

```python
bt.register_function("bump", lambda a: pc.add(a, 100), result_type="int64")
print(bt.sql("SELECT bump(qty) AS q FROM t", t=ds).to_pydict())
# {'q': [101, 102, 103]}
```

Scalar SQL functions do not work inside `GROUP BY` keys, aggregate arguments, or
`ORDER BY`. Compute them in a subquery or a projected alias first. For a function that
transforms a whole table, register it with `table=True` and it follows the `map_batches`
contract.

## The distributed caveat

:::{warning}
Under `distributed=True`, a worker that gets preempted mid-batch is reassigned and its
partition **recomputed**. So `fn` must be idempotent. A pure transform is safe. A `fn`
that POSTs to an API, upserts into a vector DB, or increments an external counter can
apply that effect twice. Make the sink idempotent by upserting on a key, or move the side
effect out of the UDF and into a `write`.
:::

## See also

- {doc}`Expressions </user-guide/transform/expressions>`: check here first, because the expression usually exists.
- {doc}`Inference </ml/inference/inference>`: the class-per-worker pattern with a real model.
- {doc}`Explain plans </user-guide/operate/explain-plans>`: see what a UDF does to the plan the optimizer builds.
- {doc}`Expression evaluation </deep-dives/query/expression-evaluation>`: what an expression
  gets that a UDF cannot, meaning vectorization, fusion, and the JIT.
- {doc}`Arrow memory </deep-dives/memory/arrow-memory>`: why `fn` is handed a zero-copy
  `RecordBatch` and what happens when you convert it.
- {doc}`Expressions API </api/relational/expressions>`: the method surface to check before you write
  a function.
- {doc}`Feature pipeline </cookbook/ml/pipelines/feature-pipeline>`: batch functions and expressions
  side by side in one job.
- {doc}`/cookbook/ml/inference/batch_inference`: a model over every row without a Python loop, as a script.

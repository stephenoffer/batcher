# Running a UDF at scale

This page covers what changes when a `map_batches` stage runs over a cluster rather than one machine: how a UDF with a pipeline breaker above it is staged, what happens to a device request or a function the cluster cannot take, how to survive a malformed record without losing the job, how many times your function runs, and the idempotency a distributed retry demands. The callback contract itself is on {doc}`User-defined functions <udfs>`.

The examples run against these imports:

```python
import batcher as bt
import pyarrow as pa
```

## Running a UDF pipeline on a cluster

A `map_batches` chain distributes on its own. Each worker reads its own splits and runs the function over them, and nothing passes through the driver.

A UDF with a *breaker* above it needs one more step, because the breaker cannot see through an opaque Python function to co-partition anything. Batcher runs the UDF as its own distributed stage, lands its output on cluster-shared scratch, and then dispatches the breaker over that, so the operator above gets the shuffle, the skew handling and the spill it always had. Sorts, `distinct`, windows and limits go through this, and so do joins and unions, where each operand that holds a UDF is staged separately:

```python
# docs: skip
embedded = docs.map_batches(Embedder, num_gpus=1)
enriched = embedded.join(metadata, on="doc_id").group_by("topic").agg(n=bt.col("doc_id").count())
enriched.collect(distributed=True)
```

The staging needs a scratch directory every node can reach. On a cluster with no shared mount, point `memory.spill_dir` at a shared filesystem. Without one, Batcher raises rather than writing files a worker cannot open.

## Requesting a GPU the cluster does not have

`num_gpus` and `resources` are Ray resource requests on a cluster. Before a stage is submitted, Batcher checks them against the live cluster, and a request no alive node can meet raises a `PlanError` naming the argument, the amount, and what the cluster offers, instead of leaving a worker waiting forever to be placed. The check applies to a cluster that cannot grow, meaning one this process started with `ray.init()` or one with no autoscaling signal. On an autoscaling cluster the request is passed to the autoscaler, which is what brings a GPU node up.

A single-node run has no scheduler to wait on. There a `num_gpus` stage runs its function in this process, and when this process demonstrably has no accelerator Batcher says so with a `PerformanceWarning`, because the model then runs on CPU. The two paths differ on purpose: locally the request is a hint, and on a cluster it is binding.

## Shipping the function to the workers

On a cluster the function is pickled and sent to every worker. A lambda or a closure pickles by value, but an object it captures has to pickle too. A closure over a lock, a socket, or an open client raises a `PlanError` naming the captured variable. Create that object inside the function, or in a class's `__init__` so each worker builds its own. A function or class defined in a module the workers cannot import, such as a test file, has to be defined inside a function instead, because Ray pickles an importable name by reference.

## Tolerating dirty data

A single malformed record should not kill a six-hour job. With `max_errored_rows` set, a batch whose `fn` raises is bisected to isolate the offending rows, and those rows are *dropped* up to the budget. Past the budget the error propagates, so a genuine bug on clean data still fails fast.

```python
raw = bt.from_pydict({"s": ["1", "2", "oops", "4"]})


def parse(batch):
    return pa.RecordBatch.from_pydict({"n": [int(v) for v in batch.column("s").to_pylist()]})


print(raw.map_batches(parse, output_columns=["n"], max_errored_rows=10).to_pydict())
# {'n': [1, 2, 4]}
```

:::{important}
The default is 0, which is strict. Set it deliberately and keep it small. A budget of 1,000,000 silently deleted rows is not resilience. It is a deletion policy nobody agreed to.
:::

The budget is one allowance per worker process, whichever way the stage runs: threads, worker processes, or a streamed window all draw down the same count. Across a cluster the honest bound is therefore `workers x max_errored_rows`. The allowance belongs to the function rather than to one query, so a second run of the same function in the same process draws on what the first run left. Every drop is published to the observability bus with the running total and the error text, so a long job reports the loss while it happens rather than at the end.

The row callbacks take it too. `ds.map`, `ds.flat_map`, and a callable `ds.filter` all lower to a `map_batches` stage, so the same budget isolates a raising callback down to the rows that raised:

```python
def parse_row(row):
    return {"n": int(row["s"])}


rows = bt.from_pydict({"s": ["1", "2", "oops", "4"]})
print(rows.map(parse_row, output_columns=["n"], max_errored_rows=10).to_pydict())
# {'n': [1, 2, 4]}
```

### Keeping the failed rows

A dropped row is gone, and at scale you can't tell afterwards which rows they were. Pass `error_column` to keep them instead. Each row the budget would drop comes out with its output columns null and the error, as `"<ExcType>: <message>"`, in the column you named. Every other row has null there. A kept row still counts against `max_errored_rows`, so the budget still stops a job whose function is simply broken.

The failed row is built without your function's output, so its types have to be known up front. Declare `output_columns` as a `pyarrow.Schema`. A callable `filter` needs no declaration, because its output columns are its input's, and it keeps the failed row with its input values. A column named in `preserves_columns` on `map_batches` keeps its input value too.

```python
def parse_kept(row):
    return {"n": int(row["s"])}


kept = rows.map(
    parse_kept,
    output_columns=pa.schema([("n", pa.int64())]),
    max_errored_rows=10,
    error_column="error",
)
print(kept.to_pydict())
# {'n': [1, 2, None, 4], 'error': [None, None, "ValueError: invalid literal for int() with base 10: 'oops'", None]}
```

Split the result with an ordinary filter, such as `kept.filter(bt.col("error").is_not_null())` for the rows to inspect or write to a quarantine table. An `error_column` stage runs on threads rather than worker processes, because the kept row is assembled in the process that isolated it.

## Retrying a flaky call

Dropping a row is right for data that can never parse. A call that fails *sometimes*, such as a request to a rate-limited model endpoint, wants a retry instead. `max_retries` retries a batch whose `fn` raises before failing the stage, waiting `retry_backoff * 2**k` seconds before attempt `k`. `retry_on` narrows the retry to the exception types worth retrying, so a genuine bug still fails on the first attempt, and `timeout` caps the wall-clock time of one call.

```python
# docs: skip
scored = docs.map_batches(
    call_endpoint,
    max_retries=3,
    retry_backoff=0.5,
    retry_on=(TimeoutError, ConnectionError),
    timeout=30.0,
)
```

`map`, `flat_map`, and a callable `filter` take the same four options. The retry unit is the batch, not the row: a row callback that fails on one row re-runs every row of that batch, so a side effect in it must be idempotent.

`retry_on=None`, the default, retries any `Exception` once `max_retries` is set. A retry happens inside the worker that saw the failure, so it behaves the same single-node and under `distributed=True`. A failure that survives every retry falls through to `max_errored_rows` if that is set. When the budget is spent the stage raises the function's own exception, with a note saying the `max_errored_rows` allowance was exceeded and how many rows it already dropped.

## How many times your function runs

Batcher promises the rows your function returns, not the calls it makes to get them. Four things are unspecified and may change between runs, releases, and cluster sizes:

- How many calls a stage makes.
- How many rows each call gets. With `batch_size` set, a call gets at most that many, and some get fewer, such as the last batch of a partition.
- Where a batch boundary falls.
- The order in which batches are called, across threads, async tasks, and workers.

What does hold is order within a call. The rows your function returns for a batch keep their place, and a single-node run returns the batches in input order.

A row can also reach your function more than once. These are the cases that cause it:

- **Each terminal operation runs the plan again.** `collect`, `count`, and a write each call the function anew, unless the dataset was cached.
- **`schema` probes an undeclared stage.** It calls a batch function on an empty batch, and a `map` or `flat_map` function, which is never called on an empty batch, on one input row. An `INSERT` into a session table does the same to align types. Declare `output_columns` as a `pyarrow.Schema` and neither calls it.
- **Batcher measures a plain function once.** Over a large input, the first run of a plain function, not a class or a GPU stage, times it on a sample of its first batch to size the batches, and discards the result. The measurement is kept, so later runs skip it.
- **`max_errored_rows` bisects a failing batch.** The halves are called again until the failing rows are isolated, so the rows around a bad one run several times.
- **`max_retries` re-runs the whole batch.** Every row in it is called again, including the rows that succeeded.
- **A preempted worker's partition is recomputed** under `distributed=True`.

So a function whose effect lands outside its return value, such as a write, a POST, or a counter, must be safe to repeat.

## Retries and idempotency

:::{warning}
Under `distributed=True`, a worker that gets preempted mid-batch is reassigned and its partition *recomputed*. So `fn` must be idempotent. A pure transform is safe. A `fn` that POSTs to an API, upserts into a vector DB, or increments an external counter can apply that effect twice. Make the sink idempotent by upserting on a key, or move the side effect out of the UDF and into a `write`.
:::

## See also

- {doc}`User-defined functions <udfs>`: the `map_batches` contract, `input_columns`, the class-per-worker pattern, and `map_groups`.
- {doc}`Spilling </architecture/deep-dives/memory/spilling>`: what the scratch directory this staging needs is otherwise used for.
- {doc}`Inference </ml/inference/inference>`: the class-per-worker pattern with a real model, on real hardware.

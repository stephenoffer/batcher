# Running a UDF at scale

This page covers what changes when a `map_batches` stage runs over a cluster rather than one machine: how a UDF with a pipeline breaker above it is staged, what happens to a device request or a function the cluster cannot take, how to survive a malformed record without losing the job, and the idempotency a distributed retry demands. The callback contract itself is on {doc}`User-defined functions <udfs>`.

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

The budget is one allowance per worker process, whichever way the stage runs: threads, worker processes, or a streamed window all draw down the same count. Across a cluster the honest bound is therefore `workers x max_errored_rows`. Every drop is published to the observability bus with the running total and the error text, so a long job reports the loss while it happens rather than at the end.

The row callbacks take it too. `ds.map`, `ds.flat_map`, and a callable `ds.filter` all lower to a `map_batches` stage, so the same budget isolates a raising callback down to the rows that raised:

```python
def parse_row(row):
    return {"n": int(row["s"])}


rows = bt.from_pydict({"s": ["1", "2", "oops", "4"]})
print(rows.map(parse_row, output_columns=["n"], max_errored_rows=10).to_pydict())
# {'n': [1, 2, 4]}
```

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

`retry_on=None`, the default, retries any `Exception` once `max_retries` is set. A retry happens inside the worker that saw the failure, so it behaves the same single-node and under `distributed=True`. A failure that survives every retry falls through to `max_errored_rows` if that is set. When the budget is spent the stage raises the function's own exception, with a note saying the `max_errored_rows` allowance was exceeded and how many rows it already dropped.

## Retries and idempotency

:::{warning}
Under `distributed=True`, a worker that gets preempted mid-batch is reassigned and its partition *recomputed*. So `fn` must be idempotent. A pure transform is safe. A `fn` that POSTs to an API, upserts into a vector DB, or increments an external counter can apply that effect twice. Make the sink idempotent by upserting on a key, or move the side effect out of the UDF and into a `write`.
:::

## See also

- {doc}`User-defined functions <udfs>`: the `map_batches` contract, `input_columns`, the class-per-worker pattern, and `map_groups`.
- {doc}`Spilling </architecture/deep-dives/memory/spilling>`: what the scratch directory this staging needs is otherwise used for.
- {doc}`Inference </ml/inference/inference>`: the class-per-worker pattern with a real model, on real hardware.

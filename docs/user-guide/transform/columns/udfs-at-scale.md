# Running a UDF at scale

This page covers what changes when a `map_batches` stage runs over a cluster rather than one
machine: how a UDF with a pipeline breaker above it is staged, how to survive a malformed
record without losing the job, and the idempotency a distributed retry demands. The callback
contract itself is on {doc}`User-defined functions <udfs>`.

The examples run against these imports:

```python
import batcher as bt
import pyarrow as pa
```

## Running a UDF pipeline on a cluster

A `map_batches` chain distributes on its own: each worker reads its own splits and runs the
function over them, and nothing passes through the driver.

A UDF with a *breaker* above it needs one more step, because the breaker cannot see through
an opaque Python function to co-partition anything. Batcher runs the UDF as its own
distributed stage, lands its output on cluster-shared scratch, and then dispatches the
breaker over that, so the operator above gets the shuffle, the skew handling and the spill
it always had. Sorts, `distinct`, windows and limits go through this, and so do joins and
unions, where each operand that holds a UDF is staged separately:

```python
# docs: skip
embedded = docs.map_batches(Embedder, num_gpus=1)
enriched = embedded.join(metadata, on="doc_id").group_by("topic").agg(n=bt.col("doc_id").count())
enriched.collect(distributed=True)
```

The staging needs a scratch directory every node can reach. On a cluster with no shared
mount, point `memory.spill_dir` at a shared filesystem; without one Batcher raises rather
than writing files a worker cannot open.

## Tolerating dirty data

A single malformed record should not kill a six-hour job. With `max_errored_rows` set,
a batch whose `fn` raises is bisected to isolate the offending rows, and those rows are
*dropped* up to the budget. Past the budget the error propagates, so a genuine bug on
clean data still fails fast.

```python
raw = bt.from_pydict({"s": ["1", "2", "oops", "4"]})


def parse(batch):
    return pa.RecordBatch.from_pydict({"n": [int(v) for v in batch.column("s").to_pylist()]})


print(raw.map_batches(parse, output_columns=["n"], max_errored_rows=10).to_pydict())
# {'n': [1, 2, 4]}
```

:::{important}
Default is 0 (strict). Set it deliberately and keep it small. A budget of 1,000,000
silently deleted rows is not resilience, it is a deletion policy nobody agreed to.
:::

The budget is one allowance per worker process, whichever way the stage runs: threads,
worker processes, or a streamed window all draw down the same count. Across a cluster the
honest bound is therefore `workers x max_errored_rows`. Every drop is published to the
observability bus with the running total and the error text, so a long job reports the loss
while it happens rather than at the end.

The row callbacks take it too. `ds.map`, `ds.flat_map`, and `ds.ml.filter` all lower to a
`map_batches` stage, so the same budget isolates a raising callback down to the rows that
raised:

```python
def parse_row(row):
    return {"n": int(row["s"])}


rows = bt.from_pydict({"s": ["1", "2", "oops", "4"]})
print(rows.map(parse_row, output_columns=["n"], max_errored_rows=10).to_pydict())
# {'n': [1, 2, 4]}
```

## The distributed caveat

:::{warning}
Under `distributed=True`, a worker that gets preempted mid-batch is reassigned and its
partition *recomputed*. So `fn` must be idempotent. A pure transform is safe. A `fn`
that POSTs to an API, upserts into a vector DB, or increments an external counter can
apply that effect twice. Make the sink idempotent by upserting on a key, or move the side
effect out of the UDF and into a `write`.
:::

## See also

- {doc}`User-defined functions <udfs>`: the `map_batches` contract, `input_columns`, the class-per-worker pattern, and `map_groups`.
- {doc}`Spilling </architecture/deep-dives/memory/spilling>`: what the scratch directory this staging needs is otherwise used for.
- {doc}`Inference </ml/inference/inference>`: the class-per-worker pattern with a real model, on real hardware.

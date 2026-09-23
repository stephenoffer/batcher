# Orchestrators

This section covers running Batcher from a workflow scheduler: Apache Airflow, Dagster, and Prefect.

There is no Batcher operator, plugin, or provider package to install, and that is the whole integration story rather than a gap in it. Batcher is a library with no daemon, no cluster to start, and no session to open, so a task that runs it is a Python function that imports `batcher`. Everything an orchestrator needs from a task — did it succeed, what did it produce, is it safe to retry — comes from ordinary return values.

```python
import batcher as bt


def daily_revenue(raw: bt.Dataset, out: str) -> int:
    """One task body. No engine to start, no connection to close."""
    clean = raw.dq.positive("amount").drop()
    totals = clean.group_by("region").agg(total=bt.col("amount").sum())
    manifest = totals.write.parquet(out, partition_by=["region"], mode="overwrite")
    return manifest.total_rows


rows = daily_revenue(
    bt.from_pydict({"region": ["eu", "us", "eu"], "amount": [120.0, 80.0, -5.0]}),
    "warehouse/revenue",
)
print(rows)
# 2
```

That function is the same object in all three schedulers. What changes between them is the decorator around it and where the return value is recorded.

## The three things a scheduler needs

**Did it succeed?** A failing Batcher call raises a typed exception from {doc}`/api/operations/exceptions`, and every one subclasses `BatcherError`. A data-quality contract you declare with `.fail()` raises `DataQualityError`, so "the data was wrong" and "the job crashed" are distinguishable in a retry rule rather than both arriving as a generic failure.

**What did it produce?** A write returns a {py:obj}`WriteManifest <batcher.io.WriteManifest>` carrying `total_rows`, `num_files`, `total_bytes`, and the file list. Return it, log it, or push it into the scheduler's own metadata store. It is what makes a downstream task's "did anything change" check cheap.

**Is it safe to retry?** This is the question orchestrators actually exist to ask, and the answer is a property of how you write rather than of the scheduler. {doc}`retries-and-idempotency` covers it: save modes, the `_SUCCESS` marker, `replace_where` for a partition backfill, and `merge_on` for a keyed upsert that lands the same rows however many times it runs.

## The pages

| Page | Covers |
| --- | --- |
| {doc}`airflow` | A `@task` body, XCom-sized returns, and why not to put a Dataset in one |
| {doc}`dagster` | Assets and the manifest as materialization metadata |
| {doc}`prefect` | Flows, tasks, and retry policy keyed on the exception type |
| {doc}`retries-and-idempotency` | Making a task safe to run twice, which is the part that is Batcher's job |

## Where the work runs

A task that calls `collect()` runs the query in the worker process that called it, across that machine's cores. A task that calls `collect(distributed=True, num_workers=N)` schedules it across a Ray cluster and returns when the result is back; the scheduler's worker is then a driver holding a handle, not a machine doing the work.

The choice is one argument and it does not change the pipeline, so it is reasonable to leave it as a parameter of the task and decide per environment. {doc}`/user-guide/operate/running/index` covers sizing, and {doc}`/integrations/compute/schedulers` covers what Batcher reads from an HPC or cloud batch allocation when it lands inside one.

## See also

- {doc}`/integrations/compute/schedulers`: Slurm, Kubernetes, and managed job services, which allocate rather than orchestrate.
- {doc}`/user-guide/trust/data-quality`: the contracts that decide whether a task fails.
- {doc}`/user-guide/operate/running/observability`: the events and metrics a run emits while it is going.
- {doc}`/cookbook/data-engineering/ingest/index`: the pipeline bodies these tasks wrap.

```{toctree}
:hidden:

airflow
dagster
prefect
retries-and-idempotency
```

# Apache Airflow

This page covers running a Batcher pipeline from an Airflow task, what to return from it, and what not to.

There is no Batcher provider package. A Batcher task is a `@task` whose body imports `batcher`, and the engine runs inside the worker that Airflow already started.

## A task body

The part worth separating is the pipeline itself, which should be an ordinary function with no Airflow in it. That keeps it runnable in a test, in a notebook, and from the next scheduler you move to:

```python
import batcher as bt


def revenue_by_region(source: str, sink: str) -> dict[str, int]:
    """The pipeline. No Airflow imports, so it runs anywhere."""
    raw = bt.read.parquet(source)
    clean = raw.dq.positive("amount").not_null("region").drop()
    totals = clean.group_by("region").agg(total=bt.col("amount").sum())
    manifest = totals.write.parquet(sink, mode="overwrite")
    return {"rows": manifest.total_rows, "files": manifest.num_files}
```

That function executes here, against a small fixture, exactly as it would against a real prefix:

```python
bt.from_pydict(
    {"region": ["eu", "us", "eu", None], "amount": [120.0, 80.0, 45.0, 10.0]}
).write.parquet("raw/2026-01-01")

print(revenue_by_region("raw/2026-01-01", "warehouse/revenue"))
# {'rows': 2, 'files': 1}
```

The Airflow wrapper is then three lines, and holds no pipeline logic:

```python
# docs: skip
from airflow.decorators import dag, task
import pendulum


@dag(schedule="@daily", start_date=pendulum.datetime(2026, 1, 1), catchup=False)
def revenue():
    @task(retries=2)
    def compute(ds: str) -> dict[str, int]:
        return revenue_by_region(f"s3://lake/raw/day={ds}", f"s3://lake/warehouse/day={ds}")

    compute()


revenue()
```

## Return the manifest, not the data

An XCom value is pickled into the metadata database, so it must stay small. A {py:obj}`WriteManifest <batcher.io.WriteManifest>` summary is a handful of integers and is exactly what a downstream task needs to decide whether anything changed. A `Dataset` is a plan bound to the process that built it and is not a meaningful thing to hand across a task boundary at all; a `pyarrow.Table` is the data itself and will fill the metadata database.

Pass a *path* between tasks and let each one read it. That is the same discipline Airflow already asks for, and Batcher's readers are lazy, so the downstream task pays only for the columns and row groups its query touches.

## Deadlines

Airflow's `execution_timeout` kills the task. If you also export `BATCHER_DEADLINE_SECONDS`, Batcher drains in-flight work and finishes its writes rather than being cut mid-file, which is the difference between a partial output you have to detect and one that was never published. {doc}`/integrations/compute/schedulers` covers the drain path.

## Retries

`retries=2` above is only safe if the task body is idempotent, and that is decided by how the write is spelled rather than by Airflow. `mode="overwrite"` on a per-day prefix is idempotent; a bare append is not. {doc}`retries-and-idempotency` is the full set of options.

## See also

- {doc}`retries-and-idempotency`: making the body safe to run twice.
- {doc}`index`: the same task body in Dagster and Prefect.
- {doc}`/user-guide/moving-data/writing-data`: save modes, partitioned layouts, and the `_SUCCESS` marker.
- {doc}`/user-guide/trust/data-quality`: the contracts that decide whether the task fails.

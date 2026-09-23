# Prefect

This page covers running Batcher from a Prefect flow, and keying a retry policy on the exception a failure actually raises.

As in {doc}`airflow` and {doc}`dagster`, the pipeline is a plain function and the Prefect decorator is the thin part.

## A flow

```python
import batcher as bt


def load_day(events: bt.Dataset, sink: str) -> int:
    clean = events.dq.not_null("user").in_range("amount", 0, 10_000).drop()
    manifest = clean.write.parquet(sink, mode="overwrite", partition_by=["region"])
    return manifest.total_rows


events = bt.from_pydict(
    {
        "user": ["a", "b", None, "d"],
        "region": ["eu", "us", "eu", "us"],
        "amount": [10.0, 20.0, 30.0, 99_999.0],
    }
)
print(load_day(events, "warehouse/day"))
# 2
```

The Prefect wrapper adds scheduling and retries, and no logic:

```python
# docs: skip
from prefect import flow, task


@task(retries=3, retry_delay_seconds=30)
def load(day: str) -> int:
    events = bt.read.parquet(f"s3://lake/events/{day}/")
    return load_day(events, f"s3://lake/warehouse/{day}/")


@flow(name="daily-load")
def daily(day: str):
    rows = load(day)
    print(f"{day}: {rows} rows")
```

## Retry on the right failure

Batcher's exceptions are typed, so a retry policy can tell a transient problem from a permanent one instead of retrying everything three times and failing anyway.

| Exception | Retry? | Why |
| --- | --- | --- |
| {py:exc}`IOError <batcher.IOError>` | Yes | A store or a connection was briefly unavailable |
| {py:exc}`ResourceError <batcher.ResourceError>` | Yes, with more memory | The envelope was too small for this input |
| {py:exc}`DataQualityError <batcher.DataQualityError>` | No | The data is wrong; a retry reads the same rows |
| {py:exc}`PlanError <batcher.PlanError>` | No | The query is wrong; alert instead |
| {py:exc}`AccessDeniedError <batcher.AccessDeniedError>` | No | A credential or a grant is missing |

```python
# docs: skip
@task(retries=3, retry_condition_fn=lambda task, state, ctx: isinstance(state.result(), bt.IOError))
def load(day: str) -> int: ...
```

Every one subclasses `BatcherError`, and several also subclass the matching builtin, so a handler catching `ValueError` or `ImportError` keeps working. {doc}`/api/operations/exceptions` has the full hierarchy.

## Concurrency

A Prefect worker runs tasks concurrently, and each Batcher query already uses every core it is given. Two tasks on one worker will contend, and the symptom is a slowdown rather than an error.

Set a memory envelope per task rather than letting each one size itself against the whole machine. {doc}`/configuration/index` covers `Config`, and {py:obj}`bt.config_context(...) <batcher.config_context>` scopes it to a block so one task's setting does not leak into the next.

## See also

- {doc}`retries-and-idempotency`: making the retry land the same rows.
- {doc}`/api/operations/exceptions`: every typed exception, and what raises it.
- {doc}`/configuration/index`: memory envelopes, profiles, and environment variables.
- {doc}`index`: the same body in Airflow and Dagster.

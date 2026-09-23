# Dagster

This page covers running Batcher inside a Dagster asset, and turning a write manifest into materialization metadata.

Dagster asks a different question from a task scheduler: not "did this step run" but "what does this table currently contain". Batcher answers it cheaply, because a write already reports what it wrote.

## An asset body

Keep the pipeline a plain function, as in {doc}`airflow`, and let the asset be the thin wrapper:

```python
import batcher as bt


def sessionize(events: bt.Dataset) -> bt.Dataset:
    """Group a click stream into sessions with a 30-minute idle gap."""
    return events.with_columns(
        prev=bt.lag(bt.col("ts")).over(partition_by="user", order_by="ts")
    ).with_columns(
        new_session=bt.col("prev").is_null() | ((bt.col("ts") - bt.col("prev")) > 1800)
    )


clicks = bt.from_pydict(
    {
        "user": ["a", "a", "a", "b", "b"],
        "ts": [0, 60, 4000, 0, 120],
    }
)
print(sessionize(clicks).select("user", "ts", "new_session").sort("user", "ts").to_pydict())
# {'user': ['a', 'a', 'a', 'b', 'b'], 'ts': [0, 60, 4000, 0, 120], 'new_session': [True, False, True, True, False]}
```

## The manifest is the metadata

A write returns counts Dagster can record without a second pass over the data, which is what makes the materialization record honest rather than approximate:

```python
sessions = sessionize(clicks)
manifest = sessions.write.parquet("warehouse/sessions", mode="overwrite")
print({"rows": manifest.total_rows, "files": manifest.num_files, "bytes>0": manifest.total_bytes > 0})
# {'rows': 5, 'files': 1, 'bytes>0': True}
```

In Dagster that becomes the asset's metadata:

```python
# docs: skip
from dagster import asset, MaterializeResult, MetadataValue


@asset
def sessions() -> MaterializeResult:
    events = bt.read.parquet("s3://lake/clicks/")
    manifest = sessionize(events).write.parquet("s3://lake/sessions/", mode="overwrite")
    return MaterializeResult(
        metadata={
            "rows": MetadataValue.int(manifest.total_rows),
            "files": MetadataValue.int(manifest.num_files),
            "bytes": MetadataValue.int(manifest.total_bytes),
        }
    )
```

## I/O managers

Dagster's I/O managers exist to move a value between assets. With Batcher you usually want the *path* to move and the data to stay where it is, because the downstream asset's query decides what to read and a lazy reader will prune columns and row groups the query never touches. An I/O manager that materializes a whole table to hand it on undoes that.

Return a path or a manifest from the asset and read it in the next one. Reach for an I/O manager when the asset genuinely produces a small value, such as a metrics row or a validation report.

## Asset checks

A Dagster asset check and a Batcher data-quality contract answer the same question, so run the contract and report its result rather than recomputing it:

```python
report = sessions.dq.not_null("user").row_count_between(1, 1_000_000).validate()
print({"ok": report.ok, "violations": report.total_violations})
# {'ok': True, 'violations': 0}
```

{py:obj}`validate() <batcher.api.dataset.dq.accessor.DatasetDQ.validate>` returns a report instead of raising, which is what a check wants. `.fail()` raises `DataQualityError` instead, which is what a pipeline wants. {doc}`/user-guide/trust/data-quality` covers the trichotomy of fail, drop, and quarantine.

## See also

- {doc}`retries-and-idempotency`: making a re-materialization land the same rows.
- {doc}`index`: the same body in Airflow and Prefect.
- {doc}`/user-guide/trust/data-quality`: contracts, reports, and quarantine.
- {doc}`/cookbook/analytics/behavior/sessionization`: the sessionization recipe at full length.

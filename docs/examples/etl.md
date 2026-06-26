# ETL pipeline

A small but complete extract-transform-load: read raw records, clean and
deduplicate them, derive columns, roll them up, and write the result. Every block
runs as written; swap `from_pydict` for `bt.read(...)` and the same code scales to
files or object storage.

## Extract

Start from raw records — here with duplicates, mixed-case keys, and a null.

```python
import batcher as bt
from batcher import col

raw = bt.from_pydict(
    {
        "id": [1, 2, 2, 3, 4],
        "region": ["West", "west", "EAST", None, "East"],
        "amount": [100.0, 100.0, 250.0, 80.0, 300.0],
        "ts": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"],
    }
)
print(raw.count())
# 5
```

## Transform

Deduplicate to one row per `id` (keeping the earliest by `ts`), normalize the key,
and fill the missing region. Each step returns a new lazy `Dataset`; nothing runs
yet.

```python
clean = (
    raw.distinct(["id"], keep="first", order_by="ts")
    .with_columns(region=col("region").str.upper())
    .fill_null({"region": "UNKNOWN"})
)
print(clean.sort("id").to_pydict()["region"])
# ['WEST', 'WEST', 'UNKNOWN', 'EAST']
```

Roll up to revenue per region.

```python
rollup = (
    clean.group_by("region")
    .agg(orders=bt.count(), revenue=col("amount").sum())
    .sort("revenue", descending=True)
)
print(rollup.to_pydict())
# {'region': ['EAST', 'WEST', 'UNKNOWN'], 'orders': [1, 2, 1], 'revenue': [300.0, 200.0, 80.0]}
```

## Load

Write the result to Parquet and read it back. The write is the terminal operation
that executes the whole plan above.

```python
import tempfile, os

out = os.path.join(tempfile.mkdtemp(), "rollup")
rollup.write.parquet(out)

back = bt.read.parquet(out)
print(back.count())
# 3
```

At scale, only the endpoints change: `bt.read("s3://bucket/raw/*.parquet")` in,
`rollup.write.parquet("s3://bucket/curated/", partition_by=["region"])` out, and
`collect(distributed=True)` to run it across a cluster — the transform is identical.

## Next steps

- [Data quality](../user-guide/data-quality.md): turn the cleaning step into an
  enforced contract — validate, quarantine, or fail on bad rows.
- [Lakehouse tables](../user-guide/lakehouse.md): write to Delta with merge/upsert
  and slowly-changing dimensions instead of plain Parquet.
- [Performance and memory](../user-guide/performance.md): cache and spill as the
  data grows.

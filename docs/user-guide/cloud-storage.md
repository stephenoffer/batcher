# Cloud storage

Batcher reads and writes object stores through the same API as local files. A path
with a cloud scheme (`s3://`, `gs://`, `az://`) is recognized automatically; only
the path changes, the rest of the pipeline is identical.

Every example on this page needs a real bucket and credentials, so the blocks are
shown but not executed.

## Reading from object storage

`bt.read` infers the format from the extension. The format-specific readers
(`bt.read.parquet`, `bt.read.csv`, `bt.read.json`) take the same cloud paths.

```python
# docs: skip
import batcher as bt

ds = bt.read("s3://bucket/events/*.parquet")
out = ds.filter(bt.col("status") == "active").select("user_id", "amount")
print(out.to_pydict())
```

A glob reads many files as one Dataset. Reading stays lazy: no bytes are fetched
until a terminal operation runs, and projection and filter pushdown limit what is
read.

```python
# docs: skip
ds = bt.read.parquet("s3://bucket/year=2024/month=06/*.parquet")
```

## Credentials

Credentials are read from the environment, following the conventions of each
provider's SDK. Set them before importing Batcher or starting your process.

```python
# docs: skip
import os

os.environ["AWS_ACCESS_KEY_ID"] = "..."
os.environ["AWS_SECRET_ACCESS_KEY"] = "..."
os.environ["AWS_REGION"] = "us-east-1"

import batcher as bt

ds = bt.read("s3://bucket/events.parquet")
```

For Google Cloud Storage, set `GOOGLE_APPLICATION_CREDENTIALS` to a service-account
JSON path. For Azure, set the account name and key the Azure SDK expects.

## Writing to object storage

Write helpers take cloud paths as well. Combine with `partition_by` to lay out a
partitioned dataset, and `distributed=True` to write across workers.

```python
# docs: skip
ds.write.parquet("s3://bucket/curated/events.parquet")
ds.write("s3://bucket/curated/events", fmt="parquet", partition_by=["region"])
```

## Working at scale

Large cloud datasets are split into tasks so the driver never has to materialize a
whole file. For distributed reads, the data plane moves Arrow batches directly
between workers over Arrow Flight rather than through a scheduler's object store,
which keeps per-node memory bounded.

```python
# docs: skip
ds = bt.read("s3://bucket/huge/*.parquet")
result = ds.group_by("region").agg(total=bt.col("amount").sum()).collect(
    distributed=True, num_workers=16
)
```

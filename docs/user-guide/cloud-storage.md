# Cloud storage

Batcher reads and writes object stores through the same API as local files. A path
with a cloud scheme is recognized automatically; only the path changes, the rest of
the pipeline is identical. The accepted schemes are `s3://` / `s3a://`, `gs://` /
`gcs://`, `az://` / `abfs://` / `abfss://` / `wasb://` / `wasbs://`, and `hdfs://`,
plus `file://` and bare local paths. Reading and writing both go through one
`pyarrow.fs`-backed filesystem; anything pyarrow does not implement natively falls back
to fsspec behind the same interface.

Object-store access needs the cloud extra:

```
pip install 'batcher-engine[cloud]'
```

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
provider's SDK — the same variables the AWS/GCP/Azure tooling already uses. Set them
before starting your process. On-prem and self-hosted stores (MinIO, Ceph) are S3
endpoints: point at them with `AWS_ENDPOINT_URL`, or per-path with an
`endpoint_override` in the URI query string.

| Store | Environment variables / settings |
| --- | --- |
| S3 | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` (if temporary), `AWS_REGION` / `AWS_DEFAULT_REGION`. Instance/role credentials are picked up automatically when set. |
| S3-compatible (MinIO, Ceph) | The S3 variables above, plus `AWS_ENDPOINT_URL=https://minio.internal:9000` (or `?endpoint_override=...` in the path). |
| GCS | `GOOGLE_APPLICATION_CREDENTIALS` pointing at a service-account JSON, or workload-identity / application-default credentials. |
| Azure (`abfs`/`az`) | `AZURE_STORAGE_ACCOUNT_NAME` plus one of `AZURE_STORAGE_ACCOUNT_KEY`, `AZURE_STORAGE_SAS_TOKEN`, or AAD service-principal variables (`AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_CLIENT_SECRET`). |
| HDFS | `hdfs://namenode:8020/path`, with `HADOOP_HOME` / `CLASSPATH` set so the native client and `core-site.xml` are found. |

```python
# docs: skip
import os

os.environ["AWS_ACCESS_KEY_ID"] = "..."
os.environ["AWS_SECRET_ACCESS_KEY"] = "..."
os.environ["AWS_REGION"] = "us-east-1"

import batcher as bt

ds = bt.read("s3://bucket/events.parquet")
```

```python
# docs: skip
# MinIO / on-prem S3: override the endpoint, either by env or in the path.
ds = bt.read("s3://bucket/data/*.parquet?endpoint_override=https://minio.internal:9000")
```

For Delta tables read through delta-rs, credentials can also be passed explicitly as
`storage_options` instead of through the environment — the keys are delta-rs's own
(`aws_access_key_id`, `aws_secret_access_key`, `azure_storage_account_key`,
`google_service_account_token`, and so on):

```python
# docs: skip
ds = bt.read.delta(
    "s3://bucket/table",
    storage_options={"aws_access_key_id": "...", "aws_secret_access_key": "..."},
)
```

If a cloud scheme is used without the `[cloud]` extra installed, the read fails with a
clear message telling you to install it.

## Writing to object storage

Write helpers take cloud paths as well. Combine with `partition_by` to lay out a
partitioned dataset, and `distributed=True` to write across workers. Writes to an
object store go straight to the destination — a single PUT is atomic, so there is no
truncated-file window; local and HDFS writes use temp-then-rename for the same
guarantee.

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

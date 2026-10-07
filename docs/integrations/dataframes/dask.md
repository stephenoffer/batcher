# Dask

This page covers handing data between Batcher and Dask. {py:func}`bt.from_dask <batcher.from_dask>` reads a `dask.dataframe.DataFrame` one partition at a time, and {py:meth}`ds.to_dask() <batcher.Dataset.to_dask>` returns a lazy Dask frame built with `dd.from_map` over Arrow partitions. Neither direction builds one pandas frame of the whole table.

:::{warning}
`to_dask` is not yet verified against a live Dask install; see tests/PENDING_VERIFICATION.md.
:::

Both directions need `pip install 'batcher-engine[dask]'`.

## Export a result

`to_dask` converts each partition to pandas inside its own Dask task. The `materialize` argument says when and where the Batcher query runs:

| Policy | When the query runs | Where the result lives | Use it for |
| --- | --- | --- | --- |
| `"arrow"`, the default | now, once | Arrow partitions of about `partition_bytes`, 128 MiB by default, in this process | a result that fits in memory |
| `"deferred"` | when Dask computes each partition | nowhere until then | keeping the whole pipeline lazy |
| `"parquet"` | now, once | Parquet under `staging_path`, read by `dd.read_parquet` | a result larger than memory, or a Dask cluster |

```python
# docs: skip
import batcher as bt

ds = bt.from_pydict({"city": ["Oslo", "Lima", "Oslo"], "temp": [3.5, 19.0, 5.5]})
frame = ds.to_dask()
print(frame.groupby("city")["temp"].mean().compute().to_dict())
# {'Lima': 19.0, 'Oslo': 4.5}

lazy = ds.to_dask(materialize="deferred", npartitions=4)
staged = ds.to_dask(materialize="parquet", staging_path="s3://<bucket>/handoff")
```

Under `"deferred"`, each of `npartitions` tasks runs the query and keeps the rows whose content hash falls in its bucket. The input is scanned once per partition, which is the price of running nothing up front. Rows are bucketed by their non-nested columns, so equal rows share a partition, and a table whose every column is nested can only be deferred as one partition.

Under `"parquet"`, the staged directory is named `to_dask-<id>` and isn't removed, because Dask reads it lazily. Pass a `staging_path` the Dask workers can read, such as `s3://<bucket>/<prefix>`, and delete the directory when the frame is no longer used.

## Import a frame

```python
# docs: skip
import dask.dataframe as dd

import batcher as bt

ddf = dd.read_csv("s3://<bucket>/events-*.csv")
events = bt.from_dask(ddf)
print(events.count())
```

`from_dask` computes one Dask partition per engine batch, so the frame is streamed rather than collected.

## See also

- {doc}`pandas`: the in-process pandas conversion.
- {doc}`/integrations/compute/ray`: the same hand-off to Ray Data.
- {doc}`/api/symbols/dataset-terminal`: every terminal method, with signatures.

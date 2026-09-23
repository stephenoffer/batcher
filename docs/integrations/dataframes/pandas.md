# pandas

This page covers moving data between pandas and Batcher, and the two pandas concepts that have no counterpart here.

pandas is the one library in this section that is not Arrow-native, so it is the one where the conversion is not free. A pandas `DataFrame` holds NumPy blocks, and Arrow has to be built from them, which costs a pass over the data in both directions.

| | |
| --- | --- |
| **Read** | {py:obj}`bt.from_pandas(df) <batcher.from_pandas>` |
| **Write** | {py:obj}`ds.to_pandas() <batcher.Dataset.to_pandas>` |
| **Extra** | `pandas` |
| **Cost** | A conversion, not a handoff. Arrow-backed pandas columns are cheaper than NumPy-backed ones |

## Round trip

```python
# docs: skip
import batcher as bt
import pandas as pd

frame = pd.DataFrame({"user": ["a", "b", "a"], "spend": [10.0, 2.0, 7.0]})
totals = bt.from_pandas(frame).group_by("user").agg(total=bt.col("spend").sum()).sort("user")

print(totals.to_pandas())
#   user  total
# 0    a   17.0
# 1    b    2.0
```

The blocks on this page are shown rather than executed, because the test suite that runs the documentation installs the extras CI needs and pandas is not one of them. Everything here works the same way as the Polars and Arrow pages, which are executed on every run.

## Where the conversion cost goes

Object-dtype columns are the expensive case. A pandas column of Python strings is an array of pointers, and Arrow needs one contiguous buffer plus offsets, so the conversion walks every value. Numeric columns are close to a memcpy, and a pandas frame already backed by Arrow (`dtype_backend="pyarrow"`) is close to free.

If a pipeline crosses this boundary in a loop, the fix is usually to move the boundary rather than to speed it up: read the source with `bt.read.*` instead of `pd.read_*`, do the work in Batcher, and convert once at the end.

## The index does not exist here

A Batcher `Dataset` has columns and nothing else. There is no index, so `from_pandas` keeps the index only if you have made it a column first:

```python
# docs: skip
ds = bt.from_pandas(frame.reset_index())
```

This is the same decision Polars made, and for the same reason: an index is a second addressing scheme that every operator has to agree about, and joins and group-bys already say what they key on. To get row positions back, {py:obj}`ds.with_row_index() <batcher.Dataset.with_row_index>` adds an explicit column.

## `apply` has no row form

`df.apply(fn, axis=1)` calls a Python function once per row. Batcher's UDF contract is per Arrow batch, deliberately: a per-row Python call in the data plane is the overhead the engine exists to remove, and a batch callback with a thousand rows in it amortizes the interpreter to nothing.

```python
# docs: skip
import pyarrow.compute as pc


def shout(batch):
    return batch.set_column(0, "user", pc.utf8_upper(batch.column("user")))


loud = bt.from_pandas(frame).map_batches(shout)
```

Most `apply` bodies turn out to be an expression rather than a callback. {doc}`/getting-started/migration/transforming` has the verb-by-verb table, and {doc}`/user-guide/transform/columns/udfs` covers the batch contract when an expression genuinely will not do.

## See also

- {doc}`/getting-started/migration/transforming`: the pandas-to-Batcher verb table, and the eager-to-lazy shift.
- {doc}`polars`: the Arrow-native alternative, where the same round trip costs nothing.
- {doc}`arrow-and-numpy`: the zero-copy layer underneath.
- {doc}`/user-guide/transform/columns/udfs`: the batch UDF, and when to reach for one.

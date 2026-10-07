# DataFrames and arrays

This section covers the libraries that run in the same Python process as Batcher: Polars, pandas, DuckDB, PyArrow, and NumPy. Batcher holds its data as Apache Arrow, and so do all of these, so moving a table between them is a pointer handoff through the [Arrow C Data Interface](https://arrow.apache.org/docs/format/CDataInterface.html) whose cost doesn't grow with the row count.

```python
import batcher as bt
import polars as pl

frame = pl.DataFrame({"city": ["Oslo", "Lima", "Oslo"], "temp": [3.5, 19.0, 5.5]})
warm = bt.from_polars(frame).group_by("city").agg(avg=bt.col("temp").mean()).sort("city")
print(warm.to_polars())
# shape: (2, 2)
# ┌──────┬──────┐
# │ city ┆ avg  │
# │ ---  ┆ ---  │
# │ str  ┆ f64  │
# ╞══════╪══════╡
# │ Lima ┆ 19.0 │
# │ Oslo ┆ 4.5  │
# └──────┴──────┘
```

Neither call copied the columns, so you can use Batcher for one step of a pipeline written in something else and hand the answer straight back. pandas and NumPy work the same way:

```python
import pandas as pd

df = pd.DataFrame({"city": ["Oslo", "Lima"], "temp": [3.5, 19.0]})
print(bt.from_pandas(df).filter(bt.col("temp") > 10).to_pandas().to_dict("list"))
# {'city': ['Lima'], 'temp': [19.0]}
print(bt.from_pandas(df).to_numpy()["temp"])
# [ 3.5 19. ]
```

## The libraries

| Page | Covers |
| --- | --- |
| {doc}`polars` | `from_polars` and `to_polars`, and which engine to reach for |
| {doc}`pandas` | `from_pandas` and `to_pandas`, and the index Batcher does not have |
| {doc}`duckdb` | `from_duckdb`, running Batcher SQL beside DuckDB SQL, and DuckDB as the correctness oracle |
| {doc}`dask` | `to_dask`, a lazy Dask frame over Arrow partitions, and `from_dask` |
| {doc}`arrow-and-numpy` | The zero-copy contract itself: `from_arrow`, `to_arrow`, `from_numpy`, `to_numpy`, and tensor columns |

Three more adapters take a distributed frame rather than a local one, and they live with the system they belong to: {py:obj}`bt.from_ray_dataset <batcher.from_ray_dataset>` and {py:obj}`ds.to_ray_dataset() <batcher.Dataset.to_ray_dataset>` on {doc}`/integrations/compute/ray`, plus `from_spark`/`to_spark` and `from_daft`/`to_daft`, which {doc}`/getting-started/migration/index` covers alongside the porting tables.

## When you don't want to name the library

{py:obj}`bt.from_any <batcher.from_any>` dispatches on the type of whatever you hand it, which is what you want in a helper that accepts a frame from a caller you don't control. It also accepts any object exporting the DataFrame interchange protocol (`__dataframe__`), so a library this section doesn't list still converts.

```python
import pyarrow as pa

print(bt.from_any(pa.table({"a": [1, 2, 3]})).count())
# 3
print(bt.from_any({"a": [1, 2], "b": ["x", "y"]}).columns)
# ['a', 'b']
```

## See also

- {doc}`/getting-started/migration/index`: the verb-by-verb translation, when you are porting rather than interoperating.
- {doc}`/api/symbols/construction`: every constructor, with signatures.
- {doc}`/user-guide/moving-data/reading-data`: reading from a path rather than from another library's object.
- {doc}`/architecture/deep-dives/memory/arrow-memory`: what "zero-copy" means in buffers.

```{toctree}
:hidden:

polars
pandas
duckdb
dask
arrow-and-numpy
```

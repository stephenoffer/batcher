# DataFrames and arrays

This section covers the libraries that run in the same Python process as Batcher: Polars, pandas, DuckDB, PyArrow, and NumPy. They are a different kind of integration from the connectors in the rest of this section. A connector reaches a system over a network and reads it in splits. These hand a buffer across a function call.

The whole section rests on one fact. Batcher holds its data as Apache Arrow, and so do all of these, either natively or through a documented export. Moving a table between them is a pointer handoff through the [Arrow C Data Interface](https://arrow.apache.org/docs/format/CDataInterface.html), not a serialization step, so the cost does not grow with the number of rows.

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

Neither call copied the columns. That is what makes it reasonable to reach for Batcher for one step of a pipeline written in something else, and to hand the answer straight back.

## The libraries

| Page | Covers |
| --- | --- |
| {doc}`polars` | `from_polars` and `to_polars`, and which engine to reach for |
| {doc}`pandas` | `from_pandas` and `to_pandas`, and the index Batcher does not have |
| {doc}`duckdb` | `from_duckdb`, running Batcher SQL beside DuckDB SQL, and DuckDB as the correctness oracle |
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
arrow-and-numpy
```

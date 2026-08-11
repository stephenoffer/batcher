# Differences and verification

This page covers the two things a port needs after the verbs translate: which familiar
APIs Batcher deliberately does not have, and how to prove the ported script returns the
same rows as the original.

## What Batcher deliberately does not have

Some familiar APIs are absent by design rather than by omission, and knowing which is
which saves you looking for a workaround that doesn't exist. Batcher tells you at the
point of use: every one of these raises an `AttributeError` naming the reason and the
replacement, so you can discover the mapping from a traceback instead of this table.

| Absent | Why | Instead |
|---|---|---|
| `df.set_index`, `df.reset_index`, `df.loc`, `df.iloc` | A relation is an unordered multiset with no row index, as in SQL. | {py:meth}`ds.filter(...) <batcher.Dataset.filter>`, {py:meth}`ds.select(...) <batcher.Dataset.select>`, {py:meth}`ds.sort(...) <batcher.Dataset.sort>`, {py:meth}`ds.with_row_index() <batcher.Dataset.with_row_index>` |
| `df.iterrows`, `df.itertuples`, `df.applymap` | Per-row Python never runs on the hot path. | {py:meth}`ds.iter_rows(named=True) <batcher.Dataset.iter_rows>` at the end of a pipeline; expressions or {py:meth}`ds.map_batches() <batcher.Dataset.map_batches>` inside one |
| `df.apply` | Its per-row and per-column meanings don't survive a columnar engine. | {py:meth}`ds.with_columns(y=expr) <batcher.Dataset.with_columns>` or {py:meth}`ds.map_batches(fn) <batcher.Dataset.map_batches>` |
| `df.T`, `df.transpose` | Transposing needs a materialized, single-typed frame. | `ds.to_pandas().T`, or {py:meth}`ds.unpivot() <batcher.Dataset.unpivot>` / {py:meth}`ds.pivot() <batcher.Dataset.pivot>` |
| `df.shift`, `df.diff`, `df.cumsum`, `df.rolling` | Each needs a row order the relation doesn't carry. | {py:meth}`ds.window(order_by=[...], functions={...}) <batcher.Dataset.window>` |
| `df.resample` | Time bucketing is a grouping. | {py:meth}`ds.group_by(bucket=bt.col("t").dt.truncate("1h")).agg(...) <batcher.Dataset.group_by>` |
| Looping over a {py:class}`GroupBy <batcher.GroupBy>` | It materializes one frame per key in Python and caps the job at one machine. | `.agg(...)`, or `.window(partition_by=[...])` to keep every row |

Column attribute access (`df.amount`) is absent for a subtler reason: a column named
`filter` or `join` would shadow a method, which is a real source of pandas bugs. Use
`ds["amount"]` for the expression, or {py:func}`bt.col("amount") <batcher.col>` to build one.

## The error messages teach you the mapping

You don't have to memorize the translation tables at all. Type the method you already know, and the
traceback tells you the Batcher spelling. This works at every level: on a {py:class}`Dataset <batcher.Dataset>`, on
an expression, on a `GroupBy`, and on the `bt` package itself.

```python
import batcher as bt

demo = bt.from_pydict({"x": [1, 2, 3], "k": ["a", "b", "a"]})

# A pandas reshape on a Dataset:
try:
    demo.pivot_table
except AttributeError as exc:
    assert "ds.pivot" in str(exc)

# A Polars per-element UDF on an expression:
try:
    bt.col("x").map_elements
except AttributeError as exc:
    assert "map_batches" in str(exc)

# A pandas GroupBy transform:
try:
    demo.group_by("k").transform
except AttributeError as exc:
    assert "ds.window" in str(exc)

# A Polars top-level constructor:
try:
    bt.LazyFrame
except AttributeError as exc:
    assert "already lazy" in str(exc)

print("every wrong spelling names its Batcher replacement")
# every wrong spelling names its Batcher replacement
```

A near miss on a real method gets a `Did you mean ...?` suggestion instead, so a typo
such as `ds.filtr` or `bt.col("x").meen` points straight at `filter` and `mean`.

## Checking a port

{py:meth}`ds.equals(other) <batcher.Dataset.equals>` compares *results*, not plans, so it answers the only question that
matters after a migration. Row order is ignored by default, because a relation is
unordered. Pass `ordered=True` after a `sort` when the emitted order is part of the
contract.

```python
ds = bt.from_pydict({"status": ["paid", "open", "paid"], "amount": [10, 20, 30]})

ported = ds.filter(status="paid")
expected = ds.filter(bt.col("status") == "paid")
print(ported.equals(expected))
# True
```

## Requirements and limitations

- {py:func}`from_pandas <batcher.from_pandas>`, {py:func}`from_polars <batcher.from_polars>`, {py:func}`from_spark <batcher.from_spark>`, {py:func}`from_dask <batcher.from_dask>`, {py:func}`from_ray_dataset <batcher.from_ray_dataset>`,
  {py:func}`from_huggingface <batcher.from_huggingface>`, {py:func}`from_torch <batcher.from_torch>`, and {py:func}`from_tf <batcher.from_tf>` each need the source framework
  installed. Batcher doesn't depend on any of them.
- Several source systems have a constructor but no exporter. NumPy, Ray Data, Spark,
  Dask, and HuggingFace are one-way, so round-trip through {py:meth}`to_arrow <batcher.Dataset.to_arrow>` or {py:meth}`to_pandas <batcher.Dataset.to_pandas>`.
- `append` mode is accepted by lakehouse sinks only.
- `merge_on` is a `write.delta` parameter. It has no equivalent on a plain Parquet
  write.
- Distributed execution and the GPU actor pools need the optional `[ray]` extra.
- LLM generation needs a text-generation engine you install separately, such as
  `batcher-engine[vllm]`.

## See also

- {doc}`/getting-started/migration/transforming`: the replacements for most of the absent APIs above.
- {doc}`/agents`: the migration skills, each ending in this verification step.
- {doc}`/user-guide/operate/running/troubleshooting`: diagnosing a ported query that runs but misbehaves.

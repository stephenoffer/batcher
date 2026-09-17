# Differences and verification

A port isn't finished when the verbs translate. This page covers the concepts that change whichever engine you come from, the familiar APIs Batcher deliberately doesn't have, and how to prove the ported script returns the same rows as the original. The name-by-name differences for PySpark, Polars, Daft, and Ray Data are in the generated reference: {doc}`spark/index`, {doc}`polars/index`, {doc}`daft/index`, and {doc}`ray-data/index`.

## A dataset is lazy

A {py:class}`Dataset <batcher.Dataset>` is a plan, not data. Transformations such as `filter`, `with_columns`, and `join` return a new `Dataset` and run nothing, and the plan runs only at a terminal operation such as `collect`, `to_arrow`, `to_pandas`, `count`, `iter_batches`, or a write. This is the Polars `LazyFrame` and Spark `DataFrame` model. Coming from pandas or an eager Polars `DataFrame`, a line that used to compute a value now extends a plan, so a print that relied on an eager value needs an explicit terminal call.

```python
import batcher as bt

ds = bt.from_pydict({"x": [1, 2, 3]})
plan = ds.filter(bt.col("x") > 1)  # nothing has run yet
print(plan.count())
# 2
```

## One spelling per capability

Batcher keeps one spelling for each capability. Where another engine uses a different name, Batcher doesn't add that name as a second spelling. Where two engines give one name different meanings, the generated reference records the difference as a mismatch, and the planned fix is a parameter on the Batcher spelling rather than a second function. Batcher also removed the second spellings it used to accept, such as `groupby` for `group_by` and `fillna` for `fill_null`.

A removed spelling raises `AttributeError`, and so does another engine's spelling typed on a Batcher object. When the migration registry knows the name, the message names the spelling to use. The codemod rewrites a script that still uses a removed Batcher spelling. It prints a diff and changes nothing unless you pass `--write`, and `--check` exits 1 when any file would change:

```bash
python -m batcher.migrate --from batcher --to batcher <paths>
```

Replace `<paths>` with the files or directories to rewrite. Each engine's page in the generated reference says whether its codemod direction is implemented.

## Write modes have different defaults

A write that ports with its name unchanged can still behave differently, because the default save mode differs between engines. Batcher's `ds.write(path)` overwrites existing output by default, and `ds.write.delta(uri)` appends. The following table lists the defaults the migration registry records for the other engines:

| Engine | Writer | Default when the target exists |
|---|---|---|
| PySpark | `DataFrame.write` | raise an error (`errorifexists`) |
| Polars | `DataFrame.write_delta` | raise an error |
| Daft | `write_parquet`, `write_csv` | append |
| Ray Data | `write_parquet`, `write_csv` | append |

Pass `mode=` explicitly on every ported write, so the port doesn't depend on either default. Batcher's file sinks reject `append`, so a Parquet or CSV write that appended in Daft or Ray Data needs a different layout here, such as a lakehouse table.

## Integer overflow wraps

Integer arithmetic in Batcher wraps on overflow instead of raising or returning null, so adding 1 to the largest 64-bit integer returns the smallest one:

```python
big = bt.from_pydict({"x": [2**63 - 1]})
print(big.select(y=bt.col("x") + 1).to_pydict())
# {'y': [-9223372036854775808]}
```

An integer `sum` that overflows raises an `ExecutionError` instead, with a message telling you to cast the column to a wider type first. Spark's `try_add`, `try_subtract`, and `try_multiply`, which return null on overflow, have no Batcher equivalent yet. Batcher also stores unsigned 64-bit integers as signed 64-bit integers, so a `UInt64` value above `2**63 - 1` overflows on the way in.


## What Batcher deliberately does not have

Some familiar APIs are absent by design. You don't need this table to find out which: every one of them raises an `AttributeError` that names the reason and the replacement, so the traceback carries the mapping.

| Absent | Why | Instead |
|---|---|---|
| `df.set_index`, `df.reset_index`, `df.loc`, `df.iloc` | A relation is an unordered multiset with no row index, as in SQL. | {py:meth}`ds.filter(...) <batcher.Dataset.filter>`, {py:meth}`ds.select(...) <batcher.Dataset.select>`, {py:meth}`ds.sort(...) <batcher.Dataset.sort>`, {py:meth}`ds.with_row_index() <batcher.Dataset.with_row_index>` |
| `df.iterrows`, `df.itertuples`, `df.applymap` | Per-row Python never runs on the hot path. | {py:meth}`ds.iter_rows(named=True) <batcher.Dataset.iter_rows>` at the end of a pipeline; expressions or {py:meth}`ds.map_batches() <batcher.Dataset.map_batches>` inside one |
| `df.apply` | Its per-row and per-column meanings don't survive a columnar engine. | {py:meth}`ds.with_columns(y=expr) <batcher.Dataset.with_columns>` or {py:meth}`ds.map_batches(fn) <batcher.Dataset.map_batches>` |
| `df.T`, `df.transpose` | Transposing needs a materialized, single-typed frame. | `ds.to_pandas().T`, or {py:meth}`ds.unpivot() <batcher.Dataset.unpivot>` / {py:meth}`ds.pivot() <batcher.Dataset.pivot>` |
| `df.shift`, `df.diff`, `df.cumsum`, `df.rolling` | Each needs a row order the relation doesn't carry. | {py:meth}`ds.window(order_by=[...], functions={...}) <batcher.Dataset.window>` |
| `df.resample` | Time bucketing is a grouping. | {py:meth}`ds.group_by(bucket=bt.window(bt.col("t"), "1h")).agg(...) <batcher.Dataset.group_by>` |
| Looping over a {py:class}`GroupBy <batcher.GroupBy>` | It materializes one frame per key in Python and caps the job at one machine. | `.agg(...)`, or `.window(partition_by=[...])` to keep every row |

Column attribute access such as `df.amount` is absent for a subtler reason. A column
named `filter` or `join` would shadow a method, which is a real source of pandas bugs.
Use `ds["amount"]` for the expression, or {py:func}`bt.col("amount") <batcher.col>` to build one.

## The error messages teach you the mapping

You don't have to memorize the translation tables. Type the method you already know, and
the traceback tells you the Batcher spelling. This works at every level: on a
{py:class}`Dataset <batcher.Dataset>`, on an expression, on a `GroupBy`, and on the `bt`
package itself.

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

{py:meth}`ds.equals(other) <batcher.Dataset.equals>` compares *results* rather than plans, which is the question a migration raises. Both sides execute and their rows are compared. Two queries built from completely different verbs count as equal when they agree, and a plan-shape comparison can't tell you that.

Column names and types must match. Row order is ignored by default, because a relation is unordered. After a `sort`, pass `ordered=True` when the emitted order is part of the contract.

```python
ds = bt.from_pydict({"status": ["paid", "open", "paid"], "amount": [10, 20, 30]})

ported = ds.filter(status="paid")
expected = ds.filter(bt.col("status") == "paid")
print(ported.equals(expected))
# True
```

## Requirements and limitations

- {py:func}`from_pandas <batcher.from_pandas>`, {py:func}`from_polars <batcher.from_polars>`, {py:func}`from_spark <batcher.from_spark>`, {py:func}`from_daft <batcher.from_daft>`, {py:func}`from_dask <batcher.from_dask>`, {py:func}`from_ray_dataset <batcher.from_ray_dataset>`,
  {py:func}`from_huggingface <batcher.from_huggingface>`, {py:func}`from_torch <batcher.from_torch>`, and {py:func}`from_tf <batcher.from_tf>` each need the source framework
  installed. Batcher doesn't depend on any of them.
- Dask and HuggingFace have a constructor but no exporter. To hand a result back to one of them, go through {py:meth}`to_arrow <batcher.Dataset.to_arrow>` or {py:meth}`to_pandas <batcher.Dataset.to_pandas>`.
- `append` mode is accepted by lakehouse sinks only.
- `merge_on` is a `write.delta` parameter. It has no equivalent on a plain Parquet
  write.
- Distributed execution and the GPU actor pools need the optional `[ray]` extra.
- LLM generation needs a text-generation engine you install separately, such as
  `batcher-engine[vllm]`.

## See also

- {doc}`/getting-started/migration/transforming`: the replacements for most of the absent APIs above.
- {doc}`/getting-started/migration/spark/index`, {doc}`/getting-started/migration/polars/index`, {doc}`/getting-started/migration/daft/index`, {doc}`/getting-started/migration/ray-data/index`: every name in each engine, with its Batcher spelling and what differs.
- {doc}`/agents`: the migration skills, each ending in this verification step.
- {doc}`/user-guide/operate/running/troubleshooting`: diagnosing a ported query that runs but misbehaves.
